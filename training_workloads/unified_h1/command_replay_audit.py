"""Offline replay of command timing, exposure and first-flight outcomes."""
import json
import math
from statistics import fmean

from tools.r6_evidence import checked, file_pin
from .command_replay import CONDITIONS, LANES, LAYOUTS, STEP_MS, load_stream, sampled_events

# Development diagnostics. The corridor is fixed to the authored crossing centre.
CORRIDOR_HALF_WIDTH_M = .58
INTERVENTION_BETA = .98
SETTLED_SPEED_M_S = .07
STABLE_STEPS = 5


def dot(a, b):
    return sum(x*y for x, y in zip(a, b, strict=True))


def body_to_world(q, v):
    scalar, x, y, z = q
    cross = (y*v[2]-z*v[1], z*v[0]-x*v[2], x*v[1]-y*v[0])
    second = (y*cross[2]-z*cross[1], z*cross[0]-x*cross[2], x*cross[1]-y*cross[0])
    return [a+2*(scalar*b+c) for a, b, c in zip(v, cross, second, strict=True)]


def first_stable(rows, predicate):
    count = 0
    for i, row in enumerate(rows):
        count = count+1 if predicate(row) else 0
        if count == STABLE_STEPS:
            return i-count+1
    return None


def score_lane(rows, initial, lane, crossing):
    def read(row, name):
        return row[name][lane]

    def command(row):
        return row['raw_event']['velocity_w_m_s']

    def progress(row, vector):
        intent = command(row)
        return dot(intent, vector)/math.hypot(*intent[:2])

    def position(row):
        return read(row, 'root_state_after_w')[:3]

    def actual(row):
        return read(row, 'root_state_after_w')[7:10]

    def intervened(row):
        return (read(row, 'ics_emergency') or read(row, 'ics_beta') < INTERVENTION_BETA
                or read(row, 'escape_active') or read(row, 'escape_recovering'))

    def recovery_window(start):
        latest = flight[start]['event_index']
        window = []
        for row in flight[start:]:
            if row['event_index'] != latest or not math.hypot(*command(row)[:2]) or read(row, 'reset'):
                break
            window.append(row)
        times = {'event_index': latest, 'observation_s': len(window)*STEP_MS/1000.}
        for name, velocity in (('issued', lambda row: read(row, 'v_final_w')), ('actual', actual)):
            index = first_stable(window, lambda row: progress(row, velocity(row)) >= .6*math.hypot(*command(row)[:2]))
            offset = None if index is None else index + (name == 'actual')
            times[name+'_resume_s'] = None if offset is None else offset*STEP_MS/1000.
        return times

    first_reset = next((i for i, row in enumerate(rows) if read(row, 'reset')), None)
    flight = rows if first_reset is None else rows[:first_reset+1]
    active = [row for row in flight if math.hypot(*command(row)[:2])]
    centre_y = initial['path_reference_w'][lane][1]
    exposed = [row['step'] for row in flight
               if crossing and abs(read(row, 'mover_position_w')[1]-centre_y) <= CORRIDOR_HALF_WIDTH_M]
    clearance = [math.dist(position(row), read(row, 'mover_position_after_w'))-.34
                 for row in flight] if crossing else []
    reported = dict(
        layout=LAYOUTS[lane//8], elapsed_s=len(flight)*STEP_MS/1000.,
        first_reset_step=first_reset, first_contact_step=next(
            (row['step'] for row in flight if read(row, 'collision')), None),
        initial_termination_reason=None if first_reset is None else read(flight[-1], 'termination_reason'),
        minimum_sampled_ray_m=min(read(row, 'ray_minimum_m') for row in flight),
        minimum_root_to_mover_surface_m=min(clearance) if clearance else None,
        exposure_first_step=exposed[0] if exposed else None,
        exposure_last_step=exposed[-1] if exposed else None,
        reversal_in_corridor=CONDITIONS['crossing_at_reversal']//STEP_MS in exposed,
        earliest_delivered_mover_step=next((row['step'] for row in flight
            if crossing and read(row, 'mover_winning_beams') and read(row, 'actor_witness')[-1][1] > .5), None),
        first_intervention_step=next((row['step'] for row in flight if intervened(row)), None),
        average_tracking_error_m_s=fmean(math.dist(actual(row), command(row)) for row in active) if active else None,
        average_intent_progress_m_s=fmean(progress(row, actual(row)) for row in active) if active else None,
    )
    release = exposed[-1]+1 if exposed else None
    reported['after_corridor'] = recovery_window(release) if release is not None and release < len(flight) else None
    reversal = CONDITIONS['crossing_at_reversal']//STEP_MS
    compare_start = max(reversal, release or 0)
    reported['after_reversal'] = None
    if (not crossing or release is not None) and compare_start < len(flight) and flight[compare_start]['event_index'] == flight[reversal]['event_index']:
        reported['after_reversal'] = dict(start_s=compare_start*STEP_MS/1000., **recovery_window(compare_start))
    stops = []
    last_event = None
    for i, row in enumerate(flight):
        new_event = row['event_index'] != last_event
        last_event = row['event_index']
        if i == 0 or not new_event or any(command(row)) or not any(command(flight[i-1])):
            continue
        stop_rows = []
        for item in flight[i:]:
            if item['event_index'] != last_event or read(item, 'reset'):
                break
            stop_rows.append(item)
        start = position(flight[i-1])
        settled = first_stable(stop_rows, lambda item: math.hypot(*actual(item)) <= SETTLED_SPEED_M_S)
        stops.append(dict(event_index=last_event, duration_s=len(stop_rows)*STEP_MS/1000.,
                          settled_after_s=None if settled is None else (settled+1)*STEP_MS/1000.,
                          max_drift_m=max((math.dist(start, position(item)) for item in stop_rows), default=0.)))
    reported['stops'] = stops
    return reported


def audit(output, params, counters):
    from .lidar_calibration import verify as verify_mount

    path = output/'command_replay.json'
    saved = json.loads(path.read_bytes())
    if saved['counters'] != counters or set(saved['models']) != set(params['models']):
        raise ValueError('COMMAND_REPLAY_COUNTERS')
    stream = load_stream(params['stream'])
    event_ids = sampled_events(stream)[:-1]
    sources = [file_pin(path), params['stream']]
    resets = 0
    results = {}
    first_pose = None
    for label, model in saved['models'].items():
        if model['checkpoint'] != params['models'][label] or set(model['trials']) != set(CONDITIONS):
            raise ValueError('COMMAND_REPLAY_MODEL_BINDING')
        sources.append(verify_mount(model['lidar_mount'], output, label, 25))
        results[label] = {}
        active_starts = []
        for name, trial in model['trials'].items():
            trace = checked(trial['trace'])
            if trace != output/f'{label}-{name}.jsonl':
                raise ValueError('COMMAND_REPLAY_TRACE_PATH')
            sources.append(trial['trace'])
            with trace.open() as handle:
                initial = json.loads(next(handle))
                rows = [json.loads(row) for row in handle]
            if (initial['kind'] != 'initial' or len(rows) != len(event_ids)
                    or trial['physics']['block_ticks'] != 20*len(rows)):
                raise ValueError('COMMAND_REPLAY_TRANSITIONS')
            if first_pose is None:
                first_pose = initial['root_state_w']
            if any(math.dist(a, b) > .001 for a, b in zip(first_pose, initial['root_state_w'], strict=True)):
                raise ValueError('COMMAND_REPLAY_PAIRED_START')
            if CONDITIONS[name] is not None:
                active_starts.append(initial['mover_start_w'])
            for step, (row, event_id) in enumerate(zip(rows, event_ids, strict=True)):
                if (row['kind'] != 'transition' or row['step'] != step or row['t_ms'] != step*STEP_MS
                        or row['event_index'] != event_id or row['raw_event'] != stream['events'][event_id]
                        or row['after_tick']-row['physics_tick'] != 20 or row['contact_samples'] != 20
                        or len(row['reset']) != LANES):
                    raise ValueError(f'COMMAND_REPLAY_TRANSITION_BINDING {label}/{name}/{step}')
                resets += sum(row['reset'])
                for lane in range(LANES):
                    actor = row['actor_state'][lane]
                    observed = body_to_world(row['attitude_wxyz'][lane], actor[-1][6:9])
                    if math.dist(observed, row['raw_event']['velocity_w_m_s']) > 2e-6:
                        raise ValueError('COMMAND_REPLAY_ACTOR_INPUT')
                    if step and not rows[step-1]['reset'][lane]:
                        if actor[:-1] != rows[step-1]['actor_state'][lane][1:]:
                            raise ValueError('COMMAND_REPLAY_ACTOR_HISTORY')
                        if row['episode_length'][lane] != rows[step-1]['episode_length'][lane]+1:
                            raise ValueError('COMMAND_REPLAY_EPISODE_CONTINUITY')
                    if row['mover_winning_beams'][lane]:
                        distance, valid, _ = row['actor_witness'][lane][-1]
                        if valid != 1 or abs(distance-row['mover_witness_range_m'][lane]) > .002:
                            raise ValueError('COMMAND_REPLAY_MOVER_WITNESS')
            results[label][name] = dict(trace=trial['trace'], lanes=[
                score_lane(rows, initial, lane, CONDITIONS[name] is not None and LAYOUTS[lane//8] == 'street_gap')
                for lane in range(LANES)])
        if active_starts[0] != active_starts[1]:
            raise ValueError('COMMAND_REPLAY_MOVER_START')
    if resets != counters['automatic_resets']:
        raise ValueError('COMMAND_REPLAY_RESET_COUNT')
    return dict(models=results, counters=counters, witnesses_replayed=True), sources
