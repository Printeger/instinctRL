"""Offline replay of command timing, exposure and first-flight outcomes."""
import json
import math
from statistics import fmean

from tools.r6_evidence import checked, file_pin
from .command_replay import (
    CLEARANCE_SHAPE, CLEAR_PASS_EXECUTIONS, CONDITIONS, CROSSING_SPEED_M_S, CROSSING_START_M, LANES, LAYOUTS,
    NOVEL_PREFIX, RANK_EXECUTIONS, REARM_EXECUTIONS, STEP_MS, STOP_RELEASE_EXECUTION, arm_execution, conditions,
    event_schedule,
)

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


def lane_value(value, lane):
    return value[lane] if isinstance(value, list) else value


def contiguous_windows(steps):
    windows = []
    for step in steps:
        if windows and step == windows[-1][1]+1:
            windows[-1][1] = step
        else:
            windows.append([step, step])
    return windows


def dense_clearance(path, steps):
    import numpy as np

    shape = (steps, *CLEARANCE_SHAPE)
    if path.stat().st_size != math.prod(shape)*4:
        raise ValueError('COMMAND_REPLAY_CLEARANCE_SIZE')
    return np.memmap(path, dtype='<f4', mode='r', shape=shape)


def verify_clearance(path, initial, rows):
    import numpy as np

    samples = dense_clearance(path, len(rows))
    previous = initial
    for step, (sample, row) in enumerate(zip(samples, rows, strict=True)):
        uninterrupted = np.logical_not(previous['reset']) if step else np.ones(LANES, dtype=bool)
        root_before = previous['root_state_after_w'] if step else previous['root_state_w']
        if (not np.isfinite(sample).all() or (sample[..., 6] <= 0).any()
                or not np.allclose(sample[0, uninterrupted, :3], np.asarray(root_before)[uninterrupted, :3], atol=2e-5, rtol=0)
                or not np.allclose(sample[-1, :, :3], np.asarray(row['root_state_after_w'])[:, :3], atol=2e-5, rtol=0)
                or not np.allclose(sample[0, :, 3:6], row['mover_position_w'], atol=2e-5, rtol=0)
                or not np.allclose(sample[-1, :, 3:6], row['mover_position_after_w'], atol=2e-5, rtol=0)):
            raise ValueError(f'COMMAND_REPLAY_CLEARANCE_ENDPOINT {step}')
        previous = row


def verify_novel_pulse(path, initial, rows, layouts, center_times, acceleration):
    from .robustness import lane_role

    pulse = json.loads(path.read_bytes())
    if not isinstance(pulse, list) or len(pulse) != 40:
        raise ValueError('COMMAND_REPLAY_PULSE_COUNT')
    first = pulse[0]['sim_tick']
    block = {'task': 'pulse_hold'}
    for i, row in enumerate(pulse):
        if set(row) != {'block_tick', 'sim_tick', 'force_n', 'mass_kg'} or row['block_tick'] != 1000+i or row['sim_tick'] != first+i:
            raise ValueError('COMMAND_REPLAY_PULSE_CLOCK')
        for lane, (mass, force) in enumerate(zip(row['mass_kg'], row['force_n'], strict=True)):
            _, direction = lane_role(block, lane)
            expected = [axis*mass*acceleration for axis in direction]
            if abs(mass-1.181517923) > 1.e-5 or math.dist(force, expected) > 1.e-5:
                raise ValueError('COMMAND_REPLAY_PULSE_FORCE')
        if len(row['mass_kg']) != LANES:
            raise ValueError('COMMAND_REPLAY_PULSE_LANES')
    centers = iter(center_times)
    for lane, (layout, delay) in enumerate(zip(layouts, initial['departure_delay_s'], strict=True)):
        center = next(centers) if layout == 'street_gap' else None
        expected = 0. if center is None else center/1000.-CROSSING_START_M/CROSSING_SPEED_M_S
        if abs(delay-expected) > 2.e-6:
            raise ValueError('COMMAND_REPLAY_CROSSING_DELAY')
        if center is not None:
            row = rows[round(center/STEP_MS)]
            if abs(row['mover_position_w'][lane][1]-initial['path_reference_w'][lane][1]) > .04:
                raise ValueError('COMMAND_REPLAY_CROSSING_READBACK')


def score_lane(rows, initial, lane, crossing):
    def read(row, name):
        return row[name][lane]

    def command(row):
        return lane_value(row['raw_event'], lane)['velocity_w_m_s']

    def event_index(row):
        return lane_value(row['event_index'], lane)

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
        latest = event_index(flight[start])
        window = []
        for row in flight[start:]:
            if event_index(row) != latest or not math.hypot(*command(row)[:2]) or read(row, 'reset'):
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
    exposure_windows = contiguous_windows(exposed)
    reversal = CONDITIONS['crossing_at_reversal']//STEP_MS
    clearance = [math.dist(position(row), read(row, 'mover_position_after_w'))-.34
                 for row in flight] if crossing else []
    layout = initial['layouts'][lane] if 'layouts' in initial else LAYOUTS[lane//8]
    reported = dict(
        layout=layout, elapsed_s=len(flight)*STEP_MS/1000.,
        first_reset_step=first_reset, first_contact_step=next(
            (row['step'] for row in flight if read(row, 'collision')), None),
        initial_termination_reason=None if first_reset is None else read(flight[-1], 'termination_reason'),
        minimum_sampled_ray_m=min(read(row, 'ray_minimum_m') for row in flight),
        minimum_root_to_mover_surface_m=min(clearance) if clearance else None,
        exposure_first_step=exposed[0] if exposed else None,
        exposure_last_step=exposed[-1] if exposed else None,
        exposure_windows=exposure_windows,
        reversal_in_corridor=reversal in exposed,
        earliest_delivered_mover_step=next((row['step'] for row in flight
            if crossing and read(row, 'mover_winning_beams') and read(row, 'actor_witness')[-1][1] > .5), None),
        first_intervention_step=next((row['step'] for row in flight if intervened(row)), None),
        first_escape_brake_step=next((row['step'] for row in flight
            if 'escape_braking' in row and read(row, 'escape_braking')), None),
        first_recovery_retreat_step=next((row['step'] for row in flight
            if 'recovery_retreating' in row and read(row, 'recovery_retreating')), None),
        average_tracking_error_m_s=fmean(math.dist(actual(row), command(row)) for row in active) if active else None,
        average_intent_progress_m_s=fmean(progress(row, actual(row)) for row in active) if active else None,
    )
    # Pair the first commanded reversal only with the first mover event.
    release = exposure_windows[0][1]+1 if exposure_windows else None
    reported['after_corridor'] = recovery_window(release) if release is not None and release < len(flight) else None
    compare_start = max(reversal, release or 0)
    reported['after_reversal'] = None
    if ((not isinstance(flight[0]['event_index'], list) or layout == 'street_gap')
            and (not crossing or release is not None) and compare_start < len(flight)
            and event_index(flight[compare_start]) == event_index(flight[reversal])):
        reported['after_reversal'] = dict(start_s=compare_start*STEP_MS/1000., **recovery_window(compare_start))
    if isinstance(flight[0]['event_index'], list):
        origin = initial['scene_origins_w'][lane]
        local = [[value-offset for value, offset in zip(position(row), origin, strict=True)] for row in flight]
        reported['route_progress'] = dict(maximum_x_m=max(p[0] for p in local), maximum_y_m=max(p[1] for p in local),
            minimum_y_m=min(p[1] for p in local),
            indoor_door_passed=any(p[0] > 2.1 and abs(p[1]) < .87 for p in local) if layout == 'room_corner' else None,
            indoor_corner_passed=any(p[0] > 4.1 and p[1] > .55 for p in local) if layout == 'room_corner' else None)
    stops = []
    last_event = None
    for i, row in enumerate(flight):
        new_event = event_index(row) != last_event
        last_event = event_index(row)
        if i == 0 or not new_event or any(command(row)) or not any(command(flight[i-1])):
            continue
        stop_rows = []
        for item in flight[i:]:
            if event_index(item) != last_event or read(item, 'reset'):
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
    trial_conditions = conditions(params)
    sources = [file_pin(path), params['stream']]
    novel = params['experiment_id'].startswith(NOVEL_PREFIX)
    if 'candidate_stream' in params:
        sources.append(params['candidate_stream'])
    if 'clearance_envelope' in params:
        sources.append(params['clearance_envelope'])
    if 'protocol' in params:
        protocol = json.loads(checked(params['protocol']).read_bytes())
        sources.extend([params['protocol'], protocol['progress_opportunities'], protocol['acceptance_gates']])
    resets = 0
    results = {}
    first_pose = None
    for label, model in saved['models'].items():
        layouts, event_ids, events = event_schedule(params, label)
        if (model['checkpoint'] != params['models'][label] or set(model['trials']) != set(trial_conditions)
                or model['execution'] != arm_execution(params, label)):
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
            if (initial['kind'] != 'initial' or len(rows) != len(event_ids)-1
                    or initial.get('layouts', layouts) != layouts
                    or trial['physics']['block_ticks'] != 20*len(rows)):
                raise ValueError('COMMAND_REPLAY_TRANSITIONS')
            if ('clearance' in trial) != ('clearance_envelope' in params):
                raise ValueError('COMMAND_REPLAY_CLEARANCE_BINDING')
            if 'clearance' in trial:
                dense = checked(trial['clearance'])
                if dense != output/f'{label}-{name}-clearance.f32':
                    raise ValueError('COMMAND_REPLAY_CLEARANCE_PATH')
                verify_clearance(dense, initial, rows)
                sources.append(trial['clearance'])
            if ('pulse' in trial) != novel:
                raise ValueError('COMMAND_REPLAY_PULSE_BINDING')
            if novel:
                pulse = checked(trial['pulse'])
                if pulse != output/f'{label}-{name}-pulse.json':
                    raise ValueError('COMMAND_REPLAY_PULSE_PATH')
                verify_novel_pulse(pulse, initial, rows, layouts, params['street_center_ms'], params['pulse_acceleration_m_s2'])
                sources.append(trial['pulse'])
            if first_pose is None:
                first_pose = initial['root_state_w']
            if any(math.dist(a, b) > .001 for a, b in zip(first_pose, initial['root_state_w'], strict=True)):
                raise ValueError('COMMAND_REPLAY_PAIRED_START')
            if trial_conditions[name] is not None:
                active_starts.append(initial['mover_start_w'])
            for step, row in enumerate(rows):
                if (row['kind'] != 'transition' or row['step'] != step or row['t_ms'] != step*STEP_MS
                        or row['event_index'] != event_ids[step] or row['raw_event'] != events[step]
                        or row['after_tick']-row['physics_tick'] != 20 or row['contact_samples'] != 20
                        or (step > 0 and row['physics_tick'] != rows[step-1]['after_tick'])
                        or len(row['reset']) != LANES):
                    raise ValueError(f'COMMAND_REPLAY_TRANSITION_BINDING {label}/{name}/{step}')
                resets += sum(row['reset'])
                if ('escape_clear_pass_hold' in row) != (model['execution'] in CLEAR_PASS_EXECUTIONS):
                    raise ValueError('COMMAND_REPLAY_HOLD_BINDING')
                held = row.get('escape_clear_pass_hold', [False]*LANES)
                if len(held) != LANES or any(type(value) is not bool for value in held):
                    raise ValueError('COMMAND_REPLAY_HOLD_LANES')
                if ('recovery_rearm_steps' in row) != (model['execution'] in REARM_EXECUTIONS):
                    raise ValueError('COMMAND_REPLAY_REARM_BINDING')
                rearm = row.get('recovery_rearm_steps', [0]*LANES)
                if len(rearm) != LANES or any(type(value) is not int or not 0 <= value <= 10 for value in rearm):
                    raise ValueError('COMMAND_REPLAY_REARM_TIMER')
                if ('escape_ranked_direction' in row) != (model['execution'] in RANK_EXECUTIONS):
                    raise ValueError('COMMAND_REPLAY_DIRECTION_RANK_BINDING')
                ranked = row.get('escape_ranked_direction', [False]*LANES)
                if len(ranked) != LANES or any(type(value) is not bool for value in ranked):
                    raise ValueError('COMMAND_REPLAY_DIRECTION_RANK_LANES')
                if ('escape_stop_release' in row) != (model['execution'] == STOP_RELEASE_EXECUTION):
                    raise ValueError('COMMAND_REPLAY_STOP_RELEASE_BINDING')
                released = row.get('escape_stop_release', [False]*LANES)
                if len(released) != LANES or any(type(value) is not bool for value in released):
                    raise ValueError('COMMAND_REPLAY_STOP_RELEASE_LANES')
                for lane in range(LANES):
                    actor = row['actor_state'][lane]
                    observed = body_to_world(row['attitude_wxyz'][lane], actor[-1][6:9])
                    if math.dist(observed, lane_value(row['raw_event'], lane)['velocity_w_m_s']) > 2e-6:
                        raise ValueError('COMMAND_REPLAY_ACTOR_INPUT')
                    if held[lane] and (math.hypot(*observed) > .001001 or math.hypot(*row['v_final_w'][lane]) > 1.e-6
                                       or row['escape_active'][lane]):
                        raise ValueError('COMMAND_REPLAY_HOLD_BEHAVIOR')
                    if rearm[lane] and row['escape_recovering'][lane]:
                        raise ValueError('COMMAND_REPLAY_REARM_BEHAVIOR')
                    if ranked[lane] and not row['escape_active'][lane]:
                        raise ValueError('COMMAND_REPLAY_DIRECTION_RANK_BEHAVIOR')
                    if released[lane] and not held[lane]:
                        raise ValueError('COMMAND_REPLAY_STOP_RELEASE_BEHAVIOR')
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
                score_lane(rows, initial, lane, trial_conditions[name] is not None and layouts[lane] == 'street_gap')
                for lane in range(LANES)])
            if 'clearance' in trial:
                results[label][name]['clearance'] = trial['clearance']
            if novel:
                results[label][name]['pulse'] = trial['pulse']
        if active_starts and any(start != active_starts[0] for start in active_starts[1:]):
            raise ValueError('COMMAND_REPLAY_MOVER_START')
    if resets != counters['automatic_resets']:
        raise ValueError('COMMAND_REPLAY_RESET_COUNT')
    return dict(models=results, counters=counters, witnesses_replayed=True), sources
