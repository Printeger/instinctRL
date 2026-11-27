"""Offline replay of command timing, exposure and first-flight outcomes."""
import json
import math
from statistics import fmean

from tools.r6_evidence import checked, file_pin
from .command_replay import (
    CLEARANCE_SHAPE, CLEAR_PASS_EXECUTIONS, COMMAND_HANDOFF_EXECUTIONS, COMMITTED_INTERCEPT_EXECUTIONS, CONDITIONS, CROSSING_SPEED_M_S,
    CONFIRMED_STANDOFF_EXECUTION, CONFIRMED_TUBE_EXECUTIONS, CROSSING_START_M, EDGE_CONTINUATION_EXECUTION,
    EARLY_DIAGONAL_EXECUTIONS, FINITE_EDGE_EXECUTIONS, INTERCEPT_CHOICE_EXECUTIONS, LANES, LAYOUTS,
    ALONG_INTERCEPT_EXECUTIONS, CROSSING_CONTINUATION_EXECUTIONS, CROSSING_COMPATIBLE_EXECUTIONS, CROSSING_EARLY_EXECUTIONS, CROSSING_FALLBACK_EXECUTIONS,
    CROSSING_EXTENDED_RETREAT_EXECUTIONS, CROSSING_RECEDING_HOVER_EXECUTIONS, CROSSING_VERIFIED_HOVER_EXECUTIONS,
    CROSSING_REFERENCE_EXECUTIONS, CROSSING_SETTLED_EXECUTIONS, CROSSING_SETTLED_MOUNT,
    BILATERAL_STOPPING_EXECUTION, SAMPLED_TURN_EXECUTIONS, OBSERVED_HEADING_EXECUTIONS, SHALLOW_STEERING_EXECUTIONS,
    FAR_CROSSING_REFERENCE_EXECUTIONS, HOVER_OWN_SIDE_EXECUTIONS, VERIFIED_RETREAT_ONSET_EXECUTIONS,
    CLIPPED_RETREAT_ONSET_EXECUTION, TUBE_STOPPING_EXECUTION,
    CROSSING_SHALLOW_EXECUTIONS, CROSSING_STOP_EXECUTIONS,
    FINITE_INTERCEPT_EXECUTIONS,
    OPENING_STEERING_EXECUTIONS,
    RESERVE_DIRECTION_EXECUTIONS, ROTATION_TUBE_EXECUTIONS,
    TANGENT_MAGNITUDE_EXECUTION, VERIFIED_RESUME_EXECUTION, VERIFIED_ZERO_EXECUTIONS,
    INTERCEPT_EXECUTIONS, LONG_PREFIX, NEIGHBOR_HISTORY_EXECUTIONS, NOVEL_PREFIX, OFFSET_STOP_EXECUTIONS,
    RANK_EXECUTIONS, REARM_EXECUTIONS, SETTLED_OFFSET_STOP_EXECUTIONS,
    STEP_MS, STOP_RELEASE_EXECUTIONS, WALL_TANGENT_EXECUTIONS,
    arm_execution, conditions, event_schedule, open_trace, trace_path,
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


def verify_pulse_and_crossing(path, initial, rows, layouts, center_times, acceleration):
    from .robustness import lane_role

    pulse = json.loads(path.read_bytes())
    if not isinstance(pulse, list) or len(pulse) != 40:
        raise ValueError('COMMAND_REPLAY_PULSE_COUNT')
    first = pulse[0]['sim_tick']
    if first != rows[50]['physics_tick']:
        raise ValueError('COMMAND_REPLAY_PULSE_BOUNDARY')
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
            center_step = round(center/STEP_MS)
            # A contact can reset the mover before the scheduled crossing. Use
            # the last first-flight boundary, before that reset, in this case.
            step = next((i for i, row in enumerate(rows[:center_step]) if row['reset'][lane]), center_step)
            reference = initial['path_reference_w'][lane][1]
            start = initial['mover_start_w'][lane][1]-reference
            before_center_s = max(0., (center-step*STEP_MS)/1000.)
            offset = math.copysign(min(CROSSING_START_M, before_center_s*CROSSING_SPEED_M_S), start)
            if abs(rows[step]['mover_position_w'][lane][1]-reference-offset) > .04:
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
                or read(row, 'escape_active') or read(row, 'escape_recovering')
                or ('escape_intercepted' in row and read(row, 'escape_intercepted')))

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
    long_run = params['experiment_id'].startswith(LONG_PREFIX)
    pulse_enabled = long_run or params['experiment_id'].startswith(NOVEL_PREFIX)
    pulse_settings = params
    if 'candidate_stream' in params:
        sources.append(params['candidate_stream'])
    if 'clearance_envelope' in params:
        sources.append(params['clearance_envelope'])
    if 'protocol' in params:
        protocol = json.loads(checked(params['protocol']).read_bytes())
        sources.extend([params['protocol'], protocol['progress_opportunities'], protocol['acceptance_gates']])
        pulse_settings = protocol
    resets = 0
    results = {}
    first_pose = None
    for label, model in saved['models'].items():
        layouts, event_ids, events = event_schedule(params, label)
        if (model['checkpoint'] != params['models'][label] or set(model['trials']) != set(trial_conditions)
                or model['execution'] != arm_execution(params, label)):
            raise ValueError('COMMAND_REPLAY_MODEL_BINDING')
        uses_neighbors = model.get('ics_profile', {}).get('uses_neighbor_history', False)
        standoff = model.get('ics_profile', {}).get('confirmed_standoff_m', 0.)
        expected_standoff = .45 if model['execution'] == CONFIRMED_STANDOFF_EXECUTION else 0.
        inner_tube = model.get('ics_profile', {}).get('confirmed_tube_radius_m', 0.)
        expected_inner_tube = .41 if model['execution'] in CONFIRMED_TUBE_EXECUTIONS else 0.
        aligns_rotation = model.get('ics_profile', {}).get('confirmed_tube_align_rotation', False)
        intercept_speeds = (0., .30, .2999) if model['execution'] in CROSSING_FALLBACK_EXECUTIONS else (0., .30)
        if (uses_neighbors != (model['execution'] in NEIGHBOR_HISTORY_EXECUTIONS)
                or standoff != expected_standoff or inner_tube != expected_inner_tube
                or aligns_rotation is not (model['execution'] in ROTATION_TUBE_EXECUTIONS)):
            raise ValueError('COMMAND_REPLAY_NEIGHBOR_HISTORY_PROFILE')
        if model.get('settled_offset_stop', False) is not (model['execution'] in SETTLED_OFFSET_STOP_EXECUTIONS):
            raise ValueError('COMMAND_REPLAY_SETTLED_OFFSET_STOP_PROFILE')
        tangent_profile = model['execution'] in WALL_TANGENT_EXECUTIONS
        if model.get('wall_tangent', False) is not tangent_profile:
            raise ValueError('COMMAND_REPLAY_WALL_TANGENT_PROFILE')
        detour_profile = model['execution'] in FINITE_EDGE_EXECUTIONS
        if model.get('finite_edge', False) is not detour_profile:
            raise ValueError('COMMAND_REPLAY_FINITE_EDGE_PROFILE')
        continuation_profile = model['execution'] == EDGE_CONTINUATION_EXECUTION
        if model.get('edge_continuation', False) is not continuation_profile:
            raise ValueError('COMMAND_REPLAY_EDGE_CONTINUATION_PROFILE')
        if model.get('tangent_magnitude', False) is not (model['execution'] == TANGENT_MAGNITUDE_EXECUTION):
            raise ValueError('COMMAND_REPLAY_TANGENT_MAGNITUDE_PROFILE')
        resume_profile = model['execution'] == VERIFIED_RESUME_EXECUTION
        if model.get('verified_resume', False) is not resume_profile:
            raise ValueError('COMMAND_REPLAY_VERIFIED_RESUME_PROFILE')
        if model.get('verified_zero', False) is not (model['execution'] in VERIFIED_ZERO_EXECUTIONS):
            raise ValueError('COMMAND_REPLAY_VERIFIED_ZERO_PROFILE')
        choice_profile = model['execution'] in INTERCEPT_CHOICE_EXECUTIONS
        if any(model.get(key, False) is not choice_profile for key in ('intercept_choice', 'receding_stop')):
            raise ValueError('COMMAND_REPLAY_INTERCEPT_CHOICE_PROFILE')
        if any(model.get(flag, False) is not (model['execution'] in CROSSING_FALLBACK_EXECUTIONS)
               for flag in ('zero_recheck', 'reverse_diagonal')):
            raise ValueError('COMMAND_REPLAY_MODEL_EXECUTION')
        if model.get('finite_intercept', False) is not (model['execution'] in FINITE_INTERCEPT_EXECUTIONS):
            raise ValueError('COMMAND_REPLAY_FINITE_INTERCEPT_PROFILE')
        if model.get('crossing_stop', False) is not (model['execution'] in CROSSING_STOP_EXECUTIONS):
            raise ValueError('COMMAND_REPLAY_CROSSING_STOP_PROFILE')
        if model.get('crossing_stop_continuation', False) is not (model['execution'] in CROSSING_CONTINUATION_EXECUTIONS):
            raise ValueError('COMMAND_REPLAY_CROSSING_STOP_CONTINUATION_PROFILE')
        if model.get('crossing_stop_shallow', False) is not (model['execution'] in CROSSING_SHALLOW_EXECUTIONS):
            raise ValueError('COMMAND_REPLAY_CROSSING_STOP_SHALLOW_PROFILE')
        if model.get('crossing_stop_early', False) is not (model['execution'] in CROSSING_EARLY_EXECUTIONS):
            raise ValueError('COMMAND_REPLAY_EARLY_CROSSING_PROFILE')
        if model.get('crossing_stop_compatible', False) is not (model['execution'] in CROSSING_COMPATIBLE_EXECUTIONS):
            raise ValueError('COMMAND_REPLAY_COMPATIBLE_CROSSING_PROFILE')
        if model.get('crossing_stop_verified_hover', False) is not (model['execution'] in CROSSING_VERIFIED_HOVER_EXECUTIONS):
            raise ValueError('COMMAND_REPLAY_VERIFIED_CROSSING_HOVER_PROFILE')
        recovers_on_own_side = model['execution'] in CROSSING_RECEDING_HOVER_EXECUTIONS
        if (model.get('crossing_stop_receding_hover', False) is not recovers_on_own_side
                or model.get('crossing_retreat_own_side', False) is not recovers_on_own_side):
            raise ValueError('COMMAND_REPLAY_RECEDING_CROSSING_HOVER_PROFILE')
        if model.get('crossing_retreat_extended', False) is not (model['execution'] in CROSSING_EXTENDED_RETREAT_EXECUTIONS):
            raise ValueError('COMMAND_REPLAY_EXTENDED_CROSSING_RETREAT_PROFILE')
        settled_mount = list(CROSSING_SETTLED_MOUNT) if model['execution'] in CROSSING_SETTLED_EXECUTIONS else None
        if model.get('settled_crossing_mount_m') != settled_mount:
            raise ValueError('COMMAND_REPLAY_SETTLED_CROSSING_MOUNT')
        crossing_reference = model['execution'] in CROSSING_REFERENCE_EXECUTIONS
        if (model.get('crossing_reference', False) is not crossing_reference
                or model.get('static_scan_stop', False) is not crossing_reference):
            raise ValueError('COMMAND_REPLAY_CROSSING_REFERENCE_PROFILE')
        shallow_profile = model['execution'] in SHALLOW_STEERING_EXECUTIONS
        if model.get('shallow_steering', False) is not shallow_profile:
            raise ValueError('COMMAND_REPLAY_SHALLOW_STEERING_PROFILE')
        observed_profile = model['execution'] in OBSERVED_HEADING_EXECUTIONS
        if (model.get('observed_heading_steering', False) is not observed_profile
                or model.get('tracked_resume', False) is not observed_profile):
            raise ValueError('COMMAND_REPLAY_OBSERVED_HEADING_PROFILE')
        sampled_turn_profile = model['execution'] in SAMPLED_TURN_EXECUTIONS
        if model.get('crossing_turn_between_samples', False) is not sampled_turn_profile:
            raise ValueError('COMMAND_REPLAY_SAMPLED_TURN_PROFILE')
        bilateral_profile = model['execution'] == BILATERAL_STOPPING_EXECUTION
        if model.get('bilateral_stopping', False) is not bilateral_profile:
            raise ValueError('COMMAND_REPLAY_BILATERAL_PROFILE')
        hover_own_side_profile = model['execution'] in HOVER_OWN_SIDE_EXECUTIONS
        if model.get('hover_escape_own_side', False) is not hover_own_side_profile:
            raise ValueError('COMMAND_REPLAY_HOVER_OWN_SIDE_PROFILE')
        far_crossing_profile = model['execution'] in FAR_CROSSING_REFERENCE_EXECUTIONS
        if model.get('far_crossing_reference', False) is not far_crossing_profile:
            raise ValueError('COMMAND_REPLAY_FAR_CROSSING_PROFILE')
        verified_retreat_profile = model['execution'] in VERIFIED_RETREAT_ONSET_EXECUTIONS
        if model.get('verified_retreat_onset', False) is not verified_retreat_profile:
            raise ValueError('COMMAND_REPLAY_VERIFIED_RETREAT_PROFILE')
        clipped_retreat_profile = model['execution'] == CLIPPED_RETREAT_ONSET_EXECUTION
        if model.get('retain_only_clipped_retreat', False) is not clipped_retreat_profile:
            raise ValueError('COMMAND_REPLAY_CLIPPED_RETREAT_PROFILE')
        tube_stopping_profile = model['execution'] == TUBE_STOPPING_EXECUTION
        if model.get('tube_stopping', False) is not tube_stopping_profile:
            raise ValueError('COMMAND_REPLAY_FINITE_TUBE_PROFILE')
        if model.get('along_intercept', False) is not (model['execution'] in ALONG_INTERCEPT_EXECUTIONS):
            raise ValueError('COMMAND_REPLAY_ALONG_INTERCEPT_PROFILE')
        opening_profile = model['execution'] in OPENING_STEERING_EXECUTIONS
        if model.get('opening_steering', False) is not opening_profile:
            raise ValueError('COMMAND_REPLAY_OPENING_STEERING_PROFILE')
        if model.get('reserve_direction', False) is not (model['execution'] in RESERVE_DIRECTION_EXECUTIONS):
            raise ValueError('COMMAND_REPLAY_MODEL_EXECUTION')
        if model.get('early_diagonal', False) is not (model['execution'] in EARLY_DIAGONAL_EXECUTIONS):
            raise ValueError('COMMAND_REPLAY_EARLY_DIAGONAL_PROFILE')
        sources.append(verify_mount(model['lidar_mount'], output, label, 25))
        results[label] = {}
        active_starts = []
        for name, trial in model['trials'].items():
            trace = checked(trial['trace'])
            if trace != trace_path(output, label, name, run_id=params['experiment_id']):
                raise ValueError('COMMAND_REPLAY_TRACE_PATH')
            sources.append(trial['trace'])
            with open_trace(trace) as handle:
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
            if ('pulse' in trial) != pulse_enabled:
                raise ValueError('COMMAND_REPLAY_PULSE_BINDING')
            if pulse_enabled:
                pulse = checked(trial['pulse'])
                if pulse != output/f'{label}-{name}-pulse.json':
                    raise ValueError('COMMAND_REPLAY_PULSE_PATH')
                verify_pulse_and_crossing(pulse, initial, rows, layouts, pulse_settings['street_center_ms'],
                                          pulse_settings['pulse_acceleration_m_s2'])
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
                if ('escape_stop_release' in row) != (model['execution'] in STOP_RELEASE_EXECUTIONS):
                    raise ValueError('COMMAND_REPLAY_STOP_RELEASE_BINDING')
                released = row.get('escape_stop_release', [False]*LANES)
                if len(released) != LANES or any(type(value) is not bool for value in released):
                    raise ValueError('COMMAND_REPLAY_STOP_RELEASE_LANES')
                if ('escape_intercepted' in row) != (model['execution'] in INTERCEPT_EXECUTIONS):
                    raise ValueError('COMMAND_REPLAY_INTERCEPT_BINDING')
                intercepted = row.get('escape_intercepted', [False]*LANES)
                if len(intercepted) != LANES or any(type(value) is not bool for value in intercepted):
                    raise ValueError('COMMAND_REPLAY_INTERCEPT_LANES')
                if ('escape_intercept_continued' in row) != (model['execution'] in COMMITTED_INTERCEPT_EXECUTIONS):
                    raise ValueError('COMMAND_REPLAY_INTERCEPT_CONTINUATION_BINDING')
                continued = row.get('escape_intercept_continued', [False]*LANES)
                if len(continued) != LANES or any(type(value) is not bool for value in continued):
                    raise ValueError('COMMAND_REPLAY_INTERCEPT_CONTINUATION_LANES')
                if ('escape_command_handoff' in row) != (model['execution'] in COMMAND_HANDOFF_EXECUTIONS):
                    raise ValueError('COMMAND_REPLAY_HANDOFF_BINDING')
                handoff = row.get('escape_command_handoff', [False]*LANES)
                if len(handoff) != LANES or any(type(value) is not bool for value in handoff):
                    raise ValueError('COMMAND_REPLAY_HANDOFF_LANES')
                if ('escape_offset_stop' in row) != (model['execution'] in OFFSET_STOP_EXECUTIONS):
                    raise ValueError('COMMAND_REPLAY_OFFSET_STOP_BINDING')
                offset = row.get('escape_offset_stop', [False]*LANES)
                if len(offset) != LANES or any(type(value) is not bool for value in offset):
                    raise ValueError('COMMAND_REPLAY_OFFSET_STOP_LANES')
                if ('escape_wall_tangent' in row) != tangent_profile:
                    raise ValueError('COMMAND_REPLAY_WALL_TANGENT_BINDING')
                tangent = row.get('escape_wall_tangent', [False]*LANES)
                if len(tangent) != LANES or any(type(value) is not bool for value in tangent):
                    raise ValueError('COMMAND_REPLAY_WALL_TANGENT_LANES')
                if ('escape_finite_edge' in row) != detour_profile:
                    raise ValueError('COMMAND_REPLAY_FINITE_EDGE_BINDING')
                detour = row.get('escape_finite_edge', [False]*LANES)
                if ('escape_shallow_steering' in row) != shallow_profile:
                    raise ValueError('COMMAND_REPLAY_SHALLOW_STEERING_BINDING')
                shallow = row.get('escape_shallow_steering', [False]*LANES)
                if len(shallow) != LANES or any(type(value) is not bool for value in shallow):
                    raise ValueError('COMMAND_REPLAY_SHALLOW_STEERING_LANES')
                if (('escape_observed_heading' in row) != observed_profile
                        or ('escape_tracked_resume' in row) != observed_profile):
                    raise ValueError('COMMAND_REPLAY_OBSERVED_HEADING_BINDING')
                observed_heading = row.get('escape_observed_heading', [False]*LANES)
                tracked = row.get('escape_tracked_resume', [False]*LANES)
                if any(len(flags) != LANES or any(type(value) is not bool for value in flags)
                       for flags in (observed_heading, tracked)):
                    raise ValueError('COMMAND_REPLAY_OBSERVED_HEADING_LANES')
                settled_crossing = row.get('escape_settled_crossing', [False]*LANES)
                if (('escape_settled_crossing' in row) != sampled_turn_profile or len(settled_crossing) != LANES
                        or any(type(value) is not bool for value in settled_crossing)):
                    raise ValueError('COMMAND_REPLAY_SAMPLED_TURN_BINDING')
                bilateral = row.get('bilateral_stopping', [False]*LANES)
                if (('bilateral_stopping' in row) != bilateral_profile or len(bilateral) != LANES
                        or any(type(value) is not bool for value in bilateral)):
                    raise ValueError('COMMAND_REPLAY_BILATERAL_BINDING')
                hover_own_side = row.get('escape_hover_own_side', [False]*LANES)
                if (('escape_hover_own_side' in row) != hover_own_side_profile or len(hover_own_side) != LANES
                        or any(type(value) is not bool for value in hover_own_side)):
                    raise ValueError('COMMAND_REPLAY_HOVER_OWN_SIDE_BINDING')
                far_crossing = row.get('escape_far_crossing_refined', [False]*LANES)
                if (('escape_far_crossing_refined' in row) != far_crossing_profile or len(far_crossing) != LANES
                        or any(type(value) is not bool for value in far_crossing)):
                    raise ValueError('COMMAND_REPLAY_FAR_CROSSING_BINDING')
                retreat_kept = row.get('escape_retreat_request_kept', [False]*LANES)
                if (('escape_retreat_request_kept' in row) != verified_retreat_profile or len(retreat_kept) != LANES
                        or any(type(value) is not bool for value in retreat_kept)):
                    raise ValueError('COMMAND_REPLAY_VERIFIED_RETREAT_BINDING')
                tube_holds = row.get('tube_stopping_hold_s', [0.]*LANES)
                tube_targets = row.get('tube_stopping_target_b', [[0.,0.,0.]]*LANES)
                if (('tube_stopping_hold_s' in row) != tube_stopping_profile
                        or ('tube_stopping_target_b' in row) != tube_stopping_profile
                        or len(tube_holds) != LANES or len(tube_targets) != LANES):
                    raise ValueError('COMMAND_REPLAY_FINITE_TUBE_BINDING')
                if len(detour) != LANES or any(type(value) is not bool for value in detour):
                    raise ValueError('COMMAND_REPLAY_FINITE_EDGE_LANES')
                if ('escape_edge_continued' in row) != continuation_profile:
                    raise ValueError('COMMAND_REPLAY_EDGE_CONTINUATION_BINDING')
                continued_edge = row.get('escape_edge_continued', [False]*LANES)
                if len(continued_edge) != LANES or any(type(value) is not bool for value in continued_edge):
                    raise ValueError('COMMAND_REPLAY_EDGE_CONTINUATION_LANES')
                if ('escape_opening_steered' in row) != opening_profile:
                    raise ValueError('COMMAND_REPLAY_OPENING_STEERING_BINDING')
                opening = row.get('escape_opening_steered', [False]*LANES)
                if len(opening) != LANES or any(type(value) is not bool for value in opening):
                    raise ValueError('COMMAND_REPLAY_OPENING_STEERING_LANES')
                if ('escape_verified_resume' in row) != resume_profile:
                    raise ValueError('COMMAND_REPLAY_VERIFIED_RESUME_BINDING')
                resumed = row.get('escape_verified_resume', [False]*LANES)
                if len(resumed) != LANES or any(type(value) is not bool for value in resumed):
                    raise ValueError('COMMAND_REPLAY_VERIFIED_RESUME_LANES')
                for lane in range(LANES):
                    actor = row['actor_state'][lane]
                    observed = body_to_world(row['attitude_wxyz'][lane], actor[-1][6:9])
                    if math.dist(observed, lane_value(row['raw_event'], lane)['velocity_w_m_s']) > 2e-6:
                        raise ValueError('COMMAND_REPLAY_ACTOR_INPUT')
                    if tangent[lane] and (not .09999 <= math.hypot(*observed) <= .30001
                            or row['escape_active'][lane] or intercepted[lane] or row['ics_emergency'][lane] != 0.
                            or dot(observed, row['v_final_w'][lane]) <= 0.):
                        raise ValueError('COMMAND_REPLAY_WALL_TANGENT_BEHAVIOR')
                    minimum_detour_speed = .02999 if observed_heading[lane] else .09999
                    if detour[lane] and (not minimum_detour_speed <= math.hypot(*observed) <= .30001
                            or tangent[lane] or row['escape_active'][lane] or intercepted[lane]
                            or row['ics_emergency'][lane] != 0.
                            or dot(observed, row['v_final_w'][lane]) < .5999*dot(observed, observed)):
                        raise ValueError('COMMAND_REPLAY_FINITE_EDGE_BEHAVIOR')
                    if shallow[lane] and (not detour[lane] or row['ics_beta'][lane] < .89999):
                        raise ValueError('COMMAND_REPLAY_SHALLOW_STEERING_BEHAVIOR')
                    if observed_heading[lane] and (not detour[lane] or shallow[lane] or row['ics_beta'][lane] < .89999):
                        raise ValueError('COMMAND_REPLAY_OBSERVED_HEADING_BEHAVIOR')
                    if tracked[lane] and (not row['escape_active'][lane] or row['escape_braking'][lane]
                            or row['recovery_retreating'][lane] or ranked[lane] or held[lane] or intercepted[lane]
                            or row['ics_emergency'][lane] or row['ics_beta'][lane] < .89999
                            or row['selected_action'][lane] != row['action'][lane]
                            or not .079999 <= math.hypot(*observed) <= .300001
                            or dot(observed, row['v_final_w'][lane]) < .6599*dot(observed, observed)):
                        raise ValueError('COMMAND_REPLAY_TRACKED_RESUME_BEHAVIOR')
                    if continued_edge[lane] and (not detour[lane]
                            or any(math.dist(old[6:9], actor[-1][6:9]) > .040001 for old in actor[:-1])
                            or not any(old['escape_finite_edge'][lane] and not old['escape_edge_continued'][lane]
                                       for old in rows[max(0,step-8):step])):
                        raise ValueError('COMMAND_REPLAY_EDGE_CONTINUATION_BEHAVIOR')
                    if held[lane] and (math.hypot(*observed) > .001001 or math.hypot(*row['v_final_w'][lane]) > 1.e-6
                                       or row['escape_active'][lane]):
                        raise ValueError('COMMAND_REPLAY_HOLD_BEHAVIOR')
                    if rearm[lane] and row['escape_recovering'][lane]:
                        raise ValueError('COMMAND_REPLAY_REARM_BEHAVIOR')
                    if ranked[lane] and not row['escape_active'][lane]:
                        raise ValueError('COMMAND_REPLAY_DIRECTION_RANK_BEHAVIOR')
                    if released[lane] and not held[lane]:
                        raise ValueError('COMMAND_REPLAY_STOP_RELEASE_BEHAVIOR')
                    if offset[lane] and not held[lane]:
                        raise ValueError('COMMAND_REPLAY_OFFSET_STOP_BEHAVIOR')
                    if settled_crossing[lane] and (not offset[lane] or row['ics_emergency'][lane]
                            or any(math.hypot(*frame[6:9]) > 1.e-6 or math.hypot(*frame[9:12]) > .020001
                                   for frame in actor)):
                        raise ValueError('COMMAND_REPLAY_SAMPLED_TURN_BEHAVIOR')
                    if bilateral[lane] and (not .02999 <= math.hypot(*observed) <= .30001
                            or row['escape_active'][lane] or row['recovery_retreating'][lane] or held[lane]
                            or intercepted[lane] or row['ics_emergency'][lane] or row['ics_beta'][lane] < .89999
                            or dot(observed, row['v_final_w'][lane]) < .5999*dot(observed, observed)):
                        raise ValueError('COMMAND_REPLAY_BILATERAL_BEHAVIOR')
                    if tube_stopping_profile:
                        hold, target = tube_holds[lane], tube_targets[lane]
                        if not math.isfinite(hold) or len(target) != 3 or not all(math.isfinite(component) for component in target):
                            raise ValueError('COMMAND_REPLAY_FINITE_TUBE_TARGET')
                        if not hold and any(target):
                            raise ValueError('COMMAND_REPLAY_FINITE_TUBE_INACTIVE_TARGET')
                        if hold and (min(abs(hold-.30),abs(hold-.60)) > 1.e-6
                                or not .02999 <= math.hypot(*observed) <= .30001
                                or not 0. < math.hypot(*target) <= .300001
                                or row['escape_active'][lane] or row['recovery_retreating'][lane] or held[lane]
                                or row['escape_braking'][lane] or ranked[lane] or row['escape_recovering'][lane]
                                or intercepted[lane] or continued[lane] or row['ics_emergency'][lane]
                                or row['ics_beta'][lane] < .89999 or dot(actor[-1][6:9],target) <= 0.
                                or math.dist(row['v_final_b'][lane],[v*row['ics_beta'][lane] for v in target]) > 2.e-6):
                            raise ValueError('COMMAND_REPLAY_FINITE_TUBE_BEHAVIOR')
                    if hover_own_side[lane] and (not intercepted[lane] or row['escape_active'][lane]
                            or row['ics_beta'][lane] < .89999 or any(math.hypot(*frame[6:9]) > 1.e-6 for frame in actor)):
                        raise ValueError('COMMAND_REPLAY_HOVER_OWN_SIDE_BEHAVIOR')
                    if intercepted[lane]:
                        alpha, x, y, z = row['selected_action'][lane]
                        planar_speed = math.hypot(x-.5, y-.5)
                        if (held[lane] or row['ics_emergency'][lane] or abs(alpha) > 1.e-6 or abs(z-.5) > 1.e-6
                                or min(abs(planar_speed-speed) for speed in intercept_speeds) > 1.e-6):
                            raise ValueError('COMMAND_REPLAY_INTERCEPT_BEHAVIOR')
                    if handoff[lane] and (row['escape_active'][lane] or math.hypot(*observed) < .079999
                                          or (not intercepted[lane] and row['selected_action'][lane] != row['action'][lane])):
                        raise ValueError('COMMAND_REPLAY_HANDOFF_BEHAVIOR')
                    if resumed[lane] and (not step or rows[step-1]['reset'][lane]
                            or not rows[step-1]['escape_active'][lane] or row['escape_active'][lane]
                            or intercepted[lane] or row['ics_emergency'][lane] or row['ics_beta'][lane] < .89999
                            or row['selected_action'][lane] != row['action'][lane]
                            or not .079999 <= math.hypot(*observed) <= .300001
                            or math.dist(actor[-2][6:9], actor[-1][6:9]) > .040001
                            or dot(observed, row['v_final_w'][lane]) < .5999*dot(observed, observed)):
                        raise ValueError('COMMAND_REPLAY_VERIFIED_RESUME_BEHAVIOR')
                    if continued[lane] and (not intercepted[lane] or not step or rows[step-1]['reset'][lane]
                                            or not rows[step-1]['escape_intercepted'][lane]
                                            or math.hypot(*row['v_final_b'][lane]) < .119999):
                        raise ValueError('COMMAND_REPLAY_INTERCEPT_CONTINUATION_BEHAVIOR')
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
            if pulse_enabled:
                results[label][name]['pulse'] = trial['pulse']
        if active_starts and any(start != active_starts[0] for start in active_starts[1:]):
            raise ValueError('COMMAND_REPLAY_MOVER_START')
    if resets != counters['automatic_resets']:
        raise ValueError('COMMAND_REPLAY_RESET_COUNT')
    return dict(models=results, counters=counters, witnesses_replayed=True), sources
