"""Continuous, development-only command replay on the frozen B/C plant."""
from bisect import bisect_right
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from tools.r6_evidence import checked, file_pin
from .escape_holdout_scene import COMMAND_LAYOUTS, LAYOUTS, command_layout


PREFIX = 'R6-DIRECT-INSTINCT-GPU-COMMAND-'
GUARD_PREFIX = PREFIX + 'GUARD-'
ROUTE_PREFIX = PREFIX + 'ROUTE-'
CRAZY_PREFIX = PREFIX + 'CRAZY-'
ENDURANCE_PREFIX = PREFIX + 'ENDURANCE-'
RETURN_PREFIX = PREFIX + 'RETURN-'
EARLY_PREFIX = PREFIX + 'EARLY-'
HOVER_PREFIX = PREFIX + 'HOVER-'
LONG_PREFIX = PREFIX + 'LONG-'
RETURN_DURATION_MS = 36000
REGRESSION_ID = PREFIX + 'R6-01'
MODE = 'research-r6-dynamic-assay'
LANES = 32
STEP_MS = 100
CROSSING_SPEED_M_S = .58
CROSSING_START_M = 2.2
CROSSING_TRAVEL_S = 7.5
# Centre-line arrival, in simulated milliseconds. The two active arms share their path.
CONDITIONS = {'no_crossing': None, 'crossing_at_reversal': 6400, 'crossing_before_reversal': 4400}
MODELS = {
    'b': '841a0a7c7926a14584f257667c5dffe0077129c33458898026ae37405ce9e231',
    'c': '450b8675161f132178eb345f85c3dd6c27dffe9f3de6a33c792c3fabcf6ba172',
}
EXECUTION = 'm4-escape-rear-clearance-p25-v1'
BRAKE_EXECUTION = 'm4-escape-brake-forward-p25-v1'
TUBE_EXECUTION = 'm4-escape-brake-tube-p25-v1'
TUBE_CONFIRM_EXECUTION = 'm4-escape-brake-tube-p25-v2'
RECOVERY_RETREAT_EXECUTION = 'm4-escape-brake-tube-p25-recovery-retreat-v1'
EARLY_RETREAT_EXECUTION = 'm4-escape-brake-tube-p25-early-retreat-v1'
SIDE_HOVER_RETREAT_EXECUTION = 'm4-escape-brake-tube-p25-side-hover-v1'
EXECUTION_PAIRS = {
    GUARD_PREFIX: (EXECUTION, BRAKE_EXECUTION),
    CRAZY_PREFIX: (BRAKE_EXECUTION, TUBE_EXECUTION),
    ENDURANCE_PREFIX: (TUBE_EXECUTION, TUBE_CONFIRM_EXECUTION),
    RETURN_PREFIX: (TUBE_CONFIRM_EXECUTION, RECOVERY_RETREAT_EXECUTION),
    EARLY_PREFIX: (TUBE_CONFIRM_EXECUTION, EARLY_RETREAT_EXECUTION),
    HOVER_PREFIX: (EARLY_RETREAT_EXECUTION, SIDE_HOVER_RETREAT_EXECUTION),
}
CRAZY_SCHEMA = 'instinctrl-crazy-command-generator/1'
CRAZY_MODES = ('normal', 'aggressive', 'obstacleward', 'oscillation', 'stop')


def matches(run_id):
    return isinstance(run_id, str) and re.fullmatch(PREFIX + r'(?:PAIR|GUARD|ROUTE|CRAZY|ENDURANCE|RETURN|EARLY|HOVER|LONG)-[0-9]{2}', run_id) is not None


def route_layouts(run_id):
    return run_id.startswith((ROUTE_PREFIX, CRAZY_PREFIX, ENDURANCE_PREFIX, RETURN_PREFIX, EARLY_PREFIX, HOVER_PREFIX, LONG_PREFIX))


def arm_execution(params, label):
    return params.get('candidate_execution', params['execution']) if label == 'candidate' else params['execution']


def validate_stream(stream):
    generated = stream.get('source') == 'crazy_command_generator'
    if (set(stream) != {'schema_version', 'source', 'duration_ms', 'events'}
            or stream['schema_version'] != 'instinctrl-world-command-stream/1'
            or (not generated and stream['source'] != 'scripted_development')
            or type(stream['duration_ms']) is not int or stream['duration_ms'] % STEP_MS
            or not 8000 <= stream['duration_ms'] <= (3_600_000 if generated else 60000)
            or not isinstance(stream['events'], list) or not stream['events']):
        raise ValueError('COMMAND_REPLAY_STREAM')
    last = -1
    for event in stream['events']:
        fields = {'t_ms', 'velocity_w_m_s'}
        if isinstance(event, dict) and generated and 'mode' in event:
            if event['mode'] not in CRAZY_MODES:
                raise ValueError('COMMAND_REPLAY_GENERATOR_MODE')
            fields.add('mode')
        if not isinstance(event, dict) or set(event) != fields:
            raise ValueError('COMMAND_REPLAY_EVENT')
        t = event['t_ms']
        command = event['velocity_w_m_s']
        if (type(t) is not int or not last < t < stream['duration_ms']
                or not isinstance(command, list) or len(command) != 3
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in command)
                or command[2] != 0 or math.hypot(*command[:2]) > .30):
            raise ValueError('COMMAND_REPLAY_EVENT')
        last = t
    if stream['events'][0]['t_ms'] != 0:
        raise ValueError('COMMAND_REPLAY_INITIAL_EVENT')
    return stream


def load_stream(pin):
    return validate_stream(json.loads(checked(pin).read_bytes()))


def lane_seed_for(seed, lane, layout):
    digest = sha256(f'{CRAZY_SCHEMA}/{seed}/{lane}/{layout}'.encode()).digest()
    return int.from_bytes(digest[:4], 'big') % 2**31


def layout_stream(duration_ms, events, *, generated):
    return validate_stream(dict(schema_version='instinctrl-world-command-stream/1',
                    source='crazy_command_generator' if generated else 'scripted_development',
                    duration_ms=duration_ms, events=events))


def generator_prefix(generator):
    if (set(generator) != {'schema_version', 'seed', 'prefix', 'prefix_duration_ms'}
            or generator['schema_version'] != CRAZY_SCHEMA or type(generator['seed']) is not int
            or not 0 <= generator['seed'] < 2**31 or type(generator['prefix_duration_ms']) is not int):
        raise ValueError('COMMAND_REPLAY_GENERATOR_METADATA')
    template = json.loads(checked(generator['prefix']).read_bytes())
    if (set(template) != {'schema_version', 'duration_ms', 'layouts'}
            or template['duration_ms'] != generator['prefix_duration_ms']):
        raise ValueError('COMMAND_REPLAY_GENERATOR_PREFIX')
    return template


def check_generated(stream, prefix, prefix_duration):
    extension = stream['events'][len(prefix):]
    if (stream['events'][:len(prefix)] != prefix or not extension or extension[0]['t_ms'] != prefix_duration
            or {event.get('mode') for event in extension} != set(CRAZY_MODES)):
        raise ValueError('COMMAND_REPLAY_GENERATOR_COVERAGE')


def route_streams(value):
    fields = {'schema_version', 'duration_ms', 'layouts'}
    generated = 'generator' in value
    if generated:
        fields.add('generator')
    if (set(value) != fields
            or value['schema_version'] != 'instinctrl-command-route-streams/1'
            or not isinstance(value['layouts'], dict) or set(value['layouts']) != set(COMMAND_LAYOUTS)):
        raise ValueError('COMMAND_REPLAY_ROUTES')
    streams = {layout: layout_stream(value['duration_ms'], events, generated=generated)
               for layout, events in value['layouts'].items()}
    if generated:
        template = generator_prefix(value['generator'])
        for layout, stream in streams.items():
            check_generated(stream, template['layouts'][layout], template['duration_ms'])
    return streams


def load_route_streams(pin):
    return route_streams(json.loads(checked(pin).read_bytes()))


def load_lane_streams(pin, *, require_independent=False):
    value = json.loads(checked(pin).read_bytes())
    layouts = [command_layout(lane) for lane in range(LANES)]
    if value.get('schema_version') != 'instinctrl-command-lane-streams/1':
        if require_independent:
            raise ValueError('COMMAND_REPLAY_INDEPENDENT_LANES_REQUIRED')
        streams = route_streams(value)
        return layouts, [streams[name] for name in layouts]
    if set(value) != {'schema_version', 'duration_ms', 'generator', 'lanes'} or len(value['lanes']) != LANES:
        raise ValueError('COMMAND_REPLAY_LANE_STREAMS')
    generator = value['generator']
    template = generator_prefix(generator)
    streams = []
    for lane, (layout, row) in enumerate(zip(layouts, value['lanes'], strict=True)):
        if (set(row) != {'lane', 'layout', 'seed', 'events'} or row['lane'] != lane or row['layout'] != layout
                or row['seed'] != lane_seed_for(generator['seed'], lane, layout)):
            raise ValueError('COMMAND_REPLAY_LANE_SEED')
        stream = layout_stream(value['duration_ms'], row['events'], generated=True)
        check_generated(stream, template['layouts'][layout], template['duration_ms'])
        streams.append(stream)
    if len({row['seed'] for row in value['lanes']}) != LANES:
        raise ValueError('COMMAND_REPLAY_NONINDEPENDENT_SEED')
    return layouts, streams


def conditions(params):
    if params['experiment_id'].startswith(LONG_PREFIX):
        protocol = json.loads(checked(params['protocol']).read_bytes())
        return {'layout_routes': protocol['street_first_center_ms']}
    if route_layouts(params['experiment_id']):
        return {'layout_routes': CONDITIONS['crossing_at_reversal']}
    return CONDITIONS


def event_schedule(params):
    """Keep single-stream pair rows stable and expand route inputs by lane."""
    if route_layouts(params['experiment_id']):
        layouts, by_lane = load_lane_streams(params['stream'], require_independent=params['experiment_id'].startswith(LONG_PREFIX))
        limit = RETURN_DURATION_MS//STEP_MS+1 if params['experiment_id'].startswith((RETURN_PREFIX, EARLY_PREFIX)) else None
        indices = list(zip(*(sampled_events(stream)[:limit] for stream in by_lane), strict=True))
        events = [[stream['events'][index] for stream, index in zip(by_lane, row, strict=True)] for row in indices]
        return layouts, [list(row) for row in indices], events
    stream = load_stream(params['stream'])
    indices = sampled_events(stream)
    return [LAYOUTS[lane//8] for lane in range(LANES)], indices, [stream['events'][i] for i in indices]


def sampled_events(stream):
    """Sample the latest event at each boundary, including the final observation."""
    events = stream['events']
    times = [event['t_ms'] for event in events]
    return [bisect_right(times, t) - 1 for t in range(0, stream['duration_ms'] + 1, STEP_MS)]


def long_contract(params):
    fields = {'experiment_id', 'training_seed', 'seed', 'models', 'execution', 'stream', 'protocol', 'wall_seconds', 'output_cap_bytes'}
    if (set(params) != fields or not matches(params['experiment_id']) or params['training_seed'] != 20260908
            or type(params['seed']) is not int or not 0 <= params['seed'] < 2**31 or params['seed'] in (167149835, 185193671)
            or params['wall_seconds'] != 7200 or params['output_cap_bytes'] != 2**31
            or params['execution'] not in (TUBE_CONFIRM_EXECUTION, EARLY_RETREAT_EXECUTION, SIDE_HOVER_RETREAT_EXECUTION)
            or set(params['models']) != {'candidate'} or params['models']['candidate']['sha256'] != MODELS['b']):
        raise ValueError('COMMAND_REPLAY_LONG_PARAMETERS')
    checked(params['models']['candidate'])
    protocol = json.loads(checked(params['protocol']).read_bytes())
    if (set(protocol) != {'schema_version', 'experiment_id', 'scene_seed', 'stream', 'execution', 'checkpoint',
                          'progress_opportunities', 'acceptance_gates', 'street_first_center_ms'}
            or protocol['schema_version'] != 'instinctrl-command-long-protocol/1'
            or protocol['experiment_id'] != params['experiment_id'] or protocol['scene_seed'] != params['seed']
            or protocol['stream'] != params['stream'] or protocol['execution'] != params['execution']
            or protocol['checkpoint'] != params['models']['candidate']
            or type(protocol['street_first_center_ms']) is not int
            or not 4000 <= protocol['street_first_center_ms'] <= 20000):
        raise ValueError('COMMAND_REPLAY_LONG_FREEZE')
    opportunities = json.loads(checked(protocol['progress_opportunities']).read_bytes())
    if (opportunities['schema_version'] != 'instinctrl-frozen-progress-opportunities/1'
            or opportunities['run_id'] != params['experiment_id']
            or {row['layout'] for row in opportunities['windows']} != set(COMMAND_LAYOUTS)
            or any(not 0 <= row['start_step'] < row['end_step'] <= 18000 for row in opportunities['windows'])):
        raise ValueError('COMMAND_REPLAY_LONG_PROGRESS_PLAN')
    gates = json.loads(checked(protocol['acceptance_gates']).read_bytes())
    limits = {'minimum_clearance_m', 'maximum_tracking_rmse_m_s', 'minimum_progress_fraction',
              'maximum_stop_s', 'maximum_stop_drift_m', 'maximum_intervention_s', 'maximum_recovery_s'}
    if (set(gates) != {'schema_version', 'by_layout'} or gates['schema_version'] != 'instinctrl-command-acceptance-gates/1'
            or set(gates['by_layout']) != set(COMMAND_LAYOUTS)):
        raise ValueError('COMMAND_REPLAY_LONG_GATES')
    for gate in gates['by_layout'].values():
        if (set(gate) != limits | {'minimum_hazard_events'} or type(gate['minimum_hazard_events']) is not int
                or gate['minimum_hazard_events'] <= 0 or not 0 < gate['minimum_progress_fraction'] <= 1
                or any(type(gate[name]) not in (int, float) or not math.isfinite(gate[name]) or gate[name] <= 0
                       for name in limits)):
            raise ValueError('COMMAND_REPLAY_LONG_LAYOUT_GATES')
    _, streams = load_lane_streams(params['stream'], require_independent=True)
    if any(stream['duration_ms'] != 1_800_000 for stream in streams):
        raise ValueError('COMMAND_REPLAY_LONG_DURATION')
    transitions = LANES*18000
    return dict(schema_version='r6-local-runtime-contract/1', execution_mode='local-direct', operation='dynamic',
                mode=MODE, parameters=params, budgets=dict(wall_seconds=7200, live_process_starts=1,
                    simulator_initializations=1, initialization_ticks=10, training_transitions=0,
                    nontraining_transitions=transitions, optimizer_updates=0, checkpoint_writes=0,
                    explicit_resets=1, automatic_resets=transitions))


def contract(params):
    run_id = params['experiment_id']
    if run_id.startswith(LONG_PREFIX):
        return long_contract(params)
    guard = run_id.startswith(GUARD_PREFIX)
    crazy = run_id.startswith(CRAZY_PREFIX)
    endurance = run_id.startswith((ENDURANCE_PREFIX, HOVER_PREFIX))
    returns = run_id.startswith((RETURN_PREFIX, EARLY_PREFIX))
    routes = route_layouts(run_id)
    selected_pair = next((pair for prefix, pair in EXECUTION_PAIRS.items() if run_id.startswith(prefix)), None)
    control_execution, candidate_execution = selected_pair or (EXECUTION, BRAKE_EXECUTION)
    paired_weights = selected_pair is not None
    expected_models = MODELS
    if paired_weights:
        expected_models = {'control': MODELS['b'], 'candidate': MODELS['b']}
    elif routes:
        expected_models = dict(MODELS, candidate=MODELS['b'])
    fields = {'experiment_id', 'training_seed', 'seed', 'models', 'execution', 'stream', 'wall_seconds', 'output_cap_bytes'}
    if guard or routes:
        fields.add('candidate_execution')
    if (set(params) != fields or not matches(params['experiment_id']) or params['training_seed'] != 20260908
            or ((guard or routes) and params['candidate_execution'] != candidate_execution)
            or type(params['seed']) is not int
            or not 0 <= params['seed'] < 2**31 or params['execution'] != control_execution
            or not isinstance(params['models'], dict) or set(params['models']) != set(expected_models)
            or type(params['wall_seconds']) is not int or not 120 <= params['wall_seconds'] <= 1800
            or type(params['output_cap_bytes']) is not int or not 256*2**20 <= params['output_cap_bytes'] <= 2**30):
        raise ValueError('COMMAND_REPLAY_PARAMETERS')
    for label, checkpoint in params['models'].items():
        if checkpoint['sha256'] != expected_models[label]:
            raise ValueError('COMMAND_REPLAY_FROZEN_CHECKPOINT')
        checked(checkpoint)
    if paired_weights and params['models']['control'] != params['models']['candidate']:
        raise ValueError('COMMAND_REPLAY_GUARD_SAME_WEIGHTS')
    if routes and not paired_weights and params['models']['b'] != params['models']['candidate']:
        raise ValueError('COMMAND_REPLAY_ROUTES_SAME_WEIGHTS')
    stream = load_lane_streams(params['stream'])[1][16] if routes else load_stream(params['stream'])
    duration = stream['duration_ms']
    valid_duration = 101000 <= duration <= 125000 if endurance or returns else duration <= 90000
    if not valid_duration or ((crazy or endurance or returns) and stream['source'] != 'crazy_command_generator'):
        raise ValueError('COMMAND_REPLAY_DEVELOPMENT_DURATION')
    events = {event['t_ms']: event['velocity_w_m_s'] for event in stream['events']}
    reversal = CONDITIONS['crossing_at_reversal']
    previous = next(event['velocity_w_m_s'] for event in reversed(stream['events']) if event['t_ms'] < reversal)
    if events.get(reversal, [0.])[0] >= 0 or previous[0] <= 0 or stream['duration_ms'] <= reversal + 4000:
        raise ValueError('COMMAND_REPLAY_PAIRED_REVERSAL')
    trials = len(params['models']) * len(conditions(params))
    transitions = trials * LANES * ((RETURN_DURATION_MS if returns else duration)//STEP_MS)
    return dict(schema_version='r6-local-runtime-contract/1', execution_mode='local-direct',
                operation='dynamic', mode=MODE, parameters=params,
                budgets=dict(wall_seconds=params['wall_seconds'], live_process_starts=1,
                             simulator_initializations=1, initialization_ticks=10, training_transitions=0,
                             nontraining_transitions=transitions, optimizer_updates=0, checkpoint_writes=0,
                             explicit_resets=trials, automatic_resets=transitions))


def run(backend, params):
    import torch
    from instinctlab.control.escape_option import METHOD_FIELD
    from instinctlab.control.lee_velocity_controller import rotate_world_to_body_wxyz
    from instinctlab.training.checkpoint_h1 import checkpoint_actor
    from instinctlab.training.networks import NetworkConfig
    from training_workloads.scale_003.persistence import atomic_json_replace, reserve_output_bytes
    from .lidar_calibration import record as record_mount

    layouts, event_ids, events = event_schedule(params)
    env, counters = backend.environment, backend.counters
    budgets = backend.runtime_contract['budgets']
    if route_layouts(params['experiment_id']):
        commands = env._command_b.new_tensor([[event['velocity_w_m_s'] for event in row] for row in events])
    else:
        commands = env._command_b.new_tensor([event['velocity_w_m_s'] for event in events])
        commands = commands[:, None, :].expand(-1, LANES, -1)
    obstacle = env._h1_moving_sphere
    axis = env._command_b.new_zeros((LANES, 3))
    axis[:, 0] = 1.
    street = torch.tensor([layout == 'street_gap' for layout in layouts], device=env.device)
    steps = len(event_ids)-1
    if not math.isclose(env.step_dt, STEP_MS/1000., abs_tol=1e-8):
        raise ValueError('COMMAND_REPLAY_CLOCK')
    if env.max_episode_length <= steps:
        raise ValueError('COMMAND_REPLAY_EPISODE_HORIZON')
    block = dict(id='continuous-replay-pair', task='stop', speed_m_s=.25,
                 pulse_acceleration_m_s2=0., sensor_to_wall_distance_m=None, steps=steps)
    output = Path(backend.output)
    endurance = params['experiment_id'].startswith((ENDURANCE_PREFIX, HOVER_PREFIX))
    returns = params['experiment_id'].startswith((RETURN_PREFIX, EARLY_PREFIX))
    long_run = params['experiment_id'].startswith(LONG_PREFIX)
    timed_crossings = endurance or returns or long_run
    env.capture_ics_limiter = timed_crossings or params['experiment_id'].startswith(CRAZY_PREFIX)

    def save_row(handle, row):
        data = (json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n').encode()
        reserve_output_bytes(Path(handle.name), len(data))
        handle.write(data)

    results = {}
    for label, checkpoint in params['models'].items():
        payload = torch.load(checked(checkpoint), map_location='cpu', weights_only=True)
        if (NetworkConfig(**payload['network_config']) != NetworkConfig()
                or payload['method_hashes'].get(METHOD_FIELD) != EXECUTION):
            raise ValueError('COMMAND_REPLAY_EXECUTION_BINDING')
        actor = checkpoint_actor(payload).to(env.device)
        actor.load_state_dict(payload['actor'], strict=True)
        actor.eval()
        del payload
        execution = arm_execution(params, label)
        env.set_escape_option(True, version=execution)
        env.set_h1_ics_clearance_margin(.15)
        model = dict(checkpoint=checkpoint, execution=execution, lidar_mount=record_mount(env, output, label), trials={})
        if timed_crossings:
            configuration = {(p.config.ics.motion_tube_radius_m, p.config.ics.motion_tube_confirm_closure)
                             for p in env._control_pipelines}
            confirm = execution in (TUBE_CONFIRM_EXECUTION, RECOVERY_RETREAT_EXECUTION, EARLY_RETREAT_EXECUTION,
                                    SIDE_HOVER_RETREAT_EXECUTION)
            if configuration != {(.60, confirm)}:
                raise ValueError('COMMAND_REPLAY_ICS_PROFILE')
            model['ics_profile'] = dict(radius_m=.60, confirms_closure=confirm)
        for name, crossing_ms in conditions(params).items():
            env.configure_r6_physics(block=block, seed=params['seed'], split='development',
                                     reset_sink=lambda row: None, force_sink=lambda row: None,
                                     expected_mass_kg=1.181517923)
            sensor_origins = env.scene.env_origins + axis.new_tensor(
                [spec['sensor_position_local_m'] for spec in env._r6_physics['specs']])
            modes = torch.full((LANES,), 7, device=env.device, dtype=torch.long)
            delay = None
            if crossing_ms is not None:
                ids = obstacle.indices
                modes[street] = 2 + ids[street].remainder(2)
                delay = crossing_ms/1000. - CROSSING_START_M/CROSSING_SPEED_M_S
            obstacle.configure(sensor_origins, axis, modes, departure_delay_s=delay,
                               half_period_s=CROSSING_TRAVEL_S if timed_crossings else None,
                               exact_clock=long_run)
            obstacle.end_travel_s[street & obstacle.active] = CROSSING_TRAVEL_S
            env.set_h1_initial_command(torch.zeros_like(env._command_b), 0)
            env.stage_r6_world_command(commands[0])
            backend.resource_check('before_command_replay_reset', counters.explicit_resets+1)
            observation, _ = backend._preserving_reset(seed=params['seed'])
            trace = output/f'{label}-{name}.jsonl'
            # Stream completed transitions to disk. A failed run still leaves its prefix intact.
            with trace.open('xb') as handle:
                save_row(handle, dict(kind='initial', mover_start_w=obstacle.start.cpu().tolist(),
                                      path_reference_w=sensor_origins.cpu().tolist(),
                                      scene_origins_w=env.scene.env_origins.cpu().tolist(),
                                      root_state_w=env._robot.data.root_state_w.cpu().tolist(),
                                      departure_delay_s=delay, layouts=layouts))
                for step in range(steps):
                    backend.resource_check('before_command_replay_step', step)
                    counters.require_step_capacity(kind='nontraining', lanes=LANES,
                                                   maximum=budgets['nontraining_transitions'])
                    counters.require_reset_capacity(automatic=True, lanes=LANES,
                                                    maximum=budgets['automatic_resets'])
                    policy = observation['policy']
                    attitude = env._robot.data.root_state_w[:, 3:7].clone()
                    body = rotate_world_to_body_wxyz(attitude, commands[step])
                    actor_state = policy['state_vec'].reshape(LANES, 4, 13)
                    if (not torch.allclose(actor_state[:, -1, 6:9], body, atol=1e-6, rtol=0)
                            or not torch.equal(actor_state[:, -1, 6:9], env._command_b)):
                        raise RuntimeError(f'COMMAND_REPLAY_ACTOR_COMMAND {label}/{name}/{step}')
                    before = obstacle.latest
                    grid = policy['lidar_grid'].reshape(LANES, 4, 3, -1)
                    witness = grid[obstacle.indices, :, :, before['witness_beam']]
                    valid_count = grid[:, -1, 1].sum(dim=1)
                    env.stage_r6_world_command(commands[step+1])
                    env.prepare_h1_step(global_policy_step=step, command_b=env._command_b,
                                        next_command_b=rotate_world_to_body_wxyz(attitude, commands[step+1]))
                    with torch.no_grad():
                        action = actor(policy).mean_action
                    counters.attempt_step()
                    observation, _, terminated, truncated, extras = env.step(action)
                    counters.complete_step(LANES, kind='nontraining')
                    terminal = (terminated | truncated).reshape(-1)
                    counters.reserve_reset(automatic=True, lanes=int(terminal.sum()),
                                           maximum=budgets['automatic_resets'])
                    pre = extras['pre_reset']
                    task, control, option = pre['task'], pre['control'], pre['escape_option']
                    post = obstacle.pre_reset
                    state = pre['root_state_w']
                    tensors = dict(
                        attitude_wxyz=attitude, actor_state=actor_state, action=action,
                        actor_witness=witness, valid_beams=valid_count, mover_beam=before['witness_beam'],
                        mover_winning_beams=before['winning_beams'], mover_witness_range_m=before['witness_range_m'],
                        mover_position_w=before['sphere_position_w'], mover_position_after_w=post['sphere_position_w'],
                        mover_velocity_w=before['sphere_velocity_w'], v_corr_b=control['v_corr_b'],
                        v_gov_b=control['v_gov_b'], v_final_b=control['v_final_b'], v_final_w=control['v_final_w'],
                        ics_beta=control['ics_metrics']['beta'].reshape(-1),
                        ics_emergency=control['ics_metrics']['emergency'].reshape(-1),
                        ics_active_beams=control['ics_metrics']['active_beam_count'].reshape(-1),
                        selected_action=option['applied_action'], escape_active=option['active'],
                        escape_recovering=option['recovery'], escape_redirected=option['redirected'],
                        root_state_after_w=state, ray_minimum_m=task['exact_clearance_m'].reshape(-1),
                        collision=task['termination']['masks']['collision'].reshape(-1),
                        termination_reason=task['termination']['reason_code'].reshape(-1), reset=terminal,
                        episode_length=pre['episode_length'])
                    if 'braking' in option:
                        tensors['escape_braking'] = option['braking']
                    if 'retreating' in option:
                        tensors['recovery_retreating'] = option['retreating']
                    if 'ics_limiter' in pre:
                        tensors.update(pre['ics_limiter'])
                    save_row(handle, dict(kind='transition', step=step, t_ms=step*STEP_MS,
                                          event_index=event_ids[step], raw_event=events[step],
                                          physics_tick=int(before['tick']), after_tick=int(post['tick']),
                                          contact_samples=post['contact_samples'],
                                          **{key: value.detach().cpu().tolist() for key, value in tensors.items()}))
                    handle.flush()
            completion = env.finish_r6_physics()
            obstacle.finish()
            model['trials'][name] = dict(trace=file_pin(trace), physics=completion)
        results[label] = model
    atomic_json_replace(output, 'command_replay.json', dict(models=results, counters=counters.snapshot()))
