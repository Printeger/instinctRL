"""Continuous, development-only command replay on the frozen B/C plant."""
from bisect import bisect_right
import json
import math
from pathlib import Path
import re

from tools.r6_evidence import checked, file_pin
from .escape_holdout_scene import LAYOUTS


PREFIX = 'R6-DIRECT-INSTINCT-GPU-COMMAND-'
GUARD_PREFIX = PREFIX + 'GUARD-'
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


def matches(run_id):
    return isinstance(run_id, str) and re.fullmatch(PREFIX + r'(?:PAIR|GUARD)-[0-9]{2}', run_id) is not None


def arm_execution(params, label):
    return params.get('candidate_execution', params['execution']) if label == 'candidate' else params['execution']


def load_stream(pin):
    stream = json.loads(checked(pin).read_bytes())
    if (set(stream) != {'schema_version', 'source', 'duration_ms', 'events'}
            or stream['schema_version'] != 'instinctrl-world-command-stream/1'
            or stream['source'] != 'scripted_development'
            or type(stream['duration_ms']) is not int or stream['duration_ms'] % STEP_MS
            or not 8000 <= stream['duration_ms'] <= 60000
            or not isinstance(stream['events'], list) or not stream['events']):
        raise ValueError('COMMAND_REPLAY_STREAM')
    last = -1
    for event in stream['events']:
        if not isinstance(event, dict) or set(event) != {'t_ms', 'velocity_w_m_s'}:
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


def sampled_events(stream):
    """Sample the latest event at each boundary, including the final observation."""
    events = stream['events']
    times = [event['t_ms'] for event in events]
    return [bisect_right(times, t) - 1 for t in range(0, stream['duration_ms'] + 1, STEP_MS)]


def contract(params):
    guard = params['experiment_id'].startswith(GUARD_PREFIX)
    expected_models = {'control': MODELS['b'], 'candidate': MODELS['b']} if guard else MODELS
    fields = {'experiment_id', 'training_seed', 'seed', 'models', 'execution', 'stream', 'wall_seconds', 'output_cap_bytes'}
    if guard:
        fields.add('candidate_execution')
    if (set(params) != fields or not matches(params['experiment_id']) or params['training_seed'] != 20260908
            or (guard and params['candidate_execution'] != BRAKE_EXECUTION)
            or type(params['seed']) is not int
            or not 0 <= params['seed'] < 2**31 or params['execution'] != EXECUTION
            or not isinstance(params['models'], dict) or set(params['models']) != set(expected_models)
            or type(params['wall_seconds']) is not int or not 120 <= params['wall_seconds'] <= 1800
            or type(params['output_cap_bytes']) is not int or not 256*2**20 <= params['output_cap_bytes'] <= 2**30):
        raise ValueError('COMMAND_REPLAY_PARAMETERS')
    for label, checkpoint in params['models'].items():
        if checkpoint['sha256'] != expected_models[label]:
            raise ValueError('COMMAND_REPLAY_FROZEN_CHECKPOINT')
        checked(checkpoint)
    if guard and params['models']['control'] != params['models']['candidate']:
        raise ValueError('COMMAND_REPLAY_GUARD_SAME_WEIGHTS')
    stream = load_stream(params['stream'])
    events = {event['t_ms']: event['velocity_w_m_s'] for event in stream['events']}
    reversal = CONDITIONS['crossing_at_reversal']
    previous = next(event['velocity_w_m_s'] for event in reversed(stream['events']) if event['t_ms'] < reversal)
    if events.get(reversal, [0.])[0] >= 0 or previous[0] <= 0 or stream['duration_ms'] <= reversal + 4000:
        raise ValueError('COMMAND_REPLAY_PAIRED_REVERSAL')
    trials = len(params['models']) * len(CONDITIONS)
    transitions = trials * LANES * (stream['duration_ms']//STEP_MS)
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

    stream = load_stream(params['stream'])
    events = stream['events']
    event_ids = sampled_events(stream)
    env, counters = backend.environment, backend.counters
    budgets = backend.runtime_contract['budgets']
    commands = env._command_b.new_tensor([events[i]['velocity_w_m_s'] for i in event_ids])
    commands = commands[:, None, :].expand(-1, LANES, -1)
    obstacle = env._h1_moving_sphere
    axis = env._command_b.new_zeros((LANES, 3))
    axis[:, 0] = 1.
    street = torch.tensor([LAYOUTS[i//8] == 'street_gap' for i in range(LANES)], device=env.device)
    steps = len(event_ids)-1
    if not math.isclose(env.step_dt, STEP_MS/1000., abs_tol=1e-8):
        raise ValueError('COMMAND_REPLAY_CLOCK')
    block = dict(id='continuous-replay-pair', task='stop', speed_m_s=.25,
                 pulse_acceleration_m_s2=0., sensor_to_wall_distance_m=None, steps=steps)
    output = Path(backend.output)

    def save_row(handle, row):
        data = (json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n').encode()
        reserve_output_bytes(Path(handle.name), len(data))
        handle.write(data)

    results = {}
    for label, checkpoint in params['models'].items():
        payload = torch.load(checked(checkpoint), map_location='cpu', weights_only=True)
        if (NetworkConfig(**payload['network_config']) != NetworkConfig()
                or payload['method_hashes'].get(METHOD_FIELD) != params['execution']):
            raise ValueError('COMMAND_REPLAY_EXECUTION_BINDING')
        actor = checkpoint_actor(payload).to(env.device)
        actor.load_state_dict(payload['actor'], strict=True)
        actor.eval()
        del payload
        execution = arm_execution(params, label)
        env.set_escape_option(True, version=execution)
        env.set_h1_ics_clearance_margin(.15)
        model = dict(checkpoint=checkpoint, execution=execution, lidar_mount=record_mount(env, output, label), trials={})
        for name, crossing_ms in CONDITIONS.items():
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
            obstacle.configure(sensor_origins, axis, modes, departure_delay_s=delay)
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
                                      root_state_w=env._robot.data.root_state_w.cpu().tolist(),
                                      departure_delay_s=delay))
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
                    save_row(handle, dict(kind='transition', step=step, t_ms=step*STEP_MS,
                                          event_index=event_ids[step], raw_event=events[event_ids[step]],
                                          physics_tick=int(before['tick']), after_tick=int(post['tick']),
                                          contact_samples=post['contact_samples'],
                                          **{key: value.detach().cpu().tolist() for key, value in tensors.items()}))
                    handle.flush()
            completion = env.finish_r6_physics()
            obstacle.finish()
            model['trials'][name] = dict(trace=file_pin(trace), physics=completion)
        results[label] = model
    atomic_json_replace(output, 'command_replay.json', dict(models=results, counters=counters.snapshot()))
