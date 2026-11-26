"""Independent Isaac closed-loop dynamic assay, with exact bounded denominators."""
import json
import math
from pathlib import Path
import re
from statistics import mean

from tools.r6_evidence import checked, file_pin
from . import pipe_authority_probe


MODE = 'research-r6-dynamic-assay'
STEPS = 120
LANES = 32
GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7))
VISIBILITY_GROUPS = ((8, 9, 10, 11),)
PERSISTENCE_RUN = 'R6-DIRECT-INSTINCT-GPU-R4-DEV-PERSISTENCE-02'
PERSISTENCE_GROUPS = (*VISIBILITY_GROUPS, (4, 1, 6, 7))
OPTION_RUN = 'R6-DIRECT-INSTINCT-GPU-R4-DEV-OPTION'
HOLDOUT_RUN = 'R6-DIRECT-INSTINCT-GPU-R4-LONG-HOLDOUT'
HOLDOUT_SMOKE = 'R6-DIRECT-INSTINCT-GPU-R4-HOLDOUT-SMOKE'
RECOVERY_PREFIX = 'R6-DIRECT-INSTINCT-GPU-R5-RECOVERY-'
THREE_PREFIX = RECOVERY_PREFIX + 'THREE-'
NEXT_LONG_PREFIX = RECOVERY_PREFIX + 'NEXT-LONG-'
U1792_STUDY_PREFIX = RECOVERY_PREFIX + 'U1792-TWO-STUDY-'
RECOVERY_MECHANISM = RECOVERY_PREFIX + 'MECHANISM-01'
RECOVERY_MECHANISM_RETRY = RECOVERY_PREFIX + 'MECHANISM-02'
RECOVERY_MECHANISM_RUNS = (RECOVERY_MECHANISM, RECOVERY_MECHANISM_RETRY)
RECOVERY_REAR = RECOVERY_PREFIX + 'REAR-01'
RECOVERY_HOLDOUT = RECOVERY_PREFIX + 'HOLDOUT'
RECOVERY_SHORT_CHECKS = tuple(RECOVERY_PREFIX + f'SHORT-CHECK-U{u}' for u in (656, 672, 704))
HOLDOUT_GROUPS = ((8, 8, 8, 8), (0, 0, 0, 0), (2, 3, 2, 4),
                  (6, 6, 6, 6), (7, 7, 7, 7), (9, 9, 9, 9))
PHYSICAL_LAYOUT_RUNS = (HOLDOUT_RUN, HOLDOUT_SMOKE, *RECOVERY_MECHANISM_RUNS, RECOVERY_REAR,
                        RECOVERY_HOLDOUT, *RECOVERY_SHORT_CHECKS)
MODES = ('approach', 'recede', 'cross_left', 'cross_right',
         'hover', 'hover_wall', 'static_travel', 'static_hover',
         'open_front', 'open_rear', 'wall_front', 'wall_rear')


def three_case(run_id):
    if run_id == pipe_authority_probe.PIPE_PROBE_RUN:
        return 'PIPE-AUTHORITY'
    if run_id == pipe_authority_probe.PIPE_SWITCH_RUN:
        return 'PIPE-SWITCH'
    if run_id == pipe_authority_probe.PIPE_CLEARANCE_RUN:
        return 'PIPE-CLEARANCE'
    match = re.fullmatch(THREE_PREFIX + r'(REAR|MECHANISM|CHECK|HOLDOUT)-[A-Z0-9-]+', run_id)
    if match is None:
        match = re.fullmatch(NEXT_LONG_PREFIX + r'(REARPAIR|MECHANISM|CROSSCHECK|CHECK|HOLDOUT)-[A-Z0-9-]+', run_id)
    if match is None:
        match = re.fullmatch(U1792_STUDY_PREFIX + r'(CHECK|HOLDOUT|MECHANISM|CROSSCHECK)-[A-Z0-9-]+', run_id)
    return match[1] if match else None


def physical_layout(run_id):
    return run_id in PHYSICAL_LAYOUT_RUNS or three_case(run_id) is not None


def groups_for(run_id):
    case = three_case(run_id)
    if case in ('PIPE-AUTHORITY', 'PIPE-SWITCH', 'PIPE-CLEARANCE'):
        return tuple(HOLDOUT_GROUPS[i] for i in (1, 5, 4))
    if case in ('REAR', 'REARPAIR'):
        return (HOLDOUT_GROUPS[-1],)
    if case == 'MECHANISM' and run_id.startswith(NEXT_LONG_PREFIX):
        return (*HOLDOUT_GROUPS, (1, 1, 1, 1))
    if case in ('MECHANISM', 'CROSSCHECK', 'CHECK', 'HOLDOUT'):
        return HOLDOUT_GROUPS
    if run_id == PERSISTENCE_RUN:
        return PERSISTENCE_GROUPS
    if run_id in (HOLDOUT_RUN, RECOVERY_HOLDOUT):
        return HOLDOUT_GROUPS
    if run_id in (*RECOVERY_MECHANISM_RUNS, *RECOVERY_SHORT_CHECKS):
        return tuple(HOLDOUT_GROUPS[i] for i in (1, 3, 4))
    if run_id == RECOVERY_REAR:
        return (HOLDOUT_GROUPS[-1],)
    if run_id == HOLDOUT_SMOKE:
        return (HOLDOUT_GROUPS[1], HOLDOUT_GROUPS[2])
    return VISIBILITY_GROUPS if run_id.endswith('-VISIBILITY') else GROUPS


def contract(params):
    fields = {'experiment_id', 'training_seed', 'models', 'seed', 'wall_seconds', 'output_cap_bytes'}
    runs = (PERSISTENCE_RUN, OPTION_RUN, *PHYSICAL_LAYOUT_RUNS)
    case = three_case(params['experiment_id'])
    next_long = params['experiment_id'].startswith(NEXT_LONG_PREFIX)
    u1792_study = params['experiment_id'].startswith(U1792_STUDY_PREFIX)
    clearance_confirmation = u1792_study and case == 'MECHANISM'
    if case:
        fields.add('arm_options')
    if set(params) != fields or (not case and params['experiment_id'] not in runs and not re.fullmatch(
            r'R6-DIRECT-INSTINCT-GPU-R[23]-(?:DYNAMIC(?:-PILOT-0[1-4])?|VISIBILITY)', params['experiment_id'])):
        raise ValueError('DYNAMIC_ASSAY_CONTRACT_IDENTITY')
    if params['training_seed'] != 20260908 or not 0 <= params['seed'] < 2**31:
        raise ValueError('DYNAMIC_ASSAY_SEED')
    if type(params['wall_seconds']) is not int or not 120 <= params['wall_seconds'] <= 1800:
        raise ValueError('DYNAMIC_ASSAY_WALL')
    if type(params['output_cap_bytes']) is not int or not 512*2**20 <= params['output_cap_bytes'] <= 2**30:
        raise ValueError('DYNAMIC_ASSAY_OUTPUT')
    models = params['models']
    three_arms = params['experiment_id'] in (HOLDOUT_RUN, *RECOVERY_MECHANISM_RUNS, RECOVERY_HOLDOUT)
    allowed_models = {'candidate', 'control', 'reference'} if three_arms else {'candidate', 'reference'}
    if case in ('HOLDOUT', 'MECHANISM', 'CROSSCHECK', 'CHECK'):
        allowed_models.add('control')
    if case == 'PIPE-AUTHORITY':
        allowed_models = {'candidate'}
    if next_long and case == 'HOLDOUT':
        allowed_models.add('prior')
    if case == 'REAR':
        allowed_models.update(('candidate_slow', 'reference_slow'))
    if not isinstance(models, dict) or not 1 <= len(models) <= len(allowed_models) or not set(models) <= allowed_models:
        raise ValueError('DYNAMIC_ASSAY_MODELS')
    for value in models.values():
        checked(value)
    if case:
        from instinctlab.control.escape_recovery import (
            FALLBACK_25_VERSION, FAST_LATERAL_VERSION, FRONT_SWITCH_25_VERSION,
            LATERAL_VERSION, MOUNT_PITCH_DEGREES, REAR_CLEARANCE_25_VERSION, REAR_SWITCH_25_VERSION, RECOVERY_VERSION,
        )

        options = params['arm_options']
        expected_models = allowed_models
        if (case == 'CHECK' or clearance_confirmation or (next_long and case == 'MECHANISM')) and 'control' not in models:
            expected_models = {'candidate', 'reference'}
        if next_long and case == 'HOLDOUT' and 'prior' not in models:
            expected_models = allowed_models - {'prior'}
        if (not isinstance(options, dict) or set(options) != expected_models or set(models) != expected_models
                or not all(version in (None, RECOVERY_VERSION, LATERAL_VERSION, FAST_LATERAL_VERSION, *MOUNT_PITCH_DEGREES)
                           for version in options.values())):
            raise ValueError('DYNAMIC_THREE_OPTIONS')
        if case in ('REAR', 'REARPAIR', 'MECHANISM') and any(value != models['reference'] for value in models.values()):
            raise ValueError('DYNAMIC_THREE_SAME_WEIGHTS')
        if case == 'REAR' and options != dict(reference=RECOVERY_VERSION, candidate=LATERAL_VERSION,
                                             reference_slow=RECOVERY_VERSION, candidate_slow=LATERAL_VERSION):
            raise ValueError('DYNAMIC_REAR_FACTORIAL')
        if case == 'CHECK' and (options['candidate'] is None or len(set(options.values())) != 1):
            raise ValueError('DYNAMIC_THREE_CHECK_EXECUTION')
        if case in ('PIPE-AUTHORITY', 'PIPE-SWITCH', 'PIPE-CLEARANCE'):
            from .robustness_training_plan import U1792_CHECKPOINT_SHA

            expected = {
                'PIPE-AUTHORITY': dict(candidate=FALLBACK_25_VERSION),
                'PIPE-SWITCH': dict(candidate=REAR_SWITCH_25_VERSION, reference=FALLBACK_25_VERSION),
                'PIPE-CLEARANCE': dict(candidate=REAR_CLEARANCE_25_VERSION, reference=FALLBACK_25_VERSION),
            }[case]
            if options!=expected or any(pin['sha256']!=U1792_CHECKPOINT_SHA or pin!=models['candidate']
                                        for pin in models.values()):
                raise ValueError('DYNAMIC_PIPE_AUTHORITY_FIXED_ORIGIN')
        if clearance_confirmation:
            from .robustness_training_plan import U1792_CHECKPOINT_SHA

            if (options != dict(candidate=REAR_CLEARANCE_25_VERSION, reference=FALLBACK_25_VERSION)
                    or models['candidate']['sha256'] != U1792_CHECKPOINT_SHA):
                raise ValueError('DYNAMIC_PIPE_CLEARANCE_CONFIRMATION_ORIGIN')
        if case == 'CROSSCHECK' and u1792_study:
            from .robustness_training_plan import U1792_CHECKPOINT_SHA

            if (models['reference'] != models['control'] or models['candidate'] == models['control']
                    or models['control']['sha256'] != U1792_CHECKPOINT_SHA
                    or options != dict(candidate=REAR_CLEARANCE_25_VERSION, control=REAR_CLEARANCE_25_VERSION,
                                       reference=FALLBACK_25_VERSION)):
                raise ValueError('DYNAMIC_REAR_CLEARANCE_CROSSCHECK')
        if case == 'CROSSCHECK' and not u1792_study and (
                models['candidate'] != models['control'] or models['reference'] == models['control']
                or options != dict(candidate=FRONT_SWITCH_25_VERSION, control=FALLBACK_25_VERSION,
                                   reference=FRONT_SWITCH_25_VERSION)):
            raise ValueError('DYNAMIC_FRONT_SWITCH_PAIR')
        if case == 'HOLDOUT' and (options['reference'] is not None or options['candidate'] is None
                                  or options['candidate'] != options['control']):
            raise ValueError('DYNAMIC_THREE_HOLDOUT_EXECUTION')
        if next_long and case == 'HOLDOUT' and 'prior' in models:
            if options['prior'] is None or (models['control'], options['control']) == (models['prior'], options['prior']):
                raise ValueError('DYNAMIC_NEXT_LONG_DISTINCT_PRIOR')
    if params['experiment_id'] in (PERSISTENCE_RUN, OPTION_RUN) and (
            set(models) != {'candidate', 'reference'} or models['candidate'] != models['reference']):
        raise ValueError('DYNAMIC_ASSAY_SAME_WEIGHT_MODELS')
    if (three_arms or params['experiment_id'] in (RECOVERY_REAR, *RECOVERY_SHORT_CHECKS)) and set(models) != allowed_models:
        raise ValueError('DYNAMIC_HOLDOUT_COMPLETE_COMPARISON')
    if params['experiment_id'] in (*RECOVERY_MECHANISM_RUNS, RECOVERY_REAR) and any(
            pin != models['reference'] for pin in models.values()):
        raise ValueError('DYNAMIC_RECOVERY_SAME_WEIGHT_MODELS')
    cohorts = len(models) * len(groups_for(params['experiment_id']))
    steps = cohorts * STEPS * LANES
    return dict(schema_version='r6-local-runtime-contract/1', execution_mode='local-direct',
                operation='dynamic', mode=MODE, parameters=params,
                budgets=dict(live_process_starts=1, simulator_initializations=1,
                             initialization_ticks=10, training_transitions=0,
                             optimizer_updates=0, checkpoint_writes=0, nontraining_transitions=steps,
                             explicit_resets=cohorts, automatic_resets=steps,
                             wall_seconds=params['wall_seconds']))


def score(rows):
    """Original lanes remain in the denominator after their first terminal step."""
    lanes = []
    for lane in range(LANES):
        alive = []
        for row in rows:
            alive.append(row)
            if row['terminal'][lane]:
                break
        last = alive[-1]
        axis = rows[0]['axis_w'][lane]
        initial = rows[0]['initial_position_w'][lane]
        delta = [x-y for x,y in zip(last['vehicle_position_w'][lane], initial)]
        clearance = [r['post']['sphere_sensor_clearance_m'][lane] for r in alive if r['post']['active'][lane]]
        # Penalize missed tracking time after a terminal transition.
        speed2 = [sum((v-c)**2 for v,c in zip(r['velocity_w'][lane],r['command_w'][lane])) for r in alive]
        command_sq = sum(v*v for v in rows[0]['command_w'][lane])
        speed2.extend([command_sq]*(STEPS-len(alive)))
        metrics = dict(lane=lane, mode=MODES[rows[0]['modes'][lane]],
                       collision=any(r['collision'][lane] for r in alive), terminal=last['terminal'][lane],
                       sphere_surface_gap_m=min(clearance) if clearance else None,
                       progress_m=sum(x*u for x,u in zip(delta,axis)),
                       horizontal_displacement_m=math.hypot(*delta[:2]), tracking_rmse_mps=math.sqrt(mean(speed2)))
        if 'sensor_approach' in rows[0]:
            metrics.update(first_visible_step=next((r['step'] for r in alive if r['pre']['winning_beams'][lane]), None),
                           first_closure_step=next((r['step'] for r in alive if r['pre']['winning_beams'][lane]
                                                    and r['sensor_approach'][lane]), None),
                           first_contact_step=next((r['step'] for r in alive if r['collision'][lane]), None))
        if 'reflex_direction_b' in rows[0]:
            detections = [r for r in alive if any(r['reflex_direction_b'][lane])]
            first_visible = next((r for r in detections if r['pre']['winning_beams'][lane]), None)
            metrics['reflex_steps'] = len(detections)
            metrics['first_visible_reflex'] = None
            if first_visible is not None:
                direction = first_visible['reflex_direction_b'][lane]
                metrics['first_visible_reflex'] = dict(
                    step=first_visible['step'],
                    governor_mps=sum(x*d for x, d in zip(first_visible['v_gov_b'][lane], direction)),
                    final_mps=sum(x*d for x, d in zip(first_visible['v_final_b'][lane], direction)),
                )
        if 'layout' in rows[0]:
            metrics.update(layout=rows[0]['layout'][lane],
                           recovery_rmse_mps=math.sqrt(mean(speed2[80:])),
                           maximum_horizontal_excursion_m=max(math.hypot(
                               r['vehicle_position_w'][lane][0]-initial[0], r['vehicle_position_w'][lane][1]-initial[1]) for r in alive))
        if 'escape_recovery' in rows[0]:
            metrics['recovery_progress_m'] = 0.
            if len(alive) > 80:
                previous = alive[79]['vehicle_position_w'][lane]
                metrics['recovery_progress_m'] = sum((x-y)*a for x,y,a in zip(
                    last['vehicle_position_w'][lane], previous, axis))
        lanes.append(metrics)
    return lanes


def measurements(trace):
    return {k: v.detach().cpu().tolist() if hasattr(v, 'detach') else v for k,v in trace.items()}


def visibility_reset(block, *, seed, lane, mode, mount, split='development'):
    from .robustness_scenarios import reset_spec, rotate_yaw

    spec = reset_spec(block, seed=seed, split=split, lane=lane % 8, mount_position_m=mount)
    sensor = spec['sensor_position_local_m']
    if mode in (10, 11):
        sensor[0] = 4.60
    yaw = math.radians(spec['yaw_degrees'] + (180. if mode in (8, 10) else 0.))
    offset = rotate_yaw(mount, yaw)
    return dict(spec, lane=lane, direction_world=[-1., 0., 0.], yaw_degrees=math.degrees(yaw),
                root_position_local_m=[x-y for x, y in zip(sensor, offset)],
                root_quaternion_wxyz=[math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])


def run(backend, params):
    import torch
    from instinctlab.training.checkpoint_h1 import checkpoint_actor
    from instinctlab.training.hover_reflex import HoverReflex
    from instinctlab.training.networks import NetworkConfig
    from instinctlab.control.lee_velocity_controller import rotate_world_to_body_wxyz
    from instinctlab.control.escape_recovery import MOUNT_PITCH_DEGREES
    from training_workloads.scale_003.persistence import atomic_json_replace, reserve_output_bytes
    from .dynamic_motion_training import plausible_approach
    from .escape_hold_probe import EscapeHoldProbe
    from .escape_holdout_scene import LAYOUTS
    from .lidar_calibration import record as record_mount
    from .robustness_scenarios import reset_spec

    env = backend.environment
    obstacle = env._h1_moving_sphere
    budgets = backend.runtime_contract['budgets']
    counters = backend.counters
    device = env.device
    groups = groups_for(params['experiment_id'])
    persistence = params['experiment_id'] == PERSISTENCE_RUN
    run_id = params['experiment_id']
    three = three_case(run_id)
    holdout = physical_layout(run_id)
    mechanism = run_id in RECOVERY_MECHANISM_RUNS
    rear_pair = run_id == RECOVERY_REAR
    recovery_assay = run_id.startswith(RECOVERY_PREFIX)
    mount_assay = any(version in MOUNT_PITCH_DEGREES for version in params.get('arm_options', {}).values())
    option = holdout or params['experiment_id'] == OPTION_RUN
    pipe_probe = three == 'PIPE-AUTHORITY'
    mirror_pipe = run_id in pipe_authority_probe.MIRRORED_PIPE_RUNS
    approved_frames = pipe_authority_probe.open_pool(backend.output) if pipe_probe else None
    split = 'heldout' if three == 'HOLDOUT' or run_id in (HOLDOUT_RUN, RECOVERY_HOLDOUT) else 'development'
    cue = (HoverReflex(device=device, dtype=torch.float32)
           if persistence or option or params['experiment_id'].startswith('R6-DIRECT-INSTINCT-GPU-R3-') else None)
    results = {}
    for label, checkpoint in params['models'].items():
        payload = torch.load(checked(checkpoint), map_location='cpu', weights_only=True)
        network = NetworkConfig(**payload['network_config'])
        if network != NetworkConfig():
            raise ValueError('DYNAMIC_ASSAY_ACTOR_BOUNDARY')
        actor = checkpoint_actor(payload).to(device)
        actor.load_state_dict(payload['actor'], strict=True)
        actor.eval()
        mount_record = None
        if option:
            from instinctlab.control.escape_option import METHOD_FIELD, METHOD_VERSION
            from instinctlab.control.escape_recovery import MIXED_VERSION, RECOVERY_VERSION
            version = METHOD_VERSION
            enabled = label in ('candidate', 'control')
            if mechanism:
                enabled = True
                version = dict(candidate=RECOVERY_VERSION, control=MIXED_VERSION, reference=METHOD_VERSION)[label]
            if rear_pair:
                enabled = True
            if run_id == RECOVERY_HOLDOUT:
                version = RECOVERY_VERSION
            if run_id in RECOVERY_SHORT_CHECKS:
                enabled, version = True, RECOVERY_VERSION
            if three:
                version = params['arm_options'][label]
                enabled = version is not None
            trained = label == 'candidate' or (three == 'CHECK' and label == 'control')
            if (three in ('CHECK', 'HOLDOUT') or run_id in (HOLDOUT_RUN, RECOVERY_HOLDOUT, *RECOVERY_SHORT_CHECKS)) and trained:
                if payload['method_hashes'].get(METHOD_FIELD) != version:
                    raise ValueError('DYNAMIC_HOLDOUT_METHOD_BINDING')
            env.set_escape_option(enabled, version=version)
            if mount_assay:
                mount_record = record_mount(env, backend.output, label)
                rays = env._ray_caster.ray_directions[0]
                cue._full_rays.copy_(rays)
                cue._rays.copy_(rays.reshape(59,360,3)[::3].reshape(-1,3))
        cohort = []
        for index, group in enumerate(groups):
            visibility = min(group) >= 8
            modes = [mode for mode in group for _ in range(8)]
            seed = params['seed']+index
            block = dict(id=f'dynamic-{index}', task='stop', speed_m_s=.25,
                         pulse_acceleration_m_s2=0., sensor_to_wall_distance_m=None, steps=STEPS)
            env.set_h1_evaluation_case(None)
            env.set_h1_ics_clearance_margin(.15)
            env.configure_r6_physics(block=block,seed=seed,split=split,
                                     reset_sink=lambda row: None, force_sink=lambda row: None,
                                     expected_mass_kg=1.181517923)
            specs = env._r6_physics['specs']
            for lane in range(LANES):
                if visibility:
                    orientation = 8 if rear_pair and label == 'candidate' else modes[lane]
                    specs[lane] = visibility_reset(block, seed=seed, lane=lane, mode=orientation,
                                                   mount=env.cfg.ray_caster.offset.pos, split=split)
                elif modes[lane] == 5:
                    wall = dict(block, task='wall_modes', sensor_to_wall_distance_m=1.40)
                    specs[lane] = reset_spec(wall, seed=seed, split=split if three else 'development', lane=lane,
                                            mount_position_m=env.cfg.ray_caster.offset.pos)
            if mirror_pipe:
                for lane in range(25, 32, 2):
                    specs[lane] = pipe_authority_probe.mirror_reset(specs[lane-1], specs[lane])
            if pipe_probe:
                approved = approved_frames.open(pipe_authority_probe.frame_name(index))
            axis_angle = torch.arange(LANES,device=device).remainder(8)*math.pi/4
            axis = torch.stack((axis_angle.cos(),axis_angle.sin(),torch.zeros_like(axis_angle)),dim=1)
            mode_tensor = torch.tensor(modes,device=device)
            axis[mode_tensor == 5] = axis.new_tensor([-1.,0.,0.])
            if visibility:
                axis[:] = axis.new_tensor([-1., 0., 0.])
            elif holdout:
                axis[:] = axis.new_tensor([1., 0., 0.])
            command = axis * .25 * ((mode_tensor < 4) | (mode_tensor == 6))[:, None]
            local = torch.tensor([s['sensor_position_local_m'] for s in specs],device=device)
            physical_modes = torch.full_like(mode_tensor, 5) if visibility else mode_tensor
            obstacle.configure(env.scene.env_origins+local, axis, physical_modes)
            if three == 'REAR' and label.endswith('_slow'):
                # Same path and terminal position, with more time after a rear return.
                obstacle.velocity.mul_(2/3)
                obstacle.end_travel_s.mul_(3/2)
            if holdout:
                # Keep a crossing inside the side gaps of the authored fixtures.
                crossing = (mode_tensor == 2) | (mode_tensor == 3)
                obstacle.end_travel_s[crossing] = 7.5
            env.set_h1_initial_command(torch.zeros_like(command),0)
            env.stage_r6_world_command(command)
            backend.resource_check('before_r6_dynamic_reset', counters.explicit_resets+1)
            observation, _ = backend._preserving_reset(seed=seed)
            hold = None
            hook = None
            if persistence and label == 'candidate':
                hold = EscapeHoldProbe(observation['policy']['state_vec'])
                hook = actor.hover_reflex.register_forward_hook(hold)
            initial = env._robot.data.root_state_w[:, :3].clone().cpu().tolist()
            fixed = dict(axis_w=axis.cpu().tolist(), modes=modes, initial_position_w=initial,
                         command_w=command.cpu().tolist())
            if holdout:
                fixed['layout'] = [LAYOUTS[i // 8] for i in range(LANES)]
            if rear_pair:
                fixed['sensor_aspect'] = 'front' if label == 'candidate' else 'rear'
            rows = []
            for step in range(STEPS):
                backend.resource_check('before_r6_dynamic_step', step)
                counters.require_step_capacity(kind='nontraining',lanes=LANES,maximum=budgets['nontraining_transitions'])
                counters.require_reset_capacity(automatic=True,lanes=LANES,maximum=budgets['automatic_resets'])
                attitude = env._robot.data.root_state_w[:,3:7]
                body = rotate_world_to_body_wxyz(attitude,command)
                policy = observation['policy']
                if pipe_probe:
                    pipe_authority_probe.capture(approved, policy)
                if not torch.allclose(policy['state_vec'][:,45:48],body,atol=1e-6,rtol=0):
                    raise RuntimeError('DYNAMIC_ASSAY_COMMAND_HISTORY')
                before = obstacle.latest
                witness = policy['lidar_grid'][:,9].flatten(1)[obstacle.indices, before['witness_beam']]
                observed = before['winning_beams'] > 0
                if three:
                    latest_valid = policy['lidar_grid'][:, 10].flatten(1)[obstacle.indices, before['witness_beam']]
                    delivered = (observed & (latest_valid > .5)).cpu().tolist()
                if observed.any() and not torch.allclose(witness[observed],before['witness_range_m'][observed],atol=.002,rtol=0):
                    raise RuntimeError('DYNAMIC_ASSAY_ACTOR_WITNESS')
                approach = plausible_approach(policy, period_s=.1).cpu().tolist() if visibility else None
                direction = (cue.escape_direction(policy['lidar_grid'], policy['state_vec']).cpu().tolist()
                             if cue is not None else None)
                env.prepare_h1_step(global_policy_step=step, command_b=env._command_b, next_command_b=body)
                with torch.no_grad():
                    action = actor(policy).mean_action
                counters.attempt_step()
                observation, _, terminated, truncated, extras = env.step(action)
                counters.complete_step(LANES, kind='nontraining')
                terminal = (terminated | truncated).reshape(-1)
                counters.reserve_reset(automatic=True,lanes=int(terminal.sum()),maximum=budgets['automatic_resets'])
                record = extras['pre_reset']
                state = record['root_state_w']
                rows.append(dict(fixed, step=step, pre=measurements(before), post=measurements(obstacle.pre_reset),
                                 actor_witness_range_m=witness.cpu().tolist(),
                                 vehicle_position_w=state[:, :3].cpu().tolist(),
                                 velocity_w=state[:, 7:10].cpu().tolist(),
                                 terminal=terminal.cpu().tolist(),
                                 collision=record['task']['termination']['masks']['collision'].reshape(-1).cpu().tolist(),
                                 v_final_b=record['control']['v_final_b'].cpu().tolist(),
                                 v_gov_b=record['control']['v_gov_b'].cpu().tolist()))
                if visibility:
                    rows[-1]['sensor_approach'] = approach
                if direction is not None:
                    rows[-1]['reflex_direction_b'] = direction
                if persistence or option:
                    history = policy['state_vec'].reshape(LANES, 4, 13)
                    applied = None if hold is None else hold.bias
                    if option and env.escape_option is not None:
                        current = env.escape_option
                        applied = current.direction * current.applied[:, None]
                        if recovery_assay:
                            saved_option = record['escape_option']
                            applied = saved_option['direction_b'] * saved_option['active'][:, None]
                    rows[-1].update(
                        gyro_peak=history[:, :, :3].norm(dim=2).amax(dim=1).cpu().tolist(),
                        range_age_s=history[:, -1, -1].cpu().tolist(),
                        held_bias_b=None if applied is None else applied.cpu().tolist(),
                        ics_beta=record['control']['ics_metrics']['beta'].reshape(-1).cpu().tolist(),
                        v_final_w=record['control']['v_final_w'].cpu().tolist(),
                    )
                    if hold is not None:
                        hold.reset(terminal)
                if recovery_assay:
                    saved_option = record.get('escape_option', {})
                    rows[-1].update(
                        escape_recovery=measurements({
                            'version': env.escape_option_version,
                            'active': saved_option.get('active', torch.zeros_like(terminal)),
                            'recovery': saved_option.get('recovery', torch.zeros_like(terminal)),
                            'redirected': saved_option.get('redirected', torch.zeros_like(terminal)),
                            'action': action,
                            'selected_action': saved_option.get('applied_action', action),
                        }),
                        ics_emergency=record['control']['ics_metrics']['emergency'].reshape(-1).cpu().tolist(),
                        ics_minimum_clearance_m=record['control']['ics_metrics']['minimum_clearance_m'].reshape(-1).cpu().tolist(),
                        ics_active_beam_count=record['control']['ics_metrics']['active_beam_count'].reshape(-1).cpu().tolist(),
                    )
                if three:
                    rows[-1]['mover_witness_delivered'] = delivered
                    rows[-1]['sensor_split'] = split
                if mirror_pipe:
                    rows[-1]['pipe_retreat_basis_b'] = record['escape_option']['retreat_basis'][24:].cpu().tolist()
            if hook is not None:
                hook.remove()
            env.finish_r6_physics()
            obstacle.finish()
            rel = f'{label}-group-{index}.jsonl'
            trace_file = Path(backend.output) / rel
            with trace_file.open('xb') as stream:
                for row in rows:
                    data = (json.dumps(row,allow_nan=False,separators=(',',':'))+'\n').encode()
                    reserve_output_bytes(trace_file,len(data))
                    stream.write(data)
            cohort.append(dict(trace=file_pin(trace_file), lanes=score(rows), transitions=STEPS*LANES))
            if pipe_probe:
                approved.close()
                cohort[-1]['approved_frames'] = file_pin(approved.path)
        results[label] = dict(checkpoint=checkpoint, groups=cohort)
        if mount_record is not None:
            results[label]['lidar_mount'] = mount_record
    if pipe_probe:
        approved_frames.close()
    summary = dict(models=results, counters=counters.snapshot())
    if pipe_probe:
        from dataclasses import asdict

        config = env._control_pipelines[24].config
        summary['pipe_control_config'] = dict(governor=asdict(config.governor), ics=asdict(config.ics))
    atomic_json_replace(backend.output,'dynamic_results.json', summary)
    return results


def audit(output, params, counters):
    """Independently recompute lane outcomes and ray witnesses from saved inputs."""
    from instinctlab.control.escape_recovery import MOUNT_PITCH_DEGREES
    from .lidar_calibration import verify as verify_mount

    path = Path(output) / 'dynamic_results.json'
    saved = json.loads(path.read_bytes())
    if saved['counters'] != counters or set(saved['models']) != set(params['models']):
        raise ValueError('DYNAMIC_ASSAY_SUMMARY_BINDING')
    sources = [file_pin(path)]
    groups = groups_for(params['experiment_id'])
    persistence = params['experiment_id'] == PERSISTENCE_RUN
    mount_assay = any(version in MOUNT_PITCH_DEGREES for version in params.get('arm_options', {}).values())
    three = three_case(params['experiment_id'])
    holdout = physical_layout(params['experiment_id'])
    if holdout:
        from .escape_holdout_scene import LAYOUTS
        preflight = Path(output) / 'initial_preflight.json'
        observed = json.loads(preflight.read_bytes())
        if observed['scene'] != 'room-door-trunks-street-tube' or observed['geometry']['status'] != 'PASS':
            raise ValueError('DYNAMIC_HOLDOUT_GEOMETRY_NOT_VERIFIED')
        sources.append(file_pin(preflight))
    matched_starts = {}
    for name, result in saved['models'].items():
        if result['checkpoint'] != params['models'][name] or len(result['groups']) != len(groups):
            raise ValueError('DYNAMIC_ASSAY_MODEL_BINDING')
        if mount_assay:
            pitch = MOUNT_PITCH_DEGREES.get(params['arm_options'][name], 30)
            sources.append(verify_mount(result['lidar_mount'], output, name, pitch))
        for index, group in enumerate(result['groups']):
            trace = checked(group['trace'])
            if trace != Path(output) / f'{name}-group-{index}.jsonl':
                raise ValueError('DYNAMIC_ASSAY_TRACE_PATH')
            sources.append(file_pin(trace))
            if three == 'PIPE-AUTHORITY':
                approved = checked(group['approved_frames'])
                if approved != Path(output)/'approved_measurements'/f'{pipe_authority_probe.frame_name(index)}.r6z':
                    raise ValueError('DYNAMIC_PIPE_AUTHORITY_FRAME_PATH')
                if sum(1 for _ in pipe_authority_probe.replay_frames(group['approved_frames'], index)) != STEPS:
                    raise ValueError('DYNAMIC_PIPE_AUTHORITY_FRAME_COUNT')
                sources.append(group['approved_frames'])
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            if [r['step'] for r in rows] != list(range(STEPS)) or score(rows) != group['lanes']:
                raise ValueError('DYNAMIC_ASSAY_SCORER_REPLAY')
            expected_modes = [mode for mode in groups[index] for _ in range(8)]
            if holdout and params['experiment_id'] != RECOVERY_REAR:
                first = rows[0]
                signature = (first['initial_position_w'], first['pre']['sphere_position_w'],
                             first['axis_w'], first['modes'], first['command_w'])
                previous = matched_starts.setdefault(index, signature)
                if previous[2:] != signature[2:] or any(math.dist(a, b) > .0002
                        for aa, bb in zip(previous[:2], signature[:2]) for a, b in zip(aa, bb)):
                    raise ValueError('DYNAMIC_HOLDOUT_UNMATCHED_RESET')
            visibility = min(groups[index]) >= 8
            if visibility:
                start = rows[0]['pre']
                for center, sensor, command in zip(start['sphere_position_w'], start['sensor_position_w'], rows[0]['command_w']):
                    if any(command) or max(abs(c - s - d) for c, s, d in zip(center, sensor, (-2.75, 0., 0.))) > .003:
                        raise ValueError('DYNAMIC_ASSAY_VISIBILITY_PAIRED_START')
            for row in rows:
                before, after = row['pre'], row['post']
                if (after['tick'] - before['tick'] != 20 or after['contact_samples'] != 20
                        or row['modes'] != expected_modes):
                    raise ValueError('DYNAMIC_ASSAY_ACTUAL_PHYSICS_TICKS')
                if holdout and row['layout'] != [LAYOUTS[i // 8] for i in range(LANES)]:
                    raise ValueError('DYNAMIC_HOLDOUT_LAYOUT_ORDER')
                if three and (row['escape_recovery']['version'] != params['arm_options'][name]
                              or row['sensor_split'] != ('heldout' if three == 'HOLDOUT' else 'development')
                              or row['mover_witness_delivered'] != [bool(n) for n in before['winning_beams']]):
                    raise ValueError('DYNAMIC_THREE_EXECUTION_EVIDENCE')
                if three and any(redirected and (emergency or beta == 0.)
                                 for redirected, emergency, beta in zip(
                    row['escape_recovery'].get('redirected', [False] * LANES),
                    row['ics_emergency'], row['ics_beta'], strict=True)):
                    raise ValueError('DYNAMIC_REDIRECTION_WAS_NOT_ISSUED')
                if three in ('REAR', 'REARPAIR'):
                    expected = .28 if name.endswith('_slow') else .42
                    if any(abs(v[0] - expected) > 1.e-6 for v in before['expected_velocity_w']
                           if any(v)):
                        raise ValueError('DYNAMIC_REAR_FACTORIAL_SPEED')
                if visibility and (
                        len(row['sensor_approach']) != LANES or any(type(v) is not bool for v in row['sensor_approach'])):
                    raise ValueError('DYNAMIC_ASSAY_VISIBILITY_CLOSURE_EVIDENCE')
                if persistence or params['experiment_id'].startswith('R6-DIRECT-INSTINCT-GPU-R3-'):
                    directions = row['reflex_direction_b']
                    if len(directions) != LANES or any(
                            len(v) != 3 or not all(math.isfinite(x) for x in v)
                            or abs(v[2]) > 1e-6 or math.hypot(*v[:2]) > 1.000001 for v in directions):
                        raise ValueError('DYNAMIC_ASSAY_HOVER_CUE_EVIDENCE')
                for j in range(LANES):
                    if max(abs(x-y) for x,y in zip(after['sphere_position_w'][j],after['expected_position_w'][j])) > .003:
                        raise ValueError('DYNAMIC_ASSAY_REPLAYED_POSE')
                    if math.sqrt(sum(x*x for x in after['servo_velocity_w'][j])) > .950001:
                        raise ValueError('DYNAMIC_ASSAY_SERVO_SPEED_BUDGET')
                    if before['winning_beams'][j]:
                        relative = [x-y for x,y in zip(before['sphere_position_w'][j],before['sensor_position_w'][j])]
                        distance = before['witness_range_m'][j]
                        ray = before['witness_direction_w'][j]
                        length = math.sqrt(sum((x-distance*d)**2 for x,d in zip(relative,ray)))
                        if abs(length-.34) > .003 or abs(distance-row['actor_witness_range_m'][j]) > .002:
                            raise ValueError('DYNAMIC_ASSAY_REPLAYED_SENSOR')
            if (visibility or min(groups[index]) < 6) and not any(any(r['pre']['winning_beams']) for r in rows):
                raise ValueError('DYNAMIC_ASSAY_NO_VISIBLE_MOVING_OBJECT')
    return saved, sources
