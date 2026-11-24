"""Independent Isaac closed-loop dynamic assay, with exact bounded denominators."""
import json
import math
from pathlib import Path
import re
from statistics import mean

from tools.r6_evidence import checked, file_pin


MODE = 'research-r6-dynamic-assay'
STEPS = 120
LANES = 32
GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7))
VISIBILITY_GROUPS = ((8, 9, 10, 11),)
PERSISTENCE_RUN = 'R6-DIRECT-INSTINCT-GPU-R4-DEV-PERSISTENCE'
PERSISTENCE_GROUPS = (*VISIBILITY_GROUPS, (4, 1, 6, 7))
MODES = ('approach', 'recede', 'cross_left', 'cross_right',
         'hover', 'hover_wall', 'static_travel', 'static_hover',
         'open_front', 'open_rear', 'wall_front', 'wall_rear')


def groups_for(run_id):
    if run_id == PERSISTENCE_RUN:
        return PERSISTENCE_GROUPS
    return VISIBILITY_GROUPS if run_id.endswith('-VISIBILITY') else GROUPS


def contract(params):
    fields = {'experiment_id', 'training_seed', 'models', 'seed', 'wall_seconds', 'output_cap_bytes'}
    if set(params) != fields or (params['experiment_id'] != PERSISTENCE_RUN and not re.fullmatch(
            r'R6-DIRECT-INSTINCT-GPU-R[23]-(?:DYNAMIC(?:-PILOT-0[1-4])?|VISIBILITY)', params['experiment_id'])):
        raise ValueError('DYNAMIC_ASSAY_CONTRACT_IDENTITY')
    if params['training_seed'] != 20260908 or not 0 <= params['seed'] < 2**31:
        raise ValueError('DYNAMIC_ASSAY_SEED')
    if type(params['wall_seconds']) is not int or not 120 <= params['wall_seconds'] <= 1800:
        raise ValueError('DYNAMIC_ASSAY_WALL')
    if type(params['output_cap_bytes']) is not int or not 512*2**20 <= params['output_cap_bytes'] <= 2**30:
        raise ValueError('DYNAMIC_ASSAY_OUTPUT')
    models = params['models']
    if not isinstance(models, dict) or not 1 <= len(models) <= 2 or not set(models) <= {'candidate', 'reference'}:
        raise ValueError('DYNAMIC_ASSAY_MODELS')
    for value in models.values():
        checked(value)
    if params['experiment_id'] == PERSISTENCE_RUN and (
            set(models) != {'candidate', 'reference'} or models['candidate'] != models['reference']):
        raise ValueError('DYNAMIC_ASSAY_SAME_WEIGHT_MODELS')
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
        lanes.append(metrics)
    return lanes


def measurements(trace):
    return {k: v.detach().cpu().tolist() if hasattr(v, 'detach') else v for k,v in trace.items()}


def visibility_reset(block, *, seed, lane, mode, mount):
    from .robustness_scenarios import reset_spec, rotate_yaw

    spec = reset_spec(block, seed=seed, split='development', lane=lane % 8, mount_position_m=mount)
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
    from training_workloads.scale_003.persistence import atomic_json_replace, reserve_output_bytes
    from .dynamic_motion_training import plausible_approach
    from .escape_hold_probe import EscapeHoldProbe
    from .robustness_scenarios import reset_spec

    env = backend.environment
    obstacle = env._h1_moving_sphere
    budgets = backend.runtime_contract['budgets']
    counters = backend.counters
    device = env.device
    groups = groups_for(params['experiment_id'])
    persistence = params['experiment_id'] == PERSISTENCE_RUN
    cue = (HoverReflex(device=device, dtype=torch.float32)
           if persistence or params['experiment_id'].startswith('R6-DIRECT-INSTINCT-GPU-R3-') else None)
    results = {}
    for label, checkpoint in params['models'].items():
        payload = torch.load(checked(checkpoint), map_location='cpu', weights_only=True)
        network = NetworkConfig(**payload['network_config'])
        if network != NetworkConfig():
            raise ValueError('DYNAMIC_ASSAY_ACTOR_BOUNDARY')
        actor = checkpoint_actor(payload).to(device)
        actor.load_state_dict(payload['actor'], strict=True)
        actor.eval()
        cohort = []
        for index, group in enumerate(groups):
            visibility = group in VISIBILITY_GROUPS
            modes = [mode for mode in group for _ in range(8)]
            seed = params['seed']+index
            block = dict(id=f'dynamic-{index}', task='stop', speed_m_s=.25,
                         pulse_acceleration_m_s2=0., sensor_to_wall_distance_m=None, steps=STEPS)
            env.set_h1_evaluation_case(None)
            env.set_h1_ics_clearance_margin(.15)
            env.configure_r6_physics(block=block,seed=seed,split='development',
                                     reset_sink=lambda row: None, force_sink=lambda row: None,
                                     expected_mass_kg=1.181517923)
            specs = env._r6_physics['specs']
            for lane in range(LANES):
                if visibility:
                    specs[lane] = visibility_reset(block, seed=seed, lane=lane, mode=modes[lane],
                                                   mount=env.cfg.ray_caster.offset.pos)
                elif modes[lane] == 5:
                    wall = dict(block, task='wall_modes', sensor_to_wall_distance_m=1.40)
                    specs[lane] = reset_spec(wall, seed=seed, split='development', lane=lane,
                                            mount_position_m=env.cfg.ray_caster.offset.pos)
            axis_angle = torch.arange(LANES,device=device).remainder(8)*math.pi/4
            axis = torch.stack((axis_angle.cos(),axis_angle.sin(),torch.zeros_like(axis_angle)),dim=1)
            mode_tensor = torch.tensor(modes,device=device)
            axis[mode_tensor == 5] = axis.new_tensor([-1.,0.,0.])
            if visibility:
                axis[:] = axis.new_tensor([-1., 0., 0.])
            command = axis * .25 * ((mode_tensor < 4) | (mode_tensor == 6))[:, None]
            local = torch.tensor([s['sensor_position_local_m'] for s in specs],device=device)
            physical_modes = torch.full_like(mode_tensor, 5) if visibility else mode_tensor
            obstacle.configure(env.scene.env_origins+local, axis, physical_modes)
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
            rows = []
            for step in range(STEPS):
                backend.resource_check('before_r6_dynamic_step', step)
                counters.require_step_capacity(kind='nontraining',lanes=LANES,maximum=budgets['nontraining_transitions'])
                counters.require_reset_capacity(automatic=True,lanes=LANES,maximum=budgets['automatic_resets'])
                attitude = env._robot.data.root_state_w[:,3:7]
                body = rotate_world_to_body_wxyz(attitude,command)
                policy = observation['policy']
                if not torch.allclose(policy['state_vec'][:,45:48],body,atol=1e-6,rtol=0):
                    raise RuntimeError('DYNAMIC_ASSAY_COMMAND_HISTORY')
                before = obstacle.latest
                witness = policy['lidar_grid'][:,9].flatten(1)[obstacle.indices, before['witness_beam']]
                observed = before['winning_beams'] > 0
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
                if persistence:
                    history = policy['state_vec'].reshape(LANES, 4, 13)
                    rows[-1].update(
                        gyro_peak=history[:, :, :3].norm(dim=2).amax(dim=1).cpu().tolist(),
                        range_age_s=history[:, -1, -1].cpu().tolist(),
                        held_bias_b=None if hold is None else hold.bias.cpu().tolist(),
                        ics_beta=record['control']['ics_metrics']['beta'].reshape(-1).cpu().tolist(),
                        v_final_w=record['control']['v_final_w'].cpu().tolist(),
                    )
                    if hold is not None:
                        hold.reset(terminal)
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
        results[label] = dict(checkpoint=checkpoint, groups=cohort)
    atomic_json_replace(backend.output,'dynamic_results.json', dict(models=results, counters=counters.snapshot()))
    return results


def audit(output, params, counters):
    """Independently recompute lane outcomes and ray witnesses from saved inputs."""
    path = Path(output) / 'dynamic_results.json'
    saved = json.loads(path.read_bytes())
    if saved['counters'] != counters or set(saved['models']) != set(params['models']):
        raise ValueError('DYNAMIC_ASSAY_SUMMARY_BINDING')
    sources = [file_pin(path)]
    groups = groups_for(params['experiment_id'])
    persistence = params['experiment_id'] == PERSISTENCE_RUN
    for name, result in saved['models'].items():
        if result['checkpoint'] != params['models'][name] or len(result['groups']) != len(groups):
            raise ValueError('DYNAMIC_ASSAY_MODEL_BINDING')
        for index, group in enumerate(result['groups']):
            trace = checked(group['trace'])
            if trace != Path(output) / f'{name}-group-{index}.jsonl':
                raise ValueError('DYNAMIC_ASSAY_TRACE_PATH')
            sources.append(file_pin(trace))
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            if [r['step'] for r in rows] != list(range(STEPS)) or score(rows) != group['lanes']:
                raise ValueError('DYNAMIC_ASSAY_SCORER_REPLAY')
            expected_modes = [mode for mode in groups[index] for _ in range(8)]
            visibility = groups[index] in VISIBILITY_GROUPS
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
            if not any(any(r['pre']['winning_beams']) for r in rows):
                raise ValueError('DYNAMIC_ASSAY_NO_VISIBLE_MOVING_OBJECT')
    return saved, sources
