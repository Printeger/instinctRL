"""R6 evaluation blocks on the existing authorized H1 backend.

No AppLauncher, checkpoint substitution, permission issuance or old-suite scoring
changes here. Caller loads the exactly authorized model and freezes the protocol.
"""
from pathlib import Path
import torch
from instinctlab.control.lee_velocity_controller import rotate_world_to_body_wxyz
from training_workloads.scale_003.persistence import atomic_json_replace
from .robustness import command,score_block
from .range_archive import RangeArchive
from .robustness_audit import AuditLog,write_block_metadata


def world_command(block,step,reference):
    return torch.tensor([command(block,i,step) for i in range(32)],device=reference.device,dtype=reference.dtype)


def lane_sample(sensor,root,masks,lane,episode,*,time_s):
    return dict(sensor_position_w_m=sensor[lane].detach().cpu().tolist(),
        velocity_w_m_s=root[lane,7:10].detach().cpu().tolist(),episode_id=episode,time_s=time_s,
        collision=bool(masks['collision'][lane,0]),
        height_violation=bool(masks['below_height'][lane,0] or masks['above_height'][lane,0]),missing=False)


def evaluate_block(backend,block,*,run_id,seed,split,expected_mass_kg,range_quota_bytes,audit_quota_bytes):
    if not isinstance(run_id,str) or not run_id or type(seed)is not int or split not in ('development','final'):
        raise ValueError('explicit R6 run/random-stream identity required')
    if block['lanes']!=32 or block['steps']!=(200 if block['task']=='reverse' else 330):
        raise ValueError('R6 evaluation block denominator')
    if not isinstance(block['id'],str) or not block['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in block['id']):
        raise ValueError('R6 block identifier')
    if backend.trainer is None:raise ValueError('load authorized checkpoint before R6 evaluation')
    env=backend.environment;budgets=backend.authorization['budgets'];counters=backend.counters
    folder=Path(backend.output)/'r6'/block['id'];folder.mkdir(parents=True,exist_ok=False)
    counts={'force':0,'reset':0}
    def record(kind,row):
        audit.append(kind,row);counts[kind]+=1
    with AuditLog(folder/'audit.bin',run_id=run_id,block_id=block['id'],maximum_bytes=audit_quota_bytes) as audit, \
         RangeArchive(folder/'measurements.bin',run_id=run_id,block_id=block['id'],maximum_bytes=range_quota_bytes) as archive:
        audit.append('identity',dict(block=block,seed=seed,split=split,expected_mass_kg=expected_mass_kg,
            actor_mode='deterministic_mean',command_frame='world converted at actual observation boundary',
            sensor_frames=block['steps']+1,physics_ticks=block['steps']*20))
        def measurement(sequence,frame):
            if sequence!=archive.frames:raise ValueError('R6 archive sequence drift')
            archive.append(frame)
        env.set_h1_evaluation_case(None)
        env.configure_r6_physics(block=block,seed=seed,split=split,reset_sink=lambda row:record('reset',row),
            force_sink=lambda row:record('force',row),expected_mass_kg=expected_mass_kg)
        env.configure_r6_measurements(seed=seed,stream_id=split+'/'+block['id'],sigma_m=block['noise_std_m'],
            delay_cycles=block['delay_cycles'],drop_steps=block.get('drop_policy_steps',[]),
            frame_limit=block['steps']+1,audit_sink=measurement)
        reference=env._command_b
        env.set_h1_initial_command(torch.zeros_like(reference),0)
        env.stage_r6_world_command(world_command(block,0,reference))
        backend.resource_check('before_r6_reset',counters.explicit_resets+1)
        observation,_=backend._preserving_reset(seed=seed)
        env._refresh_task_snapshot()
        sensor=env._evaluation_ray_snapshot.ray_starts_w[:,0].detach().clone()
        root=env._robot.data.root_state_w.detach().clone()
        masks=env._task.termination.masks
        traces=[dict(schema_version='r6-motion-trace/1',run_id=run_id,block_id=block['id'],lane=i,
            env_origin_w_m=env.scene.env_origins[i].detach().cpu().tolist(),
            initial=lane_sample(sensor,root,masks,i,0,time_s=0.),rows=[]) for i in range(32)]
        audit.append('initial',dict(root_state_w=root,sensor_position_w_m=sensor,
            env_origins_w_m=env.scene.env_origins,lanes=[t['initial'] for t in traces]))
        episodes=[0]*32
        backend.trainer.actor.eval()
        for step in range(block['steps']):
            backend.resource_check('before_r6_policy_step',step)
            counters.require_step_capacity(kind='nontraining',lanes=32,maximum=budgets['nontraining_transitions'])
            counters.require_reset_capacity(automatic=True,lanes=32,maximum=budgets['automatic_resets'])
            current=world_command(block,step,reference);following=world_command(block,step+1,reference)
            attitude=env._robot.data.root_state_w[:,3:7]
            body=rotate_world_to_body_wxyz(attitude,current)
            if not torch.allclose(observation['policy']['state_vec'][:,45:48],body,atol=1e-6,rtol=0) or not torch.equal(observation['policy']['state_vec'][:,45:48],env._command_b):
                raise ValueError('R6 actor/internal/world command mismatch')
            env.stage_r6_world_command(following)
            env.prepare_h1_step(global_policy_step=step,command_b=env._command_b,
                next_command_b=rotate_world_to_body_wxyz(attitude,following))
            with torch.no_grad():action=backend.trainer.actor(observation['policy']).mean_action.detach()
            counters.attempt_step()
            audit.append('step_attempt',dict(policy_step=step,counters=counters.snapshot(),
                command_w_m_s=current,command_b_m_s=body,attitude_wxyz=attitude,action=action,
                input_measurement_sequence=step))
            observation,_,terminated,truncated,extras=env.step(action)
            counters.complete_step(32,kind='nontraining')
            reset=(terminated|truncated).detach().cpu().reshape(-1)
            if reset.shape!=(32,):raise ValueError('R6 reset denominator')
            counters.reserve_reset(automatic=True,lanes=int(reset.sum()),maximum=budgets['automatic_resets'])
            pre=extras['pre_reset'];source=extras['evaluation_ray_source'];root=pre['root_state_w']
            sensor=source['ray_starts_w'][:,0];masks=pre['task']['termination']['masks']
            samples=[]
            for lane in range(32):
                # A reset on the last sample also fails; no last-step omission.
                episodes[lane]+=int(reset[lane])
                row=lane_sample(sensor,root,masks,lane,episodes[lane],time_s=(step+1)*.1)
                row.update(policy_step=step,command_w_m_s=current[lane].detach().cpu().tolist())
                traces[lane]['rows'].append(row);samples.append(row)
            audit.append('step_complete',dict(policy_step=step,counters=counters.snapshot(),lanes=samples,
                root_state_w=root,reset_mask=reset,control=pre['control'],reward_diagnostics=extras['reward_diagnostics'],
                output_measurement_sequence=step+1,timing='action_t/command_t before physics; root, source and reward diagnostics are pre-reset t+1; delivery archive t+1 may reflect reset'))
        physics_completion=env.finish_r6_physics();env.finish_r6_measurements()
        if archive.frames!=block['steps']+1 or counts['force']!=block['steps']*20 or counts['reset']<32:
            raise ValueError('R6 missing measurement/reset/force evidence')
        result=score_block(block,traces,run_id=run_id)
        summary=dict(schema_version='r6-block-result/1',run_id=run_id,block_id=block['id'],seed=seed,split=split,
            scoring=result,counters=counters.snapshot(),measurement_frames=archive.frames,audit_records=audit.records,
            force_records=counts['force'],reset_records=counts['reset'],measurement_bytes=archive.written,audit_bytes=audit.written,
            physics_completion=physics_completion,
            evidence_status='COMPLETE_BLOCK_REQUIRES_INDEPENDENT_REPLAY',scientific_acceptance=False)
    # Preserve the complete replayable lane records separately for the pure scorer.
    write_block_metadata(folder,traces,summary)
    return summary
