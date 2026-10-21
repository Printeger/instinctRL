"""R6 loaded-trainer phase execution; not a dispatcher or Store permission.

The caller must restore the exact checkpoint and apply the reviewed method
before this entry. The prospective training mode remains unavailable until its
phase authorization, source compatibility and live lifecycle are registered.
"""
from dataclasses import asdict
from pathlib import Path
import torch
from . import robustness_training_plan as contract
from .checkpoint_compatibility import pinned_json
from .robustness_curriculum import PhaseCurriculum,slices
from .robustness_training_rng import prepare_plan_rng
from .robustness_training_storage import create_training_evidence,R6LearningLog
from .robustness_training_driver import R6WorldCommandClock,R6TrainingDriver
from .driver import UnifiedH1Driver,UnifiedH1TrainingLoop
from .schedule import ScheduleClock,command_at,reset_local_positions


def run_loaded(backend,*,plan,metadata,source_identity,method_hashes,expected_mass_kg):
    p=contract.validate(plan);recipe=pinned_json(p['recipe']);env=backend.environment
    if (backend.authorization['mode']!='research-r6-training'
            or backend.authorization['parameters'].get('r6_training_plan')!=p
            or backend.authorization['budgets']!=contract.required_budget(p)):
        raise PermissionError('R6_EXECUTION_EXACT_PLAN_BUDGET')
    if source_identity!=recipe['source_identity'] or method_hashes.get('r6_recipe')!=p['recipe']['sha256']:
        raise PermissionError('R6_EXECUTION_SOURCE_METHOD')
    if backend.journal is not None or backend.driver is not None or backend.loop is not None:
        raise RuntimeError('R6_EXECUTION_ALREADY_INITIALIZED')
    if getattr(env,'_h1_pure_hover_training',True):raise ValueError('R6_EXECUTION_ORIGINAL_REPLAY_REQUIRED')
    actual=asdict(backend.trainer.config);method=pinned_json(recipe['method'])
    from .robustness_training_method import verify_applied
    verify_applied(backend.trainer,env,method)
    if any(actual.get(k)!=v for k,v in method['ppo'].items() if k!='permutation_seed'):
        raise ValueError('R6_EXECUTION_PPO_METHOD_NOT_APPLIED')
    if 'permutation_seed' in method['ppo'] and method['ppo']['permutation_seed']!=p['training_seed']:
        raise ValueError('R6_EXECUTION_METHOD_SEED')
    backend.trainer.actor.train();backend.trainer.critic.train()
    storage=p['storage']
    journal,diagnostic=create_training_evidence(backend.output,run_id=p['experiment_id'],
        starting_update=p['start_update'],**{k:storage[k] for k in
        ('payload_bytes','compressed_bytes','plain_bytes','maximum_raw_file_bytes')})
    backend.journal=journal;pool=journal.pool
    record=prepare_plan_rng(backend.trainer,backend.curriculum_generator,plan=p,metadata=metadata)
    # Save and close the RNG record before any reset/rollout. Pool caps and the
    # global payload write boundary apply, including partial write failures.
    stream=pool.open('rng/initialization.pt')
    try:torch.save(record,stream);stream.close()
    except BaseException:stream.abort();raise
    journal.append('plan.jsonl',dict(plan=p,source_identity=source_identity,method_hashes=method_hashes,
        physical_continuity=False,metadata=metadata,method_binding=backend._r6_method_binding))
    curriculum=pinned_json(recipe['curriculum'])
    phases=slices(curriculum,p['start_update'],p['target_update'])
    backend._checkpoint_reset_offset=metadata['reset_number']

    def checkpoint(label,trainer,driver):
        from instinctlab.training.checkpoint_h1 import save_checkpoint_h1
        from training_workloads.scale_003.persistence import guard_stream,atomic_json_replace
        from .config import sha256_file
        update=trainer.update_counter
        if label!=f'u{update}' or update not in p['checkpoint_updates'] or driver.clock.completed_policy_steps!=update*16:
            raise ValueError('R6_CHECKPOINT_UNREGISTERED_NODE')
        backend.resource_check('before_r6_checkpoint',backend.counters.real_checkpoint_writes+1)
        backend.counters.reserve_checkpoint(len(p['checkpoint_updates']))
        target=Path(backend.output)/'checkpoints'/f'{label}.pt'
        if target.exists():raise ValueError('R6_CHECKPOINT_NO_OVERWRITE')
        target.parent.mkdir(parents=True,exist_ok=True)
        clock=dict(version='curriculum-h1-r6-v1',completed_policy_steps=update*16,
            recipe_sha256=p['recipe']['sha256'],origin_checkpoint_sha256=contract.BASELINE_SHA,branch_seed=p['training_seed'])
        save_checkpoint_h1(target,trainer,method_hashes=method_hashes,source_identity=source_identity,
            transition_counter=update*512,global_policy_step=update*16,curriculum=clock,
            training_seed=p['training_seed'],reset_number=metadata['reset_number']+backend.counters.explicit_resets+backend.counters.automatic_resets,
            curriculum_generator=backend.curriculum_generator,write_guard=guard_stream)
        backend.counters.real_checkpoint_writes+=1
        atomic_json_replace(backend.output,f'checkpoints/{label}.json',dict(path=str(target),sha256=sha256_file(target),bytes=target.stat().st_size))

    for ordinal,phase in enumerate(phases):
        start=phase['start_update']*16;end=phase['target_update']*16
        if backend.trainer.update_counter*16!=start:raise RuntimeError('R6_PHASE_TRAINER_CLOCK')
        if ordinal:
            journal.finalize_right_censored();journal.close()
        else:journal.close() # Identity journal has no rollout/episode state.
        prefix=f"phases/{phase['index']}-{start}-{end}"
        sampled=None
        def audit(event,value):
            journal.append('curriculum.jsonl',dict(event=event,payload=value))
        # Sampling audit is written through the identity journal's shared pool
        # in a separate phase stream before creating lane-specific episode logs.
        bootstrap=R6LearningLog(Path(backend.output)/prefix/'setup',pool=pool,logical_prefix=prefix+'/setup')
        backend.journal=journal=bootstrap
        env.set_h1_evaluation_case(None)
        env.set_h1_initial_command(command_at(start,device=env.device),start)
        if phase['kind']=='robust':
            sampled=PhaseCurriculum(curriculum,branch_updates=recipe['branch_updates'],phase_index=phase['index'],
                generator=backend.curriculum_generator,audit_sink=audit)
            types=tuple(('near_x_wall' if lane<16 else 'near_y_wall')
                if sampled.value['conditions'][name]['task']=='wall_modes' else 'center'
                for lane,name in enumerate(sampled.assignments))
        else:types=reset_local_positions(start)[1]
        bootstrap.close()
        journal=R6LearningLog(Path(backend.output)/prefix/'learning_log',pool=pool,logical_prefix=prefix+'/learning_log',
            initial_start_types=types,fixed_reset_types=types if sampled is not None else None)
        backend.journal=journal
        if sampled is not None:
            sampled.sink=audit
            env.configure_r6_training_phase(curriculum=sampled,start_policy_step=start,end_policy_step=end,
                reset_sink=lambda row:journal.append('reset.jsonl',row),force_sink=lambda row:journal.append('force.jsonl',row),
                measurement_sink=lambda seq,row:journal.append('measurement.jsonl',dict(sequence=seq,payload=row)),
                expected_mass_kg=expected_mass_kg)
        elif getattr(env,'_r6_physics',None) is not None or getattr(env,'_r6_measurements',None) is not None:
            raise RuntimeError('R6_ORIGINAL_PHASE_CONTAMINATED')
        backend.resource_check('before_r6_phase_reset',backend.counters.explicit_resets+1)
        observation,_=backend._preserving_reset(seed=p['training_seed']+metadata['reset_number']+backend.counters.explicit_resets+1)
        backend.observation=observation
        if sampled is None:clock=ScheduleClock(start);driver_type=UnifiedH1Driver
        else:
            clock=R6WorldCommandClock(completed_policy_steps=start,final_policy_step=end,
                recipe_sha256=p['recipe']['sha256'],branch_seed=p['training_seed'],
                command_source=lambda step:sampled.world_command(step,device=env.device,dtype=env._actions.dtype),
                attitude_source=lambda:env._robot.data.root_state_w[:,3:7])
            driver_type=R6TrainingDriver
        backend.driver=driver_type(env,observation,clock=clock,counters=backend.counters,journal=journal,
            before_step=lambda n:backend.resource_check('before_r6_training_step',n),
            transition_limit=backend.authorization['budgets']['training_transitions'],
            automatic_reset_limit=backend.authorization['budgets']['automatic_resets'])
        backend.loop=UnifiedH1TrainingLoop(backend.driver,backend.trainer,journal,checkpoint,
            before_rollout=lambda n:backend.resource_check('before_r6_rollout',n),
            before_optimizer=lambda n:backend.resource_check('before_r6_optimizer',n),
            update_limit=backend.authorization['budgets']['optimizer_updates'],diagnostic=diagnostic,
            optimizer_step_origin=(p['start_update']*8,p['start_update']*8))
        while backend.trainer.update_counter<phase['target_update']:
            n=backend.trainer.update_counter+1
            backend.loop.run_update(checkpoint_label=f'u{n}' if n in p['checkpoint_updates'] else None)
        if sampled is not None:env.finish_r6_training_phase()
        journal.append('phase_complete.jsonl',dict(phase=phase,counters=backend.counters.snapshot()))
    journal.finalize_evidence()
