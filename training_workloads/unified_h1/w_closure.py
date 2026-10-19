"""Explicit new W phase; no legacy method/authority is implicitly extended."""
from pathlib import Path
import hashlib
import json
import re

CONTRACT_PATH = Path(__file__).resolve().parents[2]/'coordination/contracts/R5_W_SAFETY_CLOSURE_V1.md'
CONTRACT_SHA256 = '686ffa93eb48034af85cc0e31c026bcdba5c26014ae615238952b12e9c9744b5'
MASTER_ID = 'R5-W-CLOSURE-AUTO-20261019'
TRAIN = 'research-formal-training'
EVAL = 'research-formal-evaluation'
FIELDS = {'experiment_id','group','training_seed','evaluation_seed','start_update','target_update',
          'checkpoint_updates','evaluation_updates','command_profile','residual_neutrality_weight',
          'anchor_weight','actor_lr','critic_lr','mean_prior_weight','mean_prior_contract',
          'output_cap_bytes','master_authority','experiment_card','w_closure_contract',
          'checkpoint_source_identity','checkpoint_compatibility','initialization_checkpoint',
          'origin_method_hashes','evaluation_checkpoint_origin','ics_clearance_margin_m'}


def validate_parameters(mode, value):
    from .research_contract import validate_pin, MEAN_PRIOR_CONTRACT_SHA256
    from .checkpoint_compatibility import validate_source_identity, pinned_json
    if mode not in (TRAIN,EVAL) or not isinstance(value,dict):
        raise PermissionError('unknown W phase')
    if set(value) != FIELDS | ({'checkpoints'} if mode == EVAL else set()):
        raise PermissionError('W phase exact fields mismatch')
    p=dict(value)
    if not isinstance(p['experiment_id'],str) or not re.fullmatch(r'R5-W-[A-Z0-9-]{1,90}',p['experiment_id']):
        raise PermissionError('W experiment identity invalid')
    if p['group'] != 'R1': raise PermissionError('W preferred baseline is R1')
    if type(p['training_seed']) is not int or p['training_seed'] != 20260908:
        raise PermissionError('W warm start training seed differs')
    # A separately implemented equivalent seed suite must be reviewed before
    # extending this primary-seed entry; the original evaluator is unchanged.
    if type(p['evaluation_seed']) is not int or p['evaluation_seed'] != 20260909:
        raise PermissionError('primary original evaluation seed differs')
    if p['command_profile'] not in ('mixed-center','mixed-nearwall'):
        raise PermissionError('W curriculum unregistered')
    if p['evaluation_checkpoint_origin'] not in ('BASELINE','CANDIDATE'):
        raise PermissionError('W checkpoint interpretation unregistered')
    for k,v in {'residual_neutrality_weight':.05,'anchor_weight':4.,'actor_lr':1e-4,
                'critic_lr':3e-4,'mean_prior_weight':10.,'ics_clearance_margin_m':.15}.items():
        if type(p[k]) not in (int,float) or p[k] != v:
            raise PermissionError('W frozen method scalar differs: '+k)
    if type(p['output_cap_bytes']) is not int or p['output_cap_bytes'] != 8*1024**3:
        raise PermissionError('W output reservation differs')
    for key in ('master_authority','experiment_card','w_closure_contract','mean_prior_contract'):
        validate_pin(p[key])
    if p['w_closure_contract']['sha256'] != CONTRACT_SHA256 or p['mean_prior_contract']['sha256'] != MEAN_PRIOR_CONTRACT_SHA256:
        raise PermissionError('W contract identity differs')
    master=pinned_json(p['master_authority']);card=pinned_json(p['experiment_card'])
    if master['id'] != MASTER_ID or master['status'] != 'REGISTERED_USER_DELEGATION':
        raise PermissionError('W new master delegation missing')
    profile={k:v for k,v in p.items() if k not in ('checkpoints','experiment_card')}
    if card['experiment_id'] != p['experiment_id'] or card['profiles']['R1'] != profile:
        raise PermissionError('W card/profile differs')
    start,target=p['start_update'],p['target_update']
    if type(start) is not int or type(target) is not int or not 512<=start<=target<=2560:
        raise PermissionError('W update bounds invalid')
    baseline=p['evaluation_checkpoint_origin']=='BASELINE'
    if baseline:
        if mode != EVAL or start != 512 or target != 512 or p['checkpoint_updates'] != [] or p['evaluation_updates'] != [512] or p['command_profile'] != 'mixed-center':
            raise PermissionError('fixed baseline control experiment differs')
    else:
        if target-start != 256 or p['checkpoint_updates'] != [start+128,target] or p['evaluation_updates'] != [start+128,target] or p['command_profile']!='mixed-nearwall':
            raise PermissionError('W preregistered 256-update curriculum differs')
    for key in ('checkpoint_updates','evaluation_updates'):
        if any(type(u) is not int for u in p[key]): raise PermissionError('W node type invalid')
    ck=p['initialization_checkpoint']
    if not isinstance(ck,dict) or set(ck)!={'path','sha256','update'} or type(ck['update']) is not int or ck['update']!=start:
        raise PermissionError('W origin checkpoint node invalid')
    validate_pin({k:ck[k] for k in ('path','sha256')})
    if start==512 and ck['sha256']!='a3b9f9115268270ec000750035085b03980de2aed31987b7b08ae97bf26d8776':
        raise PermissionError('W original R1/u512 identity differs')
    hashes=p['origin_method_hashes']
    if not isinstance(hashes,dict) or not hashes or any(not isinstance(k,str) or not isinstance(v,str) or not v for k,v in hashes.items()):
        raise PermissionError('W origin method identity absent')
    validate_source_identity(p['checkpoint_source_identity']);pinned_json(p['checkpoint_compatibility'])
    if mode==EVAL:
        rows=p['checkpoints']
        if not isinstance(rows,list) or [r.get('update') for r in rows if isinstance(r,dict)]!=p['evaluation_updates']:
            raise PermissionError('W evaluation nodes differ')
        for r in rows:
            if set(r)!={'path','sha256','update'} or type(r['update']) is not int:
                raise PermissionError('W checkpoint identity invalid')
            validate_pin({k:r[k] for k in ('path','sha256')})
        if baseline and rows != [ck]: raise PermissionError('control must evaluate exactly original policy')
    return p


def required_budget(mode,p):
    p=validate_parameters(mode,p); train=mode==TRAIN
    n=p['target_update']-p['start_update']; transitions=n*512 if train else 32960*len(p['evaluation_updates'])
    return dict(live_process_starts=1,simulator_initializations=1,initialization_ticks=10,
        training_transitions=transitions if train else 0,nontraining_transitions=0 if train else transitions,
        optimizer_updates=n if train else 0,checkpoint_writes=len(p['checkpoint_updates']) if train else 0,
        explicit_resets=1 if train else 5*len(p['evaluation_updates']),automatic_resets=transitions,
        wall_seconds=3600 if train else 1800)


def method_hashes(config,p):
    from .research_contract import method_hashes as base_hashes
    result=base_hashes(config,p)
    result.update(w_closure_contract=p['w_closure_contract']['sha256'],curriculum='curriculum-h1-v1',
        reset_distribution=p['command_profile'],ics_clearance_margin_m=str(p['ics_clearance_margin_m']),
        initialization='accepted-checkpoint-fresh-physical-reset-v1',origin_checkpoint_sha256=p['initialization_checkpoint']['sha256'])
    return result


def validate_compatibility(pin,origin,config):
    """Explicit product integration delta, not inherited scientific acceptance."""
    from .checkpoint_compatibility import pinned_json,validate_source_identity,digest
    import subprocess
    validate_source_identity(origin);validate_source_identity(config['resolved_identity'])
    r=pinned_json(pin)
    fields={'schema_version','scope','origin_source_identity','execution_source_identity','origin_config','execution_config','changed_product_sources'}
    if set(r)!=fields or r['schema_version']!='instinctrl-w-checkpoint-compatibility/1' or r['scope']!='W_MARGIN_AND_CURRICULUM_INTEGRATION':
        raise PermissionError('W compatibility scope invalid')
    if r['origin_source_identity']!=origin or r['execution_source_identity']!=config['resolved_identity']:
        raise PermissionError('W compatibility source mismatch')
    old=pinned_json(r['origin_config']);new=pinned_json(r['execution_config'])
    if r['origin_config']['sha256']!=origin['config_sha256'] or r['execution_config']['sha256']!=config['resolved_identity']['config_sha256'] or new!={k:v for k,v in config.items() if k!='resolved_identity'}:
        raise PermissionError('W compatibility config mismatch')
    for key in ('method','versions','actor','training','execution'):
        # Method paths relocate with immutable snapshots; scientific bytes must match.
        if key=='method':
            if any(old[key][k]!=new[key][k] for k in ('version','proposal_sha256','hashes')):
                raise PermissionError('W umbrella method changed')
        elif old[key]!=new[key]: raise PermissionError('W invariant config changed: '+key)
    allowed={'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py'}
    a=old['product']['source_sha256'];b=new['product']['source_sha256']
    if set(a)!=set(b): raise PermissionError('W product manifest set drift')
    changed={k:{'origin':a[k],'execution':b[k]} for k in a if a[k]!=b[k]}
    if not set(changed)<=allowed or changed!=r['changed_product_sources']:
        raise PermissionError('W changes outside explicit integration boundary')
    for section in ('product','workload'):
        root=Path(old[section]['path'])
        for suffix,revision in (('commit','HEAD'),('tree','HEAD^{tree}')):
            if subprocess.check_output(['git','-C',str(root),'rev-parse',revision],text=True).strip()!=origin[section+'_'+suffix]:
                raise PermissionError('W original checkout identity drift')
        for relative,expected in old[section]['source_sha256'].items():
            f=root/relative
            if Path(relative).is_absolute() or '..' in Path(relative).parts or f.is_symlink() or digest(f)!=expected:
                raise PermissionError('W original source drift')
    return dict(origin)


def apply_profile(backend):
    p=backend.research
    backend.environment.set_h1_residual_neutrality_weight(p['residual_neutrality_weight'])
    backend.environment.set_h1_research_profile(pure_hover=p['command_profile']=='mixed-center',anchor_weight=p['anchor_weight'])
    backend.environment.set_h1_ics_clearance_margin(p['ics_clearance_margin_m'])
    assert all(pipe.config.ics.clearance_margin_m==p['ics_clearance_margin_m'] for pipe in backend.environment._control_pipelines)
    from training_workloads.scale_003.persistence import atomic_json_replace
    atomic_json_replace(backend.output,'w_profile_binding.json',dict(parameters=p,
        actual_ics_clearance_margin_m=backend.environment._h1_ics_clearance_margin_m,
        actual_anchor_weight=backend.environment._h1_anchor_weight,
        actual_residual_neutrality_weight=backend.environment._h1_residual_neutrality_weight,
        center_reset_only=backend.environment._h1_pure_hover_training,
        authorization_sha256=backend.authorization['authorization_sha256'],
        actor_input_extension=False,control_velocity_feedback_added=False))


def train_segment(backend):
    from .config import sha256_file
    from .schedule import ScheduleClock,command_at
    from .driver import UnifiedH1Driver,UnifiedH1TrainingLoop
    from .logging import LearningLog
    from .pilot_diagnostic import FirstUpdateDiagnostic
    from .mean_prior import configure_trainer
    from training_workloads.scale_003.persistence import atomic_json_replace
    if backend.authorization['mode']!=TRAIN: raise PermissionError('W train mode mismatch')
    p=backend.research;apply_profile(backend);ck=p['initialization_checkpoint']
    if sha256_file(Path(ck['path']))!=ck['sha256']: raise ValueError('W origin drift')
    meta=backend._load_evaluation_checkpoint(Path(ck['path']),actor_learning_rate=p['actor_lr'],
        expected_method_hashes=p['origin_method_hashes'],source_identity=p['checkpoint_source_identity'],restore_curriculum_generator=True)
    start=p['start_update']
    if backend.trainer.update_counter!=start or meta['training_seed']!=p['training_seed'] or meta['curriculum']['version']!='curriculum-h1-v1' or meta['counters']['global_policy_step']!=start*16:
        raise ValueError('W actual origin phase/seed/clock mismatch')
    configure_trainer(backend.trainer,p)
    backend._checkpoint_reset_offset=meta['reset_number'];backend._formal_segment=True
    backend.environment.set_h1_evaluation_case(None)
    backend.environment.set_h1_initial_command(command_at(start*16,device=backend.environment.device),start*16)
    backend.resource_check('before_w_fresh_reset',1)
    backend.observation,_=backend._preserving_reset(seed=p['training_seed']+meta['reset_number']+1)
    from .schedule import reset_local_positions
    initial_positions,initial_labels=reset_local_positions(start*16)
    actual_local=backend.environment._robot.data.root_state_w[:,:3]-backend.environment.scene.env_origins
    import torch
    if not torch.allclose(actual_local.detach().cpu(),initial_positions,atol=.02,rtol=0):
        raise RuntimeError('W fresh reset actual positions differ from declared distribution')
    journal=LearningLog(backend.output/'learning_log',initial_start_types=initial_labels);backend.journal=journal
    atomic_json_replace(backend.output,'initial_training_distribution.json',dict(labels=initial_labels,
        expected_local_positions=initial_positions.tolist(),actual_local_positions=actual_local.detach().cpu().tolist(),
        global_policy_step=start*16,position_check_atol_m=.02))
    backend.driver=UnifiedH1Driver(backend.environment,backend.observation,clock=ScheduleClock(start*16),counters=backend.counters,journal=journal,
        before_step=lambda u:backend.resource_check('before_w_step',u),transition_limit=backend.authorization['budgets']['training_transitions'],automatic_reset_limit=backend.authorization['budgets']['automatic_resets'])
    backend.loop=UnifiedH1TrainingLoop(backend.driver,backend.trainer,journal,backend._write_checkpoint,
        before_rollout=lambda u:backend.resource_check('before_w_rollout',u),before_optimizer=lambda u:backend.resource_check('before_w_optimizer',u),
        update_limit=backend.authorization['budgets']['optimizer_updates'],optimizer_step_origin=(start*8,start*8),
        diagnostic=FirstUpdateDiagnostic(backend.output/'pilot_diagnostic',enabled=True,starting_update=start))
    atomic_json_replace(backend.output,'actor_mean_prior_binding.json',dict(mean_prior_contract=p['mean_prior_contract'],actual_weight=p['mean_prior_weight'],hook_installed=backend.trainer.actor_auxiliary is not None,actor_only=True,runtime_action_replacement=False,starting_update=start,authorization_sha256=backend.authorization['authorization_sha256']))
    atomic_json_replace(backend.output,'w_initialization.json',dict(origin=ck,metadata=meta,phase=p,initial_global_policy_step=start*16,restored_adam_steps_each=start*8,physical_history_anchor_actuator_continuity_claimed=False))
    while backend.trainer.update_counter<p['target_update']:
        u=backend.trainer.update_counter+1;backend.resource_check('before_w_update',u)
        backend.loop.run_update(checkpoint_label=f'u{u}' if u in p['checkpoint_updates'] else None)


def evaluate_segment(backend):
    from .config import load_config,sha256_file
    from .evaluation import CASES,evaluate_checkpoint,wall_motion_prerequisite
    from training_workloads.scale_003.persistence import atomic_json_replace
    if backend.authorization['mode']!=EVAL: raise PermissionError('W eval mode mismatch')
    p=backend.research;apply_profile(backend);baseline=p['evaluation_checkpoint_origin']=='BASELINE'
    for ck in p['checkpoints']:
        if sha256_file(Path(ck['path']))!=ck['sha256']:raise ValueError('W evaluation checkpoint drift')
        meta=backend._load_evaluation_checkpoint(Path(ck['path']),actor_learning_rate=p['actor_lr'],
            expected_method_hashes=p['origin_method_hashes'] if baseline else method_hashes(load_config(),p),
            source_identity=p['checkpoint_source_identity'] if baseline else load_config()['resolved_identity'])
        if backend.trainer.update_counter!=ck['update'] or meta['training_seed']!=p['training_seed'] or meta['curriculum']['version']!='curriculum-h1-v1':
            raise ValueError('W evaluation actual checkpoint mismatch')
        backend.trainer.actor.eval();backend.trainer.critic.eval();backend._evaluation_rows.clear()
        label=f'w_{p["group"]}_u{ck["update"]}'
        for case in CASES:
            backend.resource_check('before_w_case',ck['update']);backend._evaluate_case(case,direct_b0=False,evaluation_label=label)
        result=evaluate_checkpoint([t for c in CASES for t in backend._evaluation_rows[c]],update=ck['update'])
        atomic_json_replace(backend.output,label+'.json',result)
        # Additional Goal condition, never overwrite original metric outputs.
        raw=backend.output/'evaluation_raw'/label/'W.jsonl'
        rows=[json.loads(line) for line in raw.read_text().splitlines()][1:]
        motion=[wall_motion_prerequisite([r['root_state_w'][lane][7+(0 if lane<16 else 1)] for r in rows]) for lane in range(32)]
        atomic_json_replace(backend.output,label+'_goal_motion.json',dict(world_normal_11_sample_approach=motion,all32=all(motion),original_metric_unchanged=True))
