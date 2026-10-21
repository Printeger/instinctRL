"""Exact R6 training plan arithmetic and loaded-checkpoint provenance.

This validates proposed inputs; it neither enables a dispatch mode nor grants
Store authority. Live integration must also validate actual trainer/RNG state,
install the curriculum, and retain the existing pre-optimizer probability gate.
"""
import re
from . import robustness_curriculum
from .research_contract import validate_pin
from .checkpoint_compatibility import pinned_json,validate_source_identity
from .robustness_contract import BASELINE_SHA,accepted_review,object_sha

FIELDS={'schema_version','experiment_id','phase','recipe','recipe_review','selection_freeze',
        'training_seed','start_update','target_update','checkpoint_updates','initialization',
        'wall_seconds','storage'}
RECIPE_FIELDS={'schema_version','candidate_id','baseline_sha256','diagnosis','hypothesis','comparison',
    'source_identity','curriculum','method','branch_updates','development_seeds','confirmation_seeds',
    'checkpoint_offsets'}
STORAGE_FIELDS={'payload_bytes','compressed_bytes','plain_bytes','maximum_raw_file_bytes',
                'checkpoint_bytes','branch_record_bytes','metadata_bytes'}


def require(condition,reason):
    if not condition:raise PermissionError('R6_TRAINING_PLAN_'+reason)


def integer(value,minimum,maximum,reason):
    require(type(value)is int and minimum<=value<=maximum,reason)
    return value


def seeds(values,count=None):
    require(isinstance(values,list) and 1<=len(values)<=3
        and (count is None or len(values)==count),'SEEDS')
    for seed in values:integer(seed,0,2**31-1,'SEED')
    require(len(set(values))==len(values) and 20260908 not in values,'SEED_REUSE')


def recipe(pin,review_pin):
    r=validate_recipe(pinned_json(pin))
    accepted_review(review_pin,'PASS_R6_TRAINING_RECIPE',payload_sha256=object_sha(r))
    return r


def validate_recipe(r):
    """Validate prospective recipe content; this never supplies review acceptance."""
    require(isinstance(r,dict) and set(r)==RECIPE_FIELDS and r['schema_version']=='r6-training-recipe/1','RECIPE_FIELDS')
    require(isinstance(r['candidate_id'],str) and re.fullmatch(r'R6-CANDIDATE-[A-Z0-9-]{1,64}',r['candidate_id']),'CANDIDATE')
    require(r['baseline_sha256']==BASELINE_SHA,'ORIGIN')
    for name in ('hypothesis','comparison'):
        require(isinstance(r[name],str) and bool(r[name].strip()) and len(r[name])<=4096,'SCIENTIFIC_DESIGN')
    require(isinstance(r['diagnosis'],list) and 1<=len(r['diagnosis'])<=16,'DIAGNOSIS')
    for evidence in r['diagnosis']:validate_pin(evidence)
    validate_source_identity(r['source_identity'])
    for name in ('curriculum','method'):
        require(isinstance(pinned_json(r[name]),dict),'RECIPE_INPUT')
    from .robustness_training_method import validate as validate_method
    method=validate_method(pinned_json(r['method']));ppo=method['ppo']
    require(isinstance(ppo,dict),'PPO')
    batch=integer(ppo.get('minibatch_size'),1,512,'MINIBATCH')
    epochs=integer(ppo.get('epoch_count'),1,8,'EPOCHS')
    require(epochs*((512+batch-1)//batch)==8,'CHECKPOINT_OPTIMIZER_CLOCK')
    integer(r['branch_updates'],1,3000000//(3*512),'BRANCH_UPDATES')
    robustness_curriculum.validate(pinned_json(r['curriculum']),branch_updates=r['branch_updates'])
    seeds(r['development_seeds']);seeds(r['confirmation_seeds'],3)
    require(not set(r['development_seeds']) & set(r['confirmation_seeds']),'CONFIRMATION_SEED_REUSE')
    nodes=r['checkpoint_offsets']
    require(isinstance(nodes,list) and 1<=len(nodes)<=96,'CHECKPOINTS')
    for node in nodes:integer(node,1,r['branch_updates'],'CHECKPOINT')
    require(nodes==sorted(set(nodes)) and nodes[-1]==r['branch_updates'],'FINAL_CHECKPOINT')
    return dict(r)


def validate(plan):
    require(isinstance(plan,dict) and set(plan)==FIELDS and plan['schema_version']=='r6-training-plan/1','FIELDS')
    require(isinstance(plan['experiment_id'],str) and re.fullmatch(r'R6-[A-Z0-9-]{1,90}',plan['experiment_id']),'EXPERIMENT')
    require(plan['phase'] in ('development','confirmation'),'PHASE')
    r=recipe(plan['recipe'],plan['recipe_review'])
    seed=integer(plan['training_seed'],0,2**31-1,'TRAINING_SEED')
    require(seed in r[plan['phase']+'_seeds'],'UNREGISTERED_SEED')
    start=integer(plan['start_update'],640,640+r['branch_updates']-1,'START')
    target=integer(plan['target_update'],start+1,640+r['branch_updates'],'TARGET')
    allowed=[640+n for n in r['checkpoint_offsets'] if start<640+n<=target]
    require(plan['checkpoint_updates']==allowed and allowed and allowed[-1]==target
        and all(type(n)is int for n in plan['checkpoint_updates']),'EXACT_CHECKPOINTS')
    initial=plan['initialization']
    require(isinstance(initial,dict) and set(initial)=={'kind','checkpoint','source_identity','method_hashes'},'INITIALIZATION')
    validate_pin(initial['checkpoint']);validate_source_identity(initial['source_identity'])
    require(isinstance(initial['method_hashes'],dict) and initial['method_hashes'],'INITIAL_METHOD')
    if initial['kind']=='COMMON_U640_NEW_STREAM':
        require(start==640 and initial['checkpoint']['sha256']==BASELINE_SHA,'COMMON_ORIGIN')
    else:
        require(initial['kind']=='RESUME_R6_STREAM' and start>640
            and initial['source_identity']==r['source_identity']
            and initial['method_hashes'].get('r6_recipe')==plan['recipe']['sha256'],'RESUME_IDENTITY')
    if plan['phase']=='confirmation':
        freeze=pinned_json(plan['selection_freeze'])
        require(set(freeze)=={'schema_version','recipe','confirmation_seeds','selection_evidence','independent_review'}
            and freeze['schema_version']=='r6-recipe-freeze/1' and freeze['recipe']==plan['recipe']
            and freeze['confirmation_seeds']==r['confirmation_seeds'],'RECIPE_FREEZE')
        validate_pin(freeze['selection_evidence'])
        accepted_review(freeze['independent_review'],'PASS_R6_RECIPE_FREEZE',
            payload_sha256=object_sha({k:v for k,v in freeze.items() if k!='independent_review'}))
    else:require(plan['selection_freeze'] is None,'DEVELOPMENT_FREEZE')
    integer(plan['wall_seconds'],1,43200,'WALL')
    storage=plan['storage'];require(isinstance(storage,dict) and set(storage)==STORAGE_FIELDS,'STORAGE_FIELDS')
    for name,value in storage.items():integer(value,1,128*1024**3,'STORAGE_'+name)
    require(storage['payload_bytes']%4096==0
        and storage['compressed_bytes']+storage['plain_bytes']<=storage['payload_bytes'],'PAYLOAD')
    branch=storage['branch_record_bytes'] if initial['kind']=='COMMON_U640_NEW_STREAM' else 0
    require(storage['plain_bytes']>=len(allowed)*storage['checkpoint_bytes']+branch+storage['metadata_bytes'],'PLAIN_CAPACITY')
    return dict(plan)


def required_budget(plan):
    p=validate(plan);updates=p['target_update']-p['start_update'];transitions=updates*512
    curriculum=pinned_json(pinned_json(p['recipe'])['curriculum'])
    resets=len(robustness_curriculum.slices(curriculum,p['start_update'],p['target_update']))
    return dict(live_process_starts=1,simulator_initializations=1,initialization_ticks=10,
        training_transitions=transitions,nontraining_transitions=0,optimizer_updates=updates,
        checkpoint_writes=len(p['checkpoint_updates']),explicit_resets=resets,
        automatic_resets=transitions,wall_seconds=p['wall_seconds'])


def expanded_budget(plan,*,physics_decimation):
    p=validate(plan);integer(physics_decimation,20,20,'PHYSICS_DECIMATION')
    updates=p['target_update']-p['start_update']
    return dict(budgets=required_budget(p),policy_steps=updates*16,
        physics_steps=updates*16*physics_decimation,
        actor_optimizer_steps=updates*8,critic_optimizer_steps=updates*8,
        retained_checkpoint_updates=list(p['checkpoint_updates']),fresh_physical_reset=True,
        dispatch_authorized=False)


def verify_loaded(plan,metadata):
    """After the existing strict tensor restore, before branch/reset/rollout."""
    p=validate(plan);start=p['start_update'];initial=p['initialization']
    counters=dict(update=start,transition=start*512,global_policy_step=start*16,
                  actor_optimizer=start*8,critic_optimizer=start*8)
    require(metadata.get('counters')==counters and all(type(v)is int for v in metadata.get('counters',{}).values()),'LOADED_COUNTERS')
    require(type(metadata.get('training_seed')) is int,'LOADED_SEED')
    curriculum=metadata.get('curriculum',{})
    if initial['kind']=='COMMON_U640_NEW_STREAM':
        require(metadata.get('training_seed')==20260908 and curriculum==dict(
            version='curriculum-h1-v1',completed_policy_steps=start*16),'LOADED_ORIGIN')
    else:
        require(metadata.get('training_seed')==p['training_seed'] and curriculum==dict(
            version='curriculum-h1-r6-v1',completed_policy_steps=start*16,recipe_sha256=p['recipe']['sha256'],
            origin_checkpoint_sha256=BASELINE_SHA,branch_seed=p['training_seed']),'LOADED_BRANCH')
    return dict(start_update=start,branch_new_rng=initial['kind']=='COMMON_U640_NEW_STREAM',
                fresh_physical_reset=True,dispatch_authorized=False)
