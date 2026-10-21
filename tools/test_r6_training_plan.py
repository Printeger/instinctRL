"""CPU input-contract tests; synthetic checkpoint pin, no tensor restore or job."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from training_workloads.unified_h1 import robustness_training_plan as p


@pytest.fixture
def case(tmp_path,monkeypatch):
    from types import SimpleNamespace
    def pin(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    identity={k:'a'*(64 if k=='config_sha256' else 40) for k in
        ('config_sha256','product_commit','product_tree','workload_commit','workload_tree')}
    ev=pin('diagnosis.json',{'fixture':True})
    r=dict(schema_version='r6-training-recipe/1',candidate_id='R6-CANDIDATE-CPU',baseline_sha256=p.BASELINE_SHA,
        diagnosis=[ev],hypothesis='CPU fixture curriculum',comparison='same u640 control',source_identity=identity,
        curriculum=pin('curriculum.json',{'fixture':True}),method=pin('method.json',{'ppo':{'minibatch_size':64,'epoch_count':1}}),
        branch_updates=1024,development_seeds=[11],confirmation_seeds=[21,22,23],checkpoint_offsets=[256,512,1024])
    rp=pin('recipe.json',r);review=pin('review.json',dict(decision='PASS_R6_TRAINING_RECIPE',must_fix=[],reviewer='fixture-independent',payload_sha256=p.object_sha(r)))
    initial=dict(kind='COMMON_U640_NEW_STREAM',checkpoint=dict(path=ev['path'],sha256=p.BASELINE_SHA),
        source_identity=identity,method_hashes={'baseline':'fixture'})
    original=p.validate_pin
    monkeypatch.setattr(p,'validate_pin',lambda v:Path(v['path']) if v==initial['checkpoint'] else original(v))
    plan=dict(schema_version='r6-training-plan/1',experiment_id='R6-TRAINING-CPU',phase='development',recipe=rp,
        recipe_review=review,selection_freeze=None,training_seed=11,start_update=640,target_update=896,
        checkpoint_updates=[896],initialization=initial,wall_seconds=2290,
        storage=dict(payload_bytes=4*1024**3,compressed_bytes=2*1024**3,plain_bytes=2*1024**3,
            maximum_raw_file_bytes=1024**3,checkpoint_bytes=256*1024**2,branch_record_bytes=1024**2,metadata_bytes=64*1024**2))
    return SimpleNamespace(plan=plan,recipe=r,pin=pin,identity=identity,ev=ev)


def update_recipe(b):
    b.plan['recipe']=b.pin('recipe.json',b.recipe)
    b.plan['recipe_review']=b.pin('review.json',dict(decision='PASS_R6_TRAINING_RECIPE',must_fix=[],reviewer='fixture-independent',payload_sha256=p.object_sha(b.recipe)))


def metadata(update=640,seed=20260908,curriculum=None):
    return dict(training_seed=seed,counters=dict(update=update,transition=update*512,global_policy_step=update*16,
        actor_optimizer=update*8,critic_optimizer=update*8),curriculum=curriculum or {'version':'curriculum-h1-v1','completed_policy_steps':update*16})


def test_development_exact_training_checkpoint_and_physics_arithmetic(case):
    result=p.expanded_budget(case.plan,physics_decimation=4)
    assert result['budgets']==dict(live_process_starts=1,simulator_initializations=1,initialization_ticks=10,
        training_transitions=131072,nontraining_transitions=0,optimizer_updates=256,checkpoint_writes=1,
        explicit_resets=1,automatic_resets=131072,wall_seconds=2290)
    assert result['policy_steps']==4096 and result['physics_steps']==16384
    assert result['actor_optimizer_steps']==result['critic_optimizer_steps']==2048
    assert not result['dispatch_authorized']
    assert p.verify_loaded(case.plan,metadata())['branch_new_rng']


@pytest.mark.parametrize('change',['node','boolean','target','plain','seed','checkpoint','source_clock'])
def test_plan_drift_is_rejected_before_any_action(case,change):
    plan=case.plan
    if change=='node':plan['checkpoint_updates']=[704,896]
    elif change=='boolean':plan['start_update']=True
    elif change=='target':plan['target_update']=900
    elif change=='plain':plan['storage']['plain_bytes']=1
    elif change=='seed':plan['training_seed']=22
    elif change=='checkpoint':plan['initialization']['kind']='RESUME_R6_STREAM'
    else:
        case.recipe['method']=case.pin('method-other.json',{'ppo':{'minibatch_size':64,'epoch_count':2}});update_recipe(case)
    with pytest.raises(PermissionError):p.validate(plan)


def test_confirmation_needs_frozen_recipe_and_distinct_registered_seeds(case):
    b=case;b.plan.update(phase='confirmation',training_seed=21)
    with pytest.raises((PermissionError,TypeError)):p.validate(b.plan)
    freeze=dict(schema_version='r6-recipe-freeze/1',recipe=b.plan['recipe'],confirmation_seeds=[21,22,23],selection_evidence=b.ev)
    review=b.pin('freeze-review.json',dict(decision='PASS_R6_RECIPE_FREEZE',must_fix=[],reviewer='fixture-independent',payload_sha256=p.object_sha(freeze)))
    b.plan['selection_freeze']=b.pin('freeze.json',dict(freeze,independent_review=review))
    assert p.validate(b.plan)['phase']=='confirmation'
    b.recipe['confirmation_seeds']=[11,22,23];update_recipe(b)
    with pytest.raises(PermissionError,match='SEED_REUSE'):p.validate(b.plan)


def test_resume_keeps_branch_recipe_and_rng_identity_without_rebranching(case):
    b=case;b.plan.update(start_update=896,target_update=1152,checkpoint_updates=[1152])
    b.plan['initialization']=dict(kind='RESUME_R6_STREAM',checkpoint=b.ev,source_identity=b.identity,
        method_hashes={'r6_recipe':b.plan['recipe']['sha256']})
    curriculum=dict(version='curriculum-h1-r6-v1',completed_policy_steps=896*16,
        recipe_sha256=b.plan['recipe']['sha256'],origin_checkpoint_sha256=p.BASELINE_SHA,branch_seed=11)
    result=p.verify_loaded(b.plan,metadata(896,11,curriculum))
    assert not result['branch_new_rng'] and result['fresh_physical_reset']
    curriculum['recipe_sha256']='f'*64
    with pytest.raises(PermissionError,match='LOADED_BRANCH'):p.verify_loaded(b.plan,metadata(896,11,curriculum))


@pytest.mark.parametrize('field',['actor_optimizer','transition','update'])
def test_loaded_origin_counter_mismatch_rejects(case,field):
    m=metadata();m['counters'][field]+=1
    with pytest.raises(PermissionError,match='LOADED_COUNTERS'):p.verify_loaded(case.plan,m)


def test_missing_or_rejected_independent_recipe_review(case):
    case.plan['recipe_review']=case.pin('rejected.json',dict(decision='MUST_FIX',must_fix=['fixture'],reviewer='fixture-independent'))
    with pytest.raises(PermissionError):p.validate(case.plan)


def test_three_confirmation_paths_must_fit_training_total(case):
    case.recipe['branch_updates']=1954;case.recipe['checkpoint_offsets']=[256,1954];update_recipe(case)
    with pytest.raises(PermissionError,match='BRANCH_UPDATES'):p.validate(case.plan)
