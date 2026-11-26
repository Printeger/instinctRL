"""Real CPU RNG composition; synthetic loaded trainer/counters, no GPU or PPO."""
from dataclasses import asdict, replace
from unittest.mock import patch
import pytest
from tools.test_r6_training_plan import case,metadata
from tools.test_r6_training_rng import restored,weights,preserve_global_rng
from training_workloads.unified_h1 import robustness_training_rng as r
from training_workloads.unified_h1.backend import _state_fingerprint
from instinctlab.training.checkpoint_h1 import _rng_payload
from tools.r6_direct.run import resources_after_validation
from tools.r6_training_rng_audit import audit
from training_workloads.unified_h1 import robustness_curriculum, robustness_training_plan as contract
from training_workloads.unified_h1.robustness_training_method import bound_method_hashes


def test_validated_common_origin_branches_once_without_optimizer(case):
    t,g,_=restored();before=weights(t)
    out=r.prepare_plan_rng(t,g,plan=case.plan,metadata=metadata())
    assert out['rng']['branch_training_seed']==11 and t.config.permutation_seed==11
    assert out['evidence_must_be_persisted_before_rollout'] and weights(t)==before
    with pytest.raises(ValueError,match='ALREADY_PREPARED'):
        r.prepare_plan_rng(t,g,plan=case.plan,metadata=metadata())


def test_r6_resume_preserves_all_streams_and_checks_permutation(case):
    b=case;t,g,_=restored();t.update_counter=896;t.config=replace(t.config,permutation_seed=11)
    for optimizer in (t.actor_optimizer,t.critic_optimizer):
        for state in optimizer.state.values():state['step'].fill_(7168)
    b.plan.update(start_update=896,target_update=1152,checkpoint_updates=[1152])
    b.plan['initialization']=dict(kind='RESUME_R6_STREAM',checkpoint=b.ev,source_identity=b.identity,
        method_hashes={'r6_recipe':b.plan['recipe']['sha256']})
    curriculum=dict(version='curriculum-h1-r6-v1',completed_policy_steps=896*16,recipe_sha256=b.plan['recipe']['sha256'],
        origin_checkpoint_sha256=r.BASELINE_SHA,branch_seed=11)
    before=_state_fingerprint(_rng_payload(g,t));state=weights(t)
    with patch.object(r,'branch_from_u640',side_effect=AssertionError('resume must not branch')):
        out=r.prepare_plan_rng(t,g,plan=b.plan,metadata=metadata(896,11,curriculum))
    assert out['rng']['strict_rng_continuation'] and not out['binding']['branch_new_rng']
    assert _state_fingerprint(_rng_payload(g,t))==before and weights(t)==state


@pytest.mark.parametrize('bad',['metadata','actual_counter','clock'])
def test_plan_or_actual_trainer_mismatch_never_branches(case,bad):
    t,g,_=restored();m=metadata()
    if bad=='metadata':m['training_seed']=11
    elif bad=='actual_counter':t.update_counter=639
    else:t.config=replace(t.config,epoch_count=2)
    before=_state_fingerprint(_rng_payload(g,t))
    with patch.object(r,'branch_from_u640',side_effect=AssertionError('must reject before branch')):
        with pytest.raises((ValueError,PermissionError)):r.prepare_plan_rng(t,g,plan=case.plan,metadata=m)
    assert _state_fingerprint(_rng_payload(g,t))==before


@pytest.mark.parametrize('start, kind, previous_seed, sha_field',[
    (768,contract.LOCAL_FORK,contract.R3_TRAINING_SEED,'R3_CHECKPOINT_SHA'),
    (1792,contract.U1792_LOCAL_FORK,contract.U1792_TRAINING_SEED,'U1792_CHECKPOINT_SHA'),
])
def test_local_branch_and_registered_resume_keep_absolute_clocks(case,monkeypatch,start,kind,previous_seed,sha_field):
    b=case
    p=b.plan
    p.pop('recipe_review')
    monkeypatch.setattr(contract,sha_field,b.ev['sha256'])
    b.recipe.update(branch_origin=dict(checkpoint=b.ev,update=start,training_seed=previous_seed))
    p.update(schema_version='r6-local-training-plan/1',recipe=b.pin('recipe-local.json',b.recipe),
        start_update=start,target_update=start+256,checkpoint_updates=[start+256],
        initialization=dict(kind=kind,checkpoint=b.ev,source_identity=b.identity,
            method_hashes={'r6_recipe':'f'*64}))
    t,g,_=restored()
    t.update_counter=start
    t.config=replace(t.config,permutation_seed=previous_seed)
    for optimizer in (t.actor_optimizer,t.critic_optimizer):
        for state in optimizer.state.values():state['step'].fill_(start*8)
    old_clock=dict(version='curriculum-h1-r6-v1',completed_policy_steps=start*16,
        recipe_sha256='f'*64,origin_checkpoint_sha256=contract.BASELINE_SHA,branch_seed=previous_seed)
    meta=metadata(start,previous_seed,old_clock)
    origin=dict(meta,rng=_rng_payload(g,t),ppo_config=asdict(t.config),
        source_identity=b.identity,method_hashes=p['initialization']['method_hashes'])
    before=weights(t)
    wrong_kind = next(name for name in contract.LOCAL_FORKS if name != kind)
    wrong_plan = dict(p,initialization=dict(p['initialization'],kind=wrong_kind))
    with pytest.raises(PermissionError,match='LOCAL_BRANCH_INITIALIZATION'):
        contract.validate(wrong_plan)
    fork=r.prepare_plan_rng(t,g,plan=p,metadata=meta)
    assert t.config.permutation_seed==11 and weights(t)==before
    assert audit(fork,origin,p)['cpu_seed_derivation_verified']
    fork['rng']['origin_training_seed']=20260908
    with pytest.raises(ValueError,match='RNG_BRANCH_IDENTITY'):audit(fork,origin,p)
    budget=contract.required_budget(p)
    assert budget['training_transitions']==256*512 and budget['explicit_resets']==2
    assert resources_after_validation(dict(operation='train',parameters={'r6_training_plan':p},
        budgets=budget))['major_scientific_candidates']==1
    curriculum=contract.pinned_json(b.recipe['curriculum'])
    phase=robustness_curriculum.PhaseCurriculum(curriculum,branch_updates=1024,phase_index=1,
        generator=g,audit_sink=lambda *_:None,branch_start_update=start)
    assert (phase.start,phase.end)==((start+64)*16,(start+1024)*16)

    hashes=bound_method_hashes(p,b.recipe)
    p.update(start_update=start+256,target_update=start+512,checkpoint_updates=[start+512],
        initialization=dict(kind='RESUME_R6_STREAM',checkpoint=b.ev,source_identity=b.identity,method_hashes=hashes))
    new_clock=dict(old_clock,completed_policy_steps=(start+256)*16,recipe_sha256=p['recipe']['sha256'],branch_seed=11)
    assert not contract.verify_loaded(p,metadata(start+256,11,new_clock))['branch_new_rng']
    assert contract.required_budget(p)['explicit_resets']==1
    hashes.pop('r6_branch_origin')
    with pytest.raises(PermissionError,match='RESUME_BRANCH_ORIGIN'):contract.validate(p)
