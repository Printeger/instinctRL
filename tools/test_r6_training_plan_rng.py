"""Real CPU RNG composition; synthetic loaded trainer/counters, no GPU or PPO."""
from dataclasses import replace
from unittest.mock import patch
import pytest
from tools.test_r6_training_plan import case,metadata
from tools.test_r6_training_rng import restored,weights,preserve_global_rng
from training_workloads.unified_h1 import robustness_training_rng as r
from training_workloads.unified_h1.backend import _state_fingerprint
from instinctlab.training.checkpoint_h1 import _rng_payload


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
