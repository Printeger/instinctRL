"""Actual strict CPU load from explicit u640 derivative; selection validation isolated."""
from dataclasses import replace,asdict
from types import SimpleNamespace
import copy
import hashlib
import pytest
import torch
from tools.test_r6_checkpoint_branch import origin
from tools.test_r6_training_rng import preserve_global_rng
from tools.test_r6_training_method import Env
from tools.test_r6_training_plan import method_fixture
from training_workloads.unified_h1 import robustness_evaluation_selection as selection
from training_workloads.unified_h1.robustness_candidate_restore import restore
from training_workloads.unified_h1.robustness_training_method import apply_method
from training_workloads.unified_h1.backend import _state_fingerprint
from instinctlab.training.checkpoint_h1 import save_checkpoint_h1


@pytest.fixture
def selected(origin,tmp_path,monkeypatch):
    t,g,meta,payload=origin
    def build(role):
        method=method_fixture();u=640;seed=20260908;hashes=payload['method_hashes'];curriculum=meta['curriculum']
        if role=='trained_candidate':
            u=896;seed=11;t.update_counter=u
            for optimizer in (t.actor_optimizer,t.critic_optimizer):
                for state in optimizer.state.values():state['step'].fill_(u*8)
            apply_method(t,Env(),method,resume=False,training_seed=seed);t.config=replace(t.config,permutation_seed=seed)
            hashes=dict(hashes,r6_recipe='1'*64)
            curriculum=dict(version='curriculum-h1-r6-v1',completed_policy_steps=u*16,recipe_sha256='1'*64,origin_checkpoint_sha256=selection.base.BASELINE_SHA,branch_seed=seed)
        path=tmp_path/(role+'.pt')
        save_checkpoint_h1(path,t,method_hashes=hashes,source_identity=payload['source_identity'],transition_counter=u*512,global_policy_step=u*16,
            curriculum=curriculum,training_seed=seed,reset_number=meta['reset_number'],curriculum_generator=g)
        model=dict(checkpoint=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),update=u,training_seed=seed,source_identity=payload['source_identity'],method_hashes=hashes,
            ppo_config=asdict(t.config) if role=='trained_candidate' else None)
        value=dict(role=role,training_plan={'recipe':{'sha256':'1'*64}})
        result=dict(selection=value,model=model,method=method)
        monkeypatch.setattr(selection,'validate',lambda incoming:result)
        obj=SimpleNamespace(authorization=dict(mode='research-r6-candidate-evaluation',parameters={'evaluation_selection':value}),environment=Env(),
            trainer=None,driver=None,loop=None,journal=None,curriculum_generator=torch.Generator().manual_seed(43))
        return obj,value,result,t
    return build


@pytest.mark.parametrize('role',['baseline_control','trained_candidate'])
def test_exact_selected_model_load_without_optimizer_update(selected,role):
    obj,value,expected,t=selected(role)
    before=_state_fingerprint([t.actor.state_dict(),t.critic.state_dict(),t.actor_optimizer.state,t.critic_optimizer.state,t.update_counter])
    result=restore(obj,value)
    after=_state_fingerprint([obj.trainer.actor.state_dict(),obj.trainer.critic.state_dict(),obj.trainer.actor_optimizer.state,obj.trainer.critic_optimizer.state,obj.trainer.update_counter])
    assert before==after and not obj.trainer.actor.training and not obj.trainer.critic.training
    assert result['optimizer_updates']==0 and not result['physical_history_continuity_claimed']
    assert obj.environment._h1_ics_clearance_margin_m==expected['method']['ics_clearance_margin_m']


def test_failed_restore_keeps_no_trainer_and_preserves_rng(selected):
    obj,value,expected,t=selected('trained_candidate');expected['model']['source_identity']={'wrong':'source'}
    before=torch.get_rng_state().clone();curr=obj.curriculum_generator.get_state().clone()
    with pytest.raises(ValueError,match='identity mismatch'):restore(obj,value)
    assert obj.trainer is None and obj._r6_load_failed
    assert torch.equal(before,torch.get_rng_state()) and torch.equal(curr,obj.curriculum_generator.get_state())
    with pytest.raises(PermissionError,match='FRESH_BACKEND'):restore(obj,value)


def test_baseline_authorization_cannot_load_candidate(selected):
    obj,value,expected,t=selected('trained_candidate');obj.authorization['mode']='research-r6-evaluation'
    with pytest.raises(PermissionError,match='EXACT_AUTHORIZATION'):restore(obj,value)
    assert obj.trainer is None
