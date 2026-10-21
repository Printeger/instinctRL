"""Strict real CPU checkpoint load; final selection validation is an explicit fixture."""
import pytest
import torch
from tools.test_r6_candidate_restore import selected
from tools.test_r6_checkpoint_branch import origin
from tools.test_r6_training_rng import preserve_global_rng
from training_workloads.unified_h1 import robustness_final_selection as final
from training_workloads.unified_h1.robustness_final_restore import restore
from training_workloads.unified_h1.backend import _state_fingerprint


def configure(selected,monkeypatch,role):
    backend,value,result,trainer=selected(role)
    if role=='baseline_control':result['method']=None
    monkeypatch.setattr(final,'validate',lambda incoming:result)
    value={'schema_version':'r6-final-selection/1','fixture':role}
    backend.authorization=dict(mode=final.MODE,parameters={'final_selection':value})
    return backend,value,result,trainer


@pytest.mark.parametrize('role',['baseline_control','trained_candidate'])
def test_final_restore_preserves_weights_adam_and_has_no_optimizer_step(selected,monkeypatch,role):
    backend,value,result,trainer=configure(selected,monkeypatch,role)
    def signature(t):return _state_fingerprint([t.actor.state_dict(),t.critic.state_dict(),t.actor_optimizer.state,t.critic_optimizer.state,t.update_counter])
    expected=signature(trainer);out=restore(backend,value)
    assert signature(backend.trainer)==expected and out['optimizer_updates']==0
    assert not backend.trainer.actor.training and not out['physical_history_continuity_claimed']
    if role=='baseline_control':
        assert backend.environment._h1_ics_clearance_margin_m==.15 and backend.environment._h1_anchor_weight==4.
        assert backend.environment._h1_residual_neutrality_weight==.05
        assert out['method_binding']['unchanged_checkpoint_ppo']is True


@pytest.mark.parametrize('mode',['research-r6-candidate-evaluation','research-r6-evaluation','research-r6-training'])
def test_other_stage_authority_cannot_load_final_model(selected,monkeypatch,mode):
    backend,value,_,_=configure(selected,monkeypatch,'trained_candidate');backend.authorization['mode']=mode
    with pytest.raises(PermissionError,match='EXACT_AUTHORIZATION'):restore(backend,value)
    assert backend.trainer is None


def test_failed_final_metadata_restore_preserves_rng_and_disables_backend(selected,monkeypatch):
    backend,value,result,_=configure(selected,monkeypatch,'trained_candidate');result['model']['training_seed']+=1
    cpu=torch.get_rng_state().clone();curriculum=backend.curriculum_generator.get_state().clone()
    with pytest.raises(PermissionError,match='COUNTERS_SEED'):restore(backend,value)
    assert backend.trainer is None and backend._r6_load_failed
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(curriculum,backend.curriculum_generator.get_state())
    with pytest.raises(PermissionError,match='FRESH_BACKEND'):restore(backend,value)


def test_original_baseline_cannot_inherit_a_changed_controller_margin(selected,monkeypatch):
    backend,value,_,_=configure(selected,monkeypatch,'baseline_control')
    backend.environment.set_h1_ics_clearance_margin(.2)
    monkeypatch.setattr(backend.environment,'set_h1_ics_clearance_margin',lambda value:None)
    before=torch.get_rng_state().clone()
    with pytest.raises(PermissionError,match='BASELINE_METHOD_READBACK'):restore(backend,value)
    assert backend.trainer is None and backend._r6_load_failed and torch.equal(before,torch.get_rng_state())
