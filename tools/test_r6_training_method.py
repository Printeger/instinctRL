"""Strict loader CPU tests: original CUDA rejection and explicit CPU fixture."""
from dataclasses import replace,asdict
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import pytest
import torch
from tools.test_r6_training_plan import case,method_fixture
from tools.test_r6_checkpoint_branch import origin
from tools.test_r6_training_rng import preserve_global_rng
from training_workloads.unified_h1 import robustness_training_method as m
from training_workloads.unified_h1 import robustness_training_plan as plan
from training_workloads.unified_h1.backend import _state_fingerprint
from instinctlab.training.checkpoint_h1 import save_checkpoint_h1,_rng_payload


class Env:
    device='cpu'
    def __init__(self):
        self._control_pipelines=[SimpleNamespace(config=SimpleNamespace(ics=SimpleNamespace(clearance_margin_m=0.)))]
    def set_h1_research_profile(self,*,pure_hover,anchor_weight):
        self._h1_pure_hover_training=pure_hover;self._h1_anchor_weight=anchor_weight
    def set_h1_residual_neutrality_weight(self,v):self._h1_residual_neutrality_weight=v
    def set_h1_ics_clearance_margin(self,v):
        self._h1_ics_clearance_margin_m=v
        for p in self._control_pipelines:p.config.ics.clearance_margin_m=v


def backend(case):
    return SimpleNamespace(environment=Env(),authorization=dict(mode='research-r6-training',
        parameters={'r6_training_plan':case.plan},budgets=plan.required_budget(case.plan)),
        trainer=None,driver=None,loop=None,journal=None,curriculum_generator=torch.Generator().manual_seed(43))


@pytest.mark.parametrize('fault',['missing','unknown','nan','boolean','clock','seed','weight'])
def test_method_schema_rejects_implicit_and_invalid_settings(fault):
    v=method_fixture()
    if fault=='missing':del v['ppo']['gamma']
    elif fault=='unknown':v['ppo']['new_algorithm']=True
    elif fault=='nan':v['ppo']['gamma']=float('nan')
    elif fault=='boolean':v['anchor_weight']=True
    elif fault=='clock':v['ppo']['epoch_count']=2
    elif fault=='seed':v['ppo']['permutation_seed']=11
    else:v['ics_clearance_margin_m']=-1
    with pytest.raises(PermissionError):m.validate(v)


def test_method_changes_only_declared_settings_and_preserves_weights_adam(origin):
    t,g,meta,payload=origin;v=method_fixture()
    env=Env();record=m.apply_method(t,env,v,resume=False,training_seed=11)
    assert record['unchanged_weights_adam_moments_and_counters']
    assert t.config.actor_learning_rate==.0003 and t.actor_optimizer.param_groups[0]['lr']==.0003
    assert record['before_ppo']['actor_learning_rate']==.0001
    assert env._h1_ics_clearance_margin_m==.15 and not env._h1_pure_hover_training
    state=torch.zeros(4,52)
    output=SimpleNamespace(mean_action=torch.tensor([[.5,.6,.5,.5]]*4))
    assert float(t.actor_auxiliary({'state_vec':state},output))==pytest.approx(10.)
    state[:,45]=.2
    assert t.actor_auxiliary({'state_vec':state},output)==0


def test_actual_original_cuda_checkpoint_is_rejected_without_rng_mutation(case):
    b=json.loads(Path('docs/experiments/r6-robustness-auto/BASELINE_BINDINGS.json').read_text())
    case.plan['initialization']=dict(kind='COMMON_U640_NEW_STREAM',checkpoint=b['selected_model'],
        source_identity=b['source_identity'],method_hashes=b['method_hashes'])
    obj=backend(case);before=torch.get_rng_state().clone();curr=obj.curriculum_generator.get_state().clone()
    with pytest.raises(ValueError,match='CPU trainer cannot restore CUDA'):
        m.restore_and_bind(obj,plan=case.plan,execution_identity=case.identity)
    assert obj.trainer is None and obj._r6_load_failed
    assert torch.equal(before,torch.get_rng_state()) and torch.equal(curr,obj.curriculum_generator.get_state())
    with pytest.raises(PermissionError,match='FRESH_BACKEND'):
        m.restore_and_bind(obj,plan=case.plan,execution_identity=case.identity)


def test_exact_cpu_resume_restores_weights_adam_rng_then_binds(case,origin,tmp_path):
    # Synthetic u641 counters, explicitly not an actual trained R6 checkpoint.
    t,g,meta,payload=origin;t.update_counter=641
    for optimizer in (t.actor_optimizer,t.critic_optimizer):
        for state in optimizer.state.values():state['step'].fill_(641*8)
    m.apply_method(t,Env(),method_fixture(),resume=False,training_seed=11)
    t.config=replace(t.config,permutation_seed=11)
    hashes={'r6_recipe':case.plan['recipe']['sha256'],'r6_method':case.recipe['method']['sha256']}
    ck=tmp_path/'CPU_SYNTHETIC_U641.pt'
    save_checkpoint_h1(ck,t,method_hashes=hashes,source_identity=case.identity,transition_counter=641*512,
        global_policy_step=641*16,curriculum=dict(version='curriculum-h1-r6-v1',completed_policy_steps=641*16,
        recipe_sha256=case.plan['recipe']['sha256'],origin_checkpoint_sha256=plan.BASELINE_SHA,branch_seed=11),
        training_seed=11,reset_number=meta['reset_number'],curriculum_generator=g)
    case.plan.update(start_update=641,initialization=dict(kind='RESUME_R6_STREAM',checkpoint=dict(path=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest()),
        source_identity=case.identity,method_hashes=hashes))
    obj=backend(case);result=m.restore_and_bind(obj,plan=case.plan,execution_identity=case.identity)
    assert result['metadata']['counters']['update']==641 and obj._r6_method_binding['strict_resume']
    assert _state_fingerprint([obj.trainer.actor.state_dict(),obj.trainer.critic.state_dict(),obj.trainer.actor_optimizer.state_dict()])==_state_fingerprint([t.actor.state_dict(),t.critic.state_dict(),t.actor_optimizer.state_dict()])
    saved=torch.load(ck,map_location='cpu',weights_only=True)
    assert _state_fingerprint(_rng_payload(obj.curriculum_generator,obj.trainer))==_state_fingerprint(saved['rng'])
    assert obj.trainer._r6_bound_method_sha256==m.object_sha(method_fixture())


def test_readback_failure_is_not_accepted_as_bound_method(origin):
    t,g,meta,payload=origin;e=Env();e._control_pipelines=[]
    with pytest.raises(PermissionError,match='ICS_READBACK'):m.apply_method(t,e,method_fixture(),resume=False,training_seed=11)
    assert not hasattr(t,'_r6_bound_method_sha256')


@pytest.mark.parametrize('fault',['reward','ics','lr','auxiliary'])
def test_runtime_drift_rejected_before_execution(origin,fault):
    t,g,meta,payload=origin;env=Env();v=method_fixture()
    m.apply_method(t,env,v,resume=False,training_seed=11);m.verify_applied(t,env,v)
    if fault=='reward':env._h1_anchor_weight+=1
    elif fault=='ics':env._control_pipelines[0].config.ics.clearance_margin_m=0
    elif fault=='lr':t.actor_optimizer.param_groups[0]['lr']=1
    else:t.actor_auxiliary=lambda *a:0
    with pytest.raises(PermissionError,match='DRIFT'):m.verify_applied(t,env,v)
