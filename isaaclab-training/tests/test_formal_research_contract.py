"""Actual pure-hover boundary, exact profile and safe seed persistence checks."""
import copy,json,hashlib,subprocess,sys
from pathlib import Path
from types import SimpleNamespace as NS,ModuleType
from unittest.mock import patch
import pytest
import torch
from training_workloads.unified_h1.schedule import ScheduleClock
from training_workloads.unified_h1.research_contract import validate_parameters,required_budget,method_hashes
from training_workloads.unified_h1.backend import UnifiedH1LiveBackend
from instinctlab.training.checkpoint_h1 import _validate_clock_and_curriculum,CheckpointError
from instinctlab.task.rewards_h1 import compute_reward_h1
def inputs():
 n=8;lidar=torch.zeros(n,12,59,360);lidar[:,0::3]=40.;lidar[:,1::3]=1.;lidar[:,2::3]=1.
 v=torch.randn(n,3)*.1;c=torch.ones(n,1);f=torch.zeros(n,1,dtype=torch.bool)
 return dict(lidar_grid=lidar,command_b=torch.zeros(n,3),v_gov_b=v,v_final_b=v*.8,previous_issued_v_final_b=v*.9,actual_velocity_b=v*.7,anchor_loss=c*.025,anchor_active=c,anchor_usable_fraction=c,ics_beta=c*.8,exact_clearance_m=c*3,collision=f,below_height=f,above_height=f)
ROOT=Path(__file__).resolve().parents[2]
def pin(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def profile(tmp_path,group='R1'):
 master=tmp_path/'master.json';master.write_text(json.dumps({'id':'R5-FORMAL-READINESS-AUTO-01','status':'REGISTERED_USER_DELEGATION'}))
 p=dict(experiment_id='R5-H128-DEV-01',group=group,training_seed=20260908,evaluation_seed=20260909,max_updates=128,checkpoint_updates=[1,2,8,16,32,64,96,128],evaluation_updates=[96,128],command_profile='pure-hover',residual_neutrality_weight=0 if group=='R0' else .05,anchor_weight=4,actor_lr=1e-4,critic_lr=3e-4,output_cap_bytes=8*1024**3,master_authority=pin(master))
 card=tmp_path/'card.json';card.write_text(json.dumps({'experiment_id':p['experiment_id'],'profiles':{group:copy.deepcopy(p)}}));p['experiment_card']=pin(card);return p

def test_pure_hover_crosses_old_mixed_boundary_without_hidden_clock_reset():
 pure=ScheduleClock(1023,pure_hover=True);old=ScheduleClock(1023)
 assert not old.plan().next_command_b.eq(0).all()
 for _ in range(3074):
  plan=pure.plan();assert plan.command_b.eq(0).all() and plan.next_command_b.eq(0).all() and plan.zero_mask.all() and plan.block is None
  pure.complete_step()
 assert pure.completed_policy_steps==4097
 saved=pure.state_dict();restored=ScheduleClock();restored.load_state_dict(saved);assert restored.pure_hover and restored.completed_policy_steps==4097 and restored.plan().command_b.eq(0).all()
 assert ScheduleClock(1024).plan().nonzero_mask.sum()==16

@pytest.mark.parametrize('field,value',[('group','C1'),('max_updates',1024),('command_profile','mixed'),('evaluation_updates',[64,96]),('training_seed',True),('anchor_weight',float('nan')),('actor_lr',3e-4)])
def test_profile_tamper_rejected_before_any_runtime(tmp_path,field,value):
 p=profile(tmp_path);p[field]=value
 with pytest.raises(PermissionError):validate_parameters('research-training',p)

def test_card_and_master_binding_drift_and_exact_budget(tmp_path):
 p=profile(tmp_path);budget=required_budget('research-training',p)
 assert budget['training_transitions']==65536 and budget['optimizer_updates']==128 and budget['checkpoint_writes']==9
 p['anchor_weight']=8
 with pytest.raises(PermissionError,match='card/profile'):validate_parameters('research-training',p)
 p=profile(tmp_path);Path(p['master_authority']['path']).write_text('{}')
 with pytest.raises(PermissionError,match='drift'):validate_parameters('research-training',p)


def transition_profile(tmp_path):
 """Parameter/file tests only; dummy tensors do not authorize a real entry."""
 p=profile(tmp_path);p['evaluation_updates']=[128]
 output=tmp_path/'actual-training/results/research-training';(output/'checkpoints').mkdir(parents=True);(output/'pilot_diagnostic').mkdir()
 for name in ['checkpoints/u1.pt','checkpoints/u128.pt','pilot_diagnostic/update_1_full_input.pt','pilot_diagnostic/first_update_probe.pt']:(output/name).write_bytes(b'opaque CPU parameter fixture only')
 compatibility=tmp_path/'compatibility.json';compatibility.write_text('{}')
 p.update(checkpoints=[{**pin(output/'checkpoints/u128.pt'),'update':128}],transition_protocol='zero-directions-reverse-zero112-v1',integration_contract=pin(ROOT/'coordination/contracts/R5_FORMAL_INTEGRATION_DIAGNOSTIC_V1.md'),checkpoint_source_identity={k:('1'*40 if k!='config_sha256' else '2'*64) for k in ['product_commit','product_tree','workload_commit','workload_tree','config_sha256']},checkpoint_compatibility=pin(compatibility),recalculation={'checkpoint':{**pin(output/'checkpoints/u1.pt'),'update':1},'input':pin(output/'pilot_diagnostic/update_1_full_input.pt'),'probe':pin(output/'pilot_diagnostic/first_update_probe.pt')})
 refresh_transition_card(p)
 return p


def refresh_transition_card(p):
 card=Path(p['experiment_card']['path']);card.write_text(json.dumps({'experiment_id':p['experiment_id'],'profiles':{p['group']:{k:v for k,v in p.items() if k not in ['checkpoints','experiment_card']}}}));p['experiment_card']=pin(card)


def test_finite_transition_budget_is_nontraining_only(tmp_path):
 p=transition_profile(tmp_path);b=required_budget('research-transition',p)
 assert (b['nontraining_transitions'],b['training_transitions'],b['optimizer_updates'],b['checkpoint_writes'])==(3584,0,0,0)
 assert (b['explicit_resets'],b['wall_seconds'],b['initialization_ticks'])==(1,900,10)
 # Transition parameters cannot masquerade as a normal training authority.
 with pytest.raises(PermissionError,match='fields mismatch'):validate_parameters('research-training',p)


def test_transition_recalculation_cannot_cross_actual_training_entry(tmp_path):
 p=transition_profile(tmp_path);other=tmp_path/'other-u1.pt';other.write_bytes(b'opaque CPU parameter fixture only');p['recalculation']['checkpoint'].update(pin(other));refresh_transition_card(p)
 with pytest.raises(PermissionError,match='same actualtraining entry'):validate_parameters('research-transition',p)


def test_transition_protocol_is_not_arbitrary_despite_card_edit(tmp_path):
 p=transition_profile(tmp_path);p['transition_protocol']='modified-temporal-sequence';refresh_transition_card(p)
 with pytest.raises(PermissionError,match='not registered'):validate_parameters('research-transition',p)


def test_transition_requires_exact_five_field_source_identity(tmp_path):
 p=transition_profile(tmp_path);p['checkpoint_source_identity'].pop('config_sha256');refresh_transition_card(p)
 with pytest.raises(PermissionError,match='five source fields'):validate_parameters('research-transition',p)

def test_checkpoint_seed_expansion_is_explicit_profile_only():
 args=dict(update=128,transition=65536,policy_step=2048,training_seed=20261017)
 with pytest.raises(CheckpointError):_validate_clock_and_curriculum(**args,curriculum={'version':'curriculum-h1-v1','completed_policy_steps':2048})
 c={'version':'curriculum-h1-research-pure-hover-v1','completed_policy_steps':2048}
 assert _validate_clock_and_curriculum(**args,curriculum=c)==c
 with pytest.raises(CheckpointError):_validate_clock_and_curriculum(**dict(args,policy_step=2047),curriculum=c)


def test_default_anchor4_matches_frozen_actual_C1_components():
 src=subprocess.check_output(['git','show','d257f9ca8ec8639627d3a025c9352b9d29883d90:isaaclab-training/instinctlab/task/rewards_h1.py'],cwd=ROOT,text=True)
 m=ModuleType('_frozen_actual_C1_reward');sys.modules[m.__name__]=m;exec(compile(src,'frozen_C1','exec'),m.__dict__)
 torch.manual_seed(17);data=inputs();data['command_b'][4:,0]=.2;kwargs=dict(v_corr_b=data['v_gov_b'],residual_neutrality_weight=.05)
 old=m.compute_reward_h1(**data,**kwargs);new=compute_reward_h1(**data,**kwargs)
 for k in old.components:assert torch.equal(old.components[k],new.components[k])
 doubled=compute_reward_h1(**data,**kwargs,anchor_weight=8)
 for k in new.components:
  if k not in ['reward_anchor','reward_total']:assert torch.equal(doubled.components[k],new.components[k])
 assert torch.equal(doubled.components['reward_anchor'],new.components['reward_anchor']*2)
 assert doubled.components['reward_anchor'][4:].eq(0).all()


def test_actual_research_coldstart_profile_same_initialization_and_seed_metadata(tmp_path):
 states=[];cfg={'method':{'hashes':{'method':'a'*64}},'resolved_identity':dict(product_commit='b'*40,product_tree='c'*40,workload_commit='b'*40,workload_tree='c'*40,config_sha256='d'*64)}
 for group in ['R0','R1']:
  folder=tmp_path/group;folder.mkdir();p=profile(folder,group);out=folder/'results/research-training';out.mkdir(parents=True)
  env=NS(prepare_h1_step=lambda **k:None,step=lambda a:None,device='cpu',set_h1_evaluation_case=lambda c:None,set_h1_initial_command=lambda c,s:None,reset=lambda seed: ({},{}))
  env.set_h1_residual_neutrality_weight=lambda w:setattr(env,'_h1_residual_neutrality_weight',w)
  def set_profile(*,pure_hover,anchor_weight):env._h1_pure_hover_training=pure_hover;env._h1_anchor_weight=anchor_weight
  env.set_h1_research_profile=set_profile
  auth={'mode':'research-training','parameters':p,'authorization_sha256':'e'*64,'budgets':required_budget('research-training',p)}
  backend=UnifiedH1LiveBackend(authorization=auth,attempt=folder,app=None,environment=env,resource_guard=NS(check=lambda **k:None))
  with patch('training_workloads.unified_h1.backend.load_config',return_value=cfg):backend.cold_start();backend.save_checkpoint('init0000')
  cp=torch.load(out/'checkpoints/init0000.pt',weights_only=True);assert cp['training_seed']==p['training_seed'] and cp['curriculum']['version']=='curriculum-h1-research-pure-hover-v1'
  assert cp['method_hashes']==method_hashes(cfg,p) and cp['ppo_config']['actor_learning_rate']==1e-4 and env._h1_pure_hover_training and backend.driver.clock.pure_hover
  assert backend.loop.diagnostic.enabled
  states.append(cp)
 for k in ['actor','critic']:
  for name in states[0][k]:assert torch.equal(states[0][k][name],states[1][k][name])
 assert torch.equal(states[0]['rng']['torch_cpu'],states[1]['rng']['torch_cpu'])


def test_actual_reset_implementation_keeps_hover_geometry_after_1024_without_changing_eval():
    import ast
    source=(ROOT/'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py').read_text()
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='UnifiedH1DirectEnv')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_reset_idx_impl')
    body=ast.unparse(method)
    ns={'torch':torch}
    exec('from __future__ import annotations\nclass Base:\n def _reset_idx(self,indices): self.base_reset_ids=indices.clone()\nclass Tested(Base):\n'+''.join(' '+line+'\n' for line in body.splitlines()),ns)
    e=ns['Tested']();e._validate_reset_indices=lambda ids:torch.as_tensor(ids);e._h1_evaluation_case=None;e._h1_global_policy_step=2048;e._h1_pure_hover_training=True
    # No robot exists in this stub: any unintended near-wall reset would access it.
    e._reset_idx_impl([6,7]);assert torch.equal(e.base_reset_ids,torch.tensor([6,7]))
    e._h1_pure_hover_training=False
    with pytest.raises(AttributeError,match='_robot'):e._reset_idx_impl([6,7])
