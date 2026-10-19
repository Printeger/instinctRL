import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from instinctlab.observations.actor import M3_MAX_RANGE_M
from instinctlab.control.ics import ICSConfig,attenuate_command,mounted_mid360_rays_body
from instinctlab.control.pipeline import ControlCoreConfig,ControlCorePipeline
from training_workloads.unified_h1 import w_closure as w
from training_workloads.unified_h1 import formal_segment as old
from training_workloads.unified_h1.schedule import command_at,reset_local_positions

ROOT=Path(__file__).resolve().parents[2]

def fixture_profile(tmp_path):
    def pin(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    master=pin('master.json',dict(id=w.MASTER_ID,status='REGISTERED_USER_DELEGATION'))
    ck=pin('checkpoint.pt',{})
    contract=dict(path=str(w.CONTRACT_PATH),sha256=w.CONTRACT_SHA256)
    mean=ROOT/'coordination/contracts/R5_ZERO_COMMAND_MEAN_PRIOR_V1.md'
    p=dict(experiment_id='R5-W-L1',group='R1',training_seed=20260908,evaluation_seed=20260909,
        start_update=768,target_update=1024,checkpoint_updates=[896,1024],evaluation_updates=[896,1024],
        command_profile='mixed-nearwall',residual_neutrality_weight=.05,anchor_weight=4.,actor_lr=1e-4,
        critic_lr=3e-4,mean_prior_weight=10.,mean_prior_contract=dict(path=str(mean),sha256=hashlib.sha256(mean.read_bytes()).hexdigest()),
        output_cap_bytes=8*1024**3,master_authority=master,w_closure_contract=contract,
        checkpoint_source_identity={k:'a'*(64 if k=='config_sha256' else 40) for k in ('product_commit','product_tree','workload_commit','workload_tree','config_sha256')},
        checkpoint_compatibility=pin('compat.json',{}),initialization_checkpoint=dict(**ck,update=768),
        origin_method_hashes={'method':'exact-origin'},evaluation_checkpoint_origin='CANDIDATE',ics_clearance_margin_m=.15)
    p['experiment_card']=pin('card.json',dict(experiment_id=p['experiment_id'],profiles={'R1':deepcopy(p)}))
    return p


def test_new_phase_dispatch_and_exact_budget(tmp_path):
    p=fixture_profile(tmp_path)
    assert old.validate_parameters(w.TRAIN,p)==p
    b=old.required_budget(w.TRAIN,p)
    assert (b['training_transitions'],b['optimizer_updates'],b['checkpoint_writes'])==(131072,256,2)
    e=deepcopy(p);e['checkpoints']=[dict(p['initialization_checkpoint'],update=u) for u in p['evaluation_updates']]
    assert old.required_budget(w.EVAL,e)['nontraining_transitions']==65920


@pytest.mark.parametrize('key,value',[('mean_prior_weight',0.),('ics_clearance_margin_m',0.),('evaluation_seed',123),('start_update',True),('target_update',4096),('command_profile','pure-hover'),('checkpoint_updates',[896]),('group','R0'),('evaluation_checkpoint_origin','BASELINE')])
def test_unregistered_variants_rejected(tmp_path,key,value):
    p=fixture_profile(tmp_path);p[key]=value
    with pytest.raises(PermissionError):old.validate_parameters(w.TRAIN,p)


def test_missing_new_marker_cannot_extend_legacy(tmp_path):
    p=fixture_profile(tmp_path);del p['w_closure_contract']
    with pytest.raises(PermissionError):old.validate_parameters(w.TRAIN,p)


def test_origin_and_master_bytes_are_rechecked(tmp_path):
    p=fixture_profile(tmp_path);Path(p['master_authority']['path']).write_text('{}')
    with pytest.raises(PermissionError):w.validate_parameters(w.TRAIN,p)


def test_nearwall_center_distribution_and_all_commands():
    positions,labels=reset_local_positions(512*16)
    assert [labels.count(k) for k in ('center','near_x_wall','near_y_wall')]==[24,4,4]
    assert torch.all(positions[[6,14,22,30],0]==4.6)
    for lane,normal in [(6,0),(7,1)]:
        cmds=torch.stack([command_at(512*16+i*128)[lane] for i in range(16)])
        assert (cmds.norm(dim=-1)==0).any() and (cmds[:,normal]>.1).any() and (cmds[:,normal]<-.1).any()
        assert ((cmds[:,normal]==0)&(cmds.norm(dim=-1)>.1)).any()


def test_margin_preserved_through_reset_without_changing_other_config():
    # Execute actual production methods against a no-physics base class.
    source=ast.parse((ROOT/'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py').read_text())
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='UnifiedH1DirectEnv')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('set_h1_ics_clearance_margin','_reset_idx_impl')]
    base=type('Base',(),{'_reset_idx':lambda self,ids:setattr(self,'_control_pipelines',[ControlCorePipeline() for _ in range(32)])})
    module=ast.Module(body=[ast.ClassDef(name='Probe',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    ast.fix_missing_locations(module);ns={'Base':base,'torch':torch,'Sequence':list}
    exec(compile(module,'actual-env-methods','exec'),ns)
    env=ns['Probe']();env._control_pipelines=[ControlCorePipeline() for _ in range(32)]
    env._validate_reset_indices=lambda ids:ids;env._h1_evaluation_case=None;env._h1_global_policy_step=0
    env.set_h1_ics_clearance_margin(.15);env._reset_idx_impl(torch.tensor([0,7]))
    assert all(p.config.ics==ICSConfig(clearance_margin_m=.15) for p in env._control_pipelines)
    assert all(p.config.governor==ControlCoreConfig().governor for p in env._control_pipelines)
    with pytest.raises(ValueError):env.set_h1_ics_clearance_margin(-.1)


def test_margin_earlier_braking_and_away_motion():
    rays=mounted_mid360_rays_body(); grid=torch.zeros(1,12,59,360)
    grid[:,0::3]=M3_MAX_RANGE_M;grid[:,1::3]=0.;grid[:,2::3]=0.
    # One actual mounted beam, commands exactly toward or away from it.
    i=int(torch.argmax(rays[:,0]));e,a=divmod(i,360)
    grid[:,0::3,e,a]=.94;grid[:,1::3,e,a]=1.;grid[:,2::3,e,a]=1.
    toward=rays[i][None]*.2
    original=attenuate_command(grid,rays,toward,.1,ICSConfig())
    candidate=attenuate_command(grid,rays,toward,.1,ICSConfig(clearance_margin_m=.15))
    away=attenuate_command(grid,rays,-toward,.1,ICSConfig(clearance_margin_m=.15))
    assert original.metrics['beta'].item()==1 and candidate.metrics['beta'].item()==0
    assert away.metrics['beta'].item()==1
    assert torch.equal(away.v_final_b,-toward)

@pytest.mark.parametrize('explicit', [False,True])
def test_first_episode_distribution_is_explicit_and_legacy_unchanged(tmp_path,explicit):
    from training_workloads.unified_h1.logging import LearningLog
    _,labels=reset_local_positions(8192)
    log=LearningLog(tmp_path,initial_start_types=labels if explicit else None)
    root=torch.zeros(32,13);root[:,2]=2
    t=SimpleNamespace(reward=torch.zeros(32),reset_boundary=torch.zeros(32,dtype=torch.bool))
    evidence={'_h1_driver':{'global_policy_step':8192},'pre_reset':{'root_state_w':root,
        'task':{'actual_velocity_b':torch.zeros(32,3),'exact_clearance_m':torch.ones(32,1)}}}
    log._record_blocks_and_episodes(t,evidence,torch.zeros(32,3),update=513,step=0)
    assert log._episode_start_types==list(labels) if explicit else log._episode_start_types==['center']*32


def test_initial_distribution_rejects_missing_lanes(tmp_path):
    from training_workloads.unified_h1.logging import LearningLog
    with pytest.raises(ValueError):LearningLog(tmp_path,initial_start_types=('center',))
