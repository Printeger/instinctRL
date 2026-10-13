"""R5 preregistered reward, group identity and before-write boundary regressions."""
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from instinctlab.task.rewards_h1 import compute_reward_h1
from instinctlab.control.governor import decode_governor, GovernorConfig
from training_workloads.unified_h1.authorization import _parameters
from training_workloads.unified_h1.pilot_contract import pilot_actor_learning_rate, pilot_residual_weight, pilot_method_hashes
from training_workloads.scale_003 import persistence

ROOT = Path(__file__).resolve().parents[2]
BASE = 'ef68ebb5682b85aa83044fdea1c13b7d0edbf325'


def inputs(command=None):
    n = 8
    lidar = torch.zeros(n, 12, 59, 360)
    lidar[:, 0::3] = 40.; lidar[:, 1::3] = 1.; lidar[:, 2::3] = 1.
    vector = torch.randn(n, 3) * .1
    col = torch.ones(n, 1)
    return dict(lidar_grid=lidar, command_b=torch.zeros(n,3) if command is None else command,
        v_gov_b=vector, v_final_b=vector*.8, previous_issued_v_final_b=vector*.9,
        actual_velocity_b=vector*.7, anchor_loss=col*.025, anchor_active=col,
        anchor_usable_fraction=col, ics_beta=col*.8, exact_clearance_m=col*3,
        collision=torch.zeros(n,1,dtype=torch.bool), below_height=torch.zeros(n,1,dtype=torch.bool),
        above_height=torch.zeros(n,1,dtype=torch.bool))


def test_w0_matches_real_frozen_baseline_every_component_and_control():
    source = subprocess.check_output(['git','show',f'{BASE}:isaaclab-training/instinctlab/task/rewards_h1.py'],cwd=ROOT,text=True)
    module = ModuleType('_r5_frozen_reward_reference'); sys.modules[module.__name__]=module
    exec(compile(source,'frozen_reward_reference','exec'),module.__dict__)
    torch.manual_seed(917)
    data=inputs(); data['command_b'][4:] = torch.tensor([.2, -.1, .05])
    old=module.compute_reward_h1(**data)
    new=compute_reward_h1(**data,v_corr_b=data['v_gov_b'],residual_neutrality_weight=0.)
    assert list(old.components)==list(new.components)
    for key in old.components: assert torch.equal(old.components[key],new.components[key])
    # Governor code is unmodified; reward cannot mutate the actual control inputs.
    before={k:v.clone() for k,v in data.items()}
    compute_reward_h1(**data,v_corr_b=data['v_gov_b'],residual_neutrality_weight=.05)
    for key in before: assert torch.equal(before[key],data[key])


def test_three_dimensional_signed_residuals_zero_boundary_and_dynamic_dependency():
    command=torch.zeros(8,3); command[6,0]=.001; command[7,0]=.001001
    data=inputs(command); corr=torch.tensor([[.1,0,0],[-.1,0,0],[0,.1,0],[0,-.1,0],[0,0,.1],[0,0,-.1],[.1,.1,.1],[.1,.1,.1]],requires_grad=True)
    result=compute_reward_h1(**data,v_corr_b=corr,residual_neutrality_weight=.05)
    torch.testing.assert_close(result.components['reward_zero_extra'].flatten(),torch.tensor([-.05]*6+[-.15,0.]))
    result.components['reward_zero_extra'].sum().backward()
    torch.testing.assert_close(corr.grad[:6],-10*corr.detach()[:6])
    assert torch.equal(corr.grad[7],torch.zeros(3))
    base=compute_reward_h1(**data)
    torch.testing.assert_close(result.total-base.total,result.components['reward_zero_extra'],atol=1e-6,rtol=1e-5)
    doubled=compute_reward_h1(**data,v_corr_b=corr.detach()*2,residual_neutrality_weight=.05)
    torch.testing.assert_close(doubled.components['reward_zero_extra'],result.components['reward_zero_extra']*4)


@pytest.mark.parametrize('weight',[True,-.05,float('nan'),float('inf')])
def test_bad_weight_fails_before_reward(weight):
    with pytest.raises(ValueError): compute_reward_h1(**inputs(),residual_neutrality_weight=weight)


def test_missing_nonfinite_wrong_shape_correction_rejected():
    data=inputs()
    with pytest.raises(ValueError): compute_reward_h1(**data,residual_neutrality_weight=.05)
    for corr in [torch.zeros(8,4),torch.full((8,3),float('nan')),torch.zeros(8,3,dtype=torch.float64)]:
        with pytest.raises((ValueError,TypeError)):
            compute_reward_h1(**data,v_corr_b=corr,residual_neutrality_weight=.05)


def test_new_groups_bind_coefficient_without_overwriting_legacy():
    assert [pilot_actor_learning_rate(g) for g in ['P0','PLR','C0','C1']]==[3e-4,1e-4,1e-4,1e-4]
    config={'method':{'hashes':{'method':'a'*64}}}
    assert pilot_method_hashes(config,'P0')==config['method']['hashes']
    assert pilot_method_hashes(config,'C0')!=pilot_method_hashes(config,'C1')
    for g in ['C0','C1']:
        params={'group':g,'residual_neutrality_weight':pilot_residual_weight(g),'output_cap_bytes':16*1024**3}
        assert _parameters('pilot-training',params)==params
        for k in ['residual_neutrality_weight','output_cap_bytes']:
            wrong=dict(params); wrong[k]+=1
            with pytest.raises(PermissionError): _parameters('pilot-training',wrong)
        with pytest.raises(PermissionError): _parameters('pilot-training',{'group':g})
    assert _parameters('pilot-training',{'group':'PLR'})=={'group':'PLR'}


def test_cpu_cold_start_same_seed_actual_lr_coefficient_and_checkpoint_identity(tmp_path):
    from training_workloads.unified_h1.backend import UnifiedH1LiveBackend
    states=[]
    config={'method':{'hashes':{'method':'a'*64}},'resolved_identity':dict(product_commit='b'*40,product_tree='c'*40,workload_commit='b'*40,workload_tree='c'*40,config_sha256='d'*64)}
    for g in ['C0','C1']:
        out=tmp_path/g/'results/pilot-training';out.mkdir(parents=True)
        env=NS(prepare_h1_step=lambda **k:None,step=lambda a:None,device='cpu',set_h1_evaluation_case=lambda c:None,set_h1_initial_command=lambda c,s:None,reset=lambda seed: ({},{}))
        env.set_h1_residual_neutrality_weight=lambda w:setattr(env,'_h1_residual_neutrality_weight',w)
        auth={'mode':'pilot-training','parameters':{'group':g,'residual_neutrality_weight':pilot_residual_weight(g)},'authorization_sha256':'e'*64,
              'budgets':dict(wall_seconds=3600,explicit_resets=1,training_transitions=32768,automatic_resets=32768,optimizer_updates=64,checkpoint_writes=7)}
        backend=UnifiedH1LiveBackend(authorization=auth,attempt=tmp_path/g,app=None,environment=env,resource_guard=NS(check=lambda **k:None))
        with patch('training_workloads.unified_h1.backend.load_config',return_value=config):
            backend.cold_start(pilot_group=g); backend.save_checkpoint('init0000')
        cp=torch.load(out/'checkpoints/init0000.pt',map_location='cpu',weights_only=True)
        assert cp['method_hashes']==pilot_method_hashes(config,g)
        assert cp['source_identity']==config['resolved_identity']
        assert cp['ppo_config']['actor_learning_rate']==1e-4 and cp['ppo_config']['critic_learning_rate']==3e-4
        assert env._h1_residual_neutrality_weight==pilot_residual_weight(g)
        assert cp['counters']['actor_optimizer']==cp['counters']['critic_optimizer']==0
        states.append(cp)
    for key in ['actor','critic']:
        for name in states[0][key]: assert torch.equal(states[0][key][name],states[1][key][name])
    assert torch.equal(states[0]['rng']['torch_cpu'],states[1]['rng']['torch_cpu'])


def test_output_quota_stops_before_overflow_and_never_refunds(tmp_path):
    with patch.object(persistence,'_output_quota',None):
        quota=persistence.install_output_write_quota(tmp_path,64)
        path=tmp_path/'raw.bin'
        with path.open('wb') as stream:
            bounded=persistence.guard_stream(stream)
            bounded.write(b'a'*40)
            with pytest.raises(persistence.PersistenceError): bounded.write(b'b'*25)
            bounded.seek(0); bounded.write(b'x'*20)
        assert path.stat().st_size==40 and quota.used==60
        with pytest.raises(persistence.PersistenceError): persistence.reserve_output_bytes(tmp_path/'append.log',5)
        assert quota.used==60
