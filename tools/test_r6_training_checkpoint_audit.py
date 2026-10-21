"""Real CPU networks/Adam state serialization; synthetic completed-update moments."""
import copy
import random
import numpy as np
import pytest
import torch
from tools.r6_training_checkpoint_audit import audit
from instinctlab.training.networks import HistoryActor,NetworkConfig
from instinctlab.training.networks_h1 import PrivilegedCriticH1
from instinctlab.training.ppo import PPOTrainer,PPOConfig
from instinctlab.training.checkpoint_h1 import CHECKPOINT_H1_KEYS,CHECKPOINT_H1_SCHEMA,CHECKPOINT_H1_VERSION,_rng_payload


def payload(update=642,ppo=None):
    before=torch.get_rng_state().clone()
    try:
        config=NetworkConfig(lidar_channels=(1,1,1),state_hidden=4,actor_hidden=4,critic_hidden=(4,4))
        trainer=PPOTrainer(HistoryActor(config),PrivilegedCriticH1(config),ppo or PPOConfig(minibatch_size=64))
        for optimizer in (trainer.actor_optimizer,trainer.critic_optimizer):
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    optimizer.state[parameter]=dict(step=torch.tensor(float(update*8)),exp_avg=torch.full_like(parameter,.1),exp_avg_sq=torch.full_like(parameter,.2))
        return dict(version=CHECKPOINT_H1_VERSION,schema=CHECKPOINT_H1_SCHEMA,network_config=config.manifest(),ppo_config=trainer.config.manifest(),
            method_hashes={'fixture':'only'},source_identity={'fixture':'only'},actor=copy.deepcopy(trainer.actor.state_dict()),critic=copy.deepcopy(trainer.critic.state_dict()),
            actor_optimizer=copy.deepcopy(trainer.actor_optimizer.state_dict()),critic_optimizer=copy.deepcopy(trainer.critic_optimizer.state_dict()),
            counters=dict(update=update,transition=update*512,global_policy_step=update*16,actor_optimizer=update*8,critic_optimizer=update*8),
            curriculum=dict(version='curriculum-h1-v1',completed_policy_steps=update*16),training_seed=20260908,reset_number=3,rng=_rng_payload(torch.Generator().manual_seed(11),trainer))
    finally:torch.set_rng_state(before)


def test_cpu_state_load_preserves_rng_and_never_probes_cuda(monkeypatch):
    value=payload();before=torch.get_rng_state().clone();py=random.getstate();np_before=np.random.get_state()
    def forbidden(*args,**kwargs):raise AssertionError('CUDA forbidden')
    for name in ('is_available','device_count','get_rng_state_all','set_rng_state_all'):monkeypatch.setattr(torch.cuda,name,forbidden)
    result=audit(value)
    assert result['adam_state_loaded'] and result['actor_critic_state_loaded'] and result['adam_step']==642*8
    assert not result['strict_checkpoint_restore_performed'] and not result['optimizer_step_executed']
    assert torch.equal(before,torch.get_rng_state()) and random.getstate()==py
    after=np.random.get_state();assert after[0]==np_before[0] and np.array_equal(after[1],np_before[1]) and after[2:]==np_before[2:]


@pytest.mark.parametrize('fault',['missing','clock','fractional','negative_second','shape','dtype','nan','hyperparameter','critic','rng','cuda_schema'])
def test_bad_state_rejected_without_global_rng_change(fault):
    value=payload();assert audit(value)['adam_state_loaded'];state=value['actor_optimizer']['state'];key=next(iter(state));item=state[key]
    if fault=='missing':del state[key]
    if fault=='clock':item['step']-=1
    if fault=='fractional':item['step']+=.5
    if fault=='negative_second':item['exp_avg_sq'].fill_(-.1)
    if fault=='shape':item['exp_avg']=torch.ones(1)
    if fault=='dtype':item['exp_avg']=item['exp_avg'].double()
    if fault=='nan':item['exp_avg'].fill_(float('nan'))
    if fault=='hyperparameter':value['actor_optimizer']['param_groups'][0]['lr']=.7
    if fault=='critic':value['critic']={}
    if fault=='rng':value['rng']['curriculum']=torch.zeros(1,dtype=torch.uint8)
    if fault=='cuda_schema':value['rng']['torch_cuda']=[torch.zeros(1)]
    before=torch.get_rng_state().clone()
    with pytest.raises((ValueError,RuntimeError)):audit(value)
    assert torch.equal(before,torch.get_rng_state())


def test_cuda_state_is_not_claimed_semantically_verified():
    value=payload();value['rng']['torch_cuda']=[torch.zeros(16,dtype=torch.uint8)]
    result=audit(value)
    assert result['cuda_rng_state_count']==1 and not result['cuda_rng_semantics_verified'] and not result['rng_restored']
