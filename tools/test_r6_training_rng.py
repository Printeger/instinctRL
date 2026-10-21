"""CPU tensor forks, synthetic restored counters; no simulation or PPO update."""
import copy
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from instinctlab.training.ppo import PPOConfig,deterministic_minibatches
from training_workloads.unified_h1.backend import _state_fingerprint
from training_workloads.unified_h1.robustness_contract import BASELINE_SHA
from training_workloads.unified_h1.robustness_training_rng import branch_from_u640


@pytest.fixture(autouse=True)
def preserve_global_rng():
    cpu=torch.get_rng_state();py=random.getstate();numpy=np.random.get_state()
    yield
    torch.set_rng_state(cpu);random.setstate(py);np.random.set_state(numpy)


def restored():
    torch.random.default_generator.manual_seed(91)
    actor=torch.nn.Linear(2,4);critic=torch.nn.Linear(2,1)
    trainer=SimpleNamespace(actor=actor,critic=critic,update_counter=640,
        actor_optimizer=torch.optim.Adam(actor.parameters()),
        critic_optimizer=torch.optim.Adam(critic.parameters()),
        config=PPOConfig(minibatch_size=64,epoch_count=1))
    for optimizer in (trainer.actor_optimizer,trainer.critic_optimizer):
        for group in optimizer.param_groups:
            for parameter in group['params']:
                optimizer.state[parameter]=dict(step=torch.tensor(5120.),
                    exp_avg=torch.full_like(parameter,.2),exp_avg_sq=torch.full_like(parameter,.3))
    meta=dict(training_seed=20260908,counters=dict(update=640,transition=327680,
        global_policy_step=10240,actor_optimizer=5120,critic_optimizer=5120))
    return trainer,torch.Generator(device='cpu').manual_seed(7),meta


def weights(trainer):
    return _state_fingerprint([trainer.actor.state_dict(),trainer.critic.state_dict(),
        trainer.actor_optimizer.state_dict(),trainer.critic_optimizer.state_dict(),trainer.update_counter])


def branch(t,g,m,seed=2026102101):
    return branch_from_u640(t,g,seed=seed,origin_sha256=BASELINE_SHA,metadata=m)


def draw(t,g):
    samples=dict(
        action=torch.distributions.Beta(torch.full((32,4),2.),torch.full((32,4),3.)).sample(),
        curriculum=torch.rand(32,generator=g),python=random.random(),numpy=np.random.random(32),
        minibatches=deterministic_minibatches(512,t.config.minibatch_size,t.config.epoch_count,
                                            t.config.permutation_seed+t.update_counter))
    return {key:_state_fingerprint(value) for key,value in samples.items()}


def test_three_real_streams_differ_and_same_branch_reproduces():
    realizations=[];permutations=[]
    for seed in (2026102101,2026102201,2026102301):
        t,g,m=restored();before=weights(t);record=branch(t,g,m,seed)
        assert weights(t)==before
        assert record['branch_training_seed']==seed and record['strict_rng_continuation'] is False
        assert not torch.equal(record['origin_rng']['torch_cpu'],record['branch_rng']['torch_cpu'])
        permutations.append(_state_fingerprint(deterministic_minibatches(512,64,1,t.config.permutation_seed+640)))
        first=draw(t,g);t,g,m=restored();branch(t,g,m,seed);assert draw(t,g)==first
        realizations.append(first)
    assert len(set(permutations))==3
    for key in realizations[0]:assert len({row[key] for row in realizations})==3


def test_cpu_branch_never_touches_cuda_and_rejects_repeat():
    t,g,m=restored()
    with patch('torch.cuda.get_rng_state_all',side_effect=AssertionError('CPU CUDA query')), \
         patch('torch.cuda.manual_seed_all',side_effect=AssertionError('CPU CUDA seed')):
        branch(t,g,m)
        with pytest.raises(ValueError,match='ALREADY_APPLIED'):branch(t,g,m)


@pytest.mark.parametrize('bad',[True,-1,2**31,20260908])
def test_invalid_seed_does_not_mutate(bad):
    t,g,m=restored();before=weights(t);rng=torch.get_rng_state().clone();config=t.config
    with pytest.raises(ValueError,match='BRANCH_SEED'):branch(t,g,m,bad)
    assert weights(t)==before and t.config is config and torch.equal(torch.get_rng_state(),rng)


def test_origin_or_actual_counter_mismatch_refuses_branch():
    t,g,m=restored();bad=copy.deepcopy(m);bad['counters']['transition']+=1
    with pytest.raises(ValueError,match='ORIGIN_COUNTERS'):branch(t,g,bad)
    next(iter(t.actor_optimizer.state.values()))['step'].fill_(5119)
    with pytest.raises((ValueError,RuntimeError)):branch(t,g,m)
    assert not getattr(t,'_r6_rng_branch_applied',False)


def test_record_capture_failure_rolls_back_all_rng_and_config():
    from instinctlab.training import checkpoint_h1
    t,g,m=restored();config=t.config;before=checkpoint_h1._rng_payload(g,t);original=checkpoint_h1._rng_payload
    def capture(generator,trainer):
        if trainer.config is not config:raise OSError('record capture failed')
        return original(generator,trainer)
    with patch.object(checkpoint_h1,'_rng_payload',side_effect=capture):
        with pytest.raises(OSError,match='record capture failed'):branch(t,g,m)
    assert _state_fingerprint(original(g,t))==_state_fingerprint(before)
    assert t.config is config and not getattr(t,'_r6_rng_branch_applied',False)
