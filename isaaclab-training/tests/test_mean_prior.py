"""New actor-only soft prior, unchanged legacy PPO and pre-mutation failure."""
from copy import deepcopy
from types import SimpleNamespace as NS
import json
from pathlib import Path

import pytest
import torch

from instinctlab.training import PPOConfig
from training_workloads.unified_h1.mean_prior import zero_command_mean_prior, configure_trainer
from training_workloads.unified_h1.research_contract import validate_parameters, method_hashes
from .test_m6_ppo_contract import _trainer, _batch
from .test_formal_research_contract import profile, pin, refresh_transition_card


def test_exact_loss_gradient_command_boundary_and_actor_coordinates():
    state = torch.zeros(4, 52)
    state[:, 45] = torch.tensor([0., .001, .00101, -.2])
    mean = torch.tensor([[.8, .51, .48, .53]] * 4, requires_grad=True)
    loss = zero_command_mean_prior({'state_vec': state}, NS(mean_action=mean), 10.)
    assert loss.item() == pytest.approx(.7, abs=3e-6)
    loss.backward()
    expected = torch.tensor([[0., 5., -10., 15.]] * 2 + [[0.] * 4] * 2)
    assert torch.allclose(mean.grad, expected, atol=2e-5, rtol=1e-5)


def test_soft_term_does_not_replace_output_or_sample_rng():
    state = torch.zeros(2, 52); mean = torch.tensor([[.7, .6, .45, .54]] * 2, requires_grad=True)
    old = mean.detach().clone(); rng = torch.get_rng_state().clone()
    loss = zero_command_mean_prior({'state_vec': state}, NS(mean_action=mean), 10.)
    assert loss > 0 and torch.equal(mean, old) and torch.equal(torch.get_rng_state(), rng)
    loss.backward(); assert mean.grad[:, 1:4].abs().sum() > 0
    assert torch.equal(mean, old)


def test_zero_weight_is_exactly_zero_and_nonzero_commands_have_zero_gradient():
    mean = torch.rand(2, 4, requires_grad=True); state = torch.zeros(2, 52)
    assert zero_command_mean_prior({'state_vec': state}, NS(mean_action=mean), 0.).item() == 0.
    state[:, 45:48] = .2
    loss = zero_command_mean_prior({'state_vec': state}, NS(mean_action=mean), 10.)
    loss.backward(); assert loss.item() == 0. and mean.grad.eq(0).all()


def test_legacy_update_is_bit_exact_against_frozen_core():
    import subprocess, types, sys
    source = subprocess.check_output(['git', 'show', 'a13c5a4:isaaclab-training/instinctlab/training/ppo.py'], text=True)
    module = types.ModuleType('instinctlab.training._mean_prior_legacy_ppo')
    sys.modules[module.__name__] = module
    exec(compile(source, 'frozen-S0-ppo', 'exec'), module.__dict__)
    current = _trainer(19); batch = _batch(current)
    legacy = module.PPOTrainer(deepcopy(current.actor), deepcopy(current.critic), module.PPOConfig(**current.config.manifest()))
    legacy_batch = module.PPOBatch(**vars(batch))
    old = legacy.update(legacy_batch); new = current.update(batch)
    assert old['records'] == new['records']
    for field in ['actor', 'critic']:
        assert all(torch.equal(v, getattr(legacy, field).state_dict()[k]) for k, v in getattr(current, field).state_dict().items())


@pytest.mark.parametrize('bad', [float('nan'), [1., 2.]])
def test_invalid_auxiliary_stops_before_both_optimizer_steps(bad):
    trainer = _trainer(21); batch = _batch(trainer)
    trainer.actor_auxiliary = lambda policy, output: torch.tensor(bad)
    before = {k: v.clone() for k, v in trainer.actor.state_dict().items()}
    with pytest.raises(FloatingPointError, match='before either optimizer'):
        trainer.update(batch)
    assert not trainer.actor_optimizer.state and not trainer.critic_optimizer.state
    assert all(torch.equal(v, before[k]) for k, v in trainer.actor.state_dict().items())


def test_auxiliary_participates_in_real_actor_backward_and_records_scalar():
    trainer = _trainer(23, PPOConfig(minibatch_size=2)); batch = _batch(trainer)
    batch.policy['state_vec'][:, 45:48] = 0.
    with torch.no_grad():
        batch.old_log_probabilities.copy_(trainer.actor.log_probability(batch.policy, batch.actions))
    configure_trainer(trainer, {'mean_prior_weight': 10.})
    result = trainer.update(batch)
    assert result['records'][0]['actor_auxiliary_loss'] > 0
    assert len(trainer.actor_optimizer.state) > 0 and trainer.update_counter == 1


def test_new_profile_requires_registered_contract_and_exact_group_weight(tmp_path):
    p = profile(tmp_path)
    contract = Path(__file__).resolve().parents[2] / 'coordination/contracts/R5_ZERO_COMMAND_MEAN_PRIOR_V1.md'
    p.update(mean_prior_contract=pin(contract), mean_prior_weight=10.)
    refresh_transition_card(p)
    assert validate_parameters('research-training', p) == p
    hashes = method_hashes({'method': {'hashes': {}}}, p)
    assert hashes['mean_prior_contract'] == pin(contract)['sha256']
    p['mean_prior_weight'] = 9.; refresh_transition_card(p)
    with pytest.raises(PermissionError, match='registered profile'):
        validate_parameters('research-training', p)


def test_new_control_arm_reward_is_point05_not_old_R0_zero(tmp_path):
    p = profile(tmp_path, 'R0')
    contract = Path(__file__).resolve().parents[2] / 'coordination/contracts/R5_ZERO_COMMAND_MEAN_PRIOR_V1.md'
    p.update(mean_prior_contract=pin(contract), mean_prior_weight=0., residual_neutrality_weight=.05)
    refresh_transition_card(p); assert validate_parameters('research-training', p) == p
    old = {k: v for k, v in p.items() if k not in ['mean_prior_weight', 'mean_prior_contract']}
    refresh_transition_card(old)
    with pytest.raises(PermissionError, match='group/weight'):
        validate_parameters('research-training', old)
