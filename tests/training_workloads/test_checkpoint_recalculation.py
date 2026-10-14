"""Saved-probe checks reject wrong boundaries and diagnostic side effects."""
from types import SimpleNamespace
import pytest
import torch

from training_workloads.unified_h1.checkpoint_recalculation import recalculate_actor_probe


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(52, 8)
        self.consume_rng = False

    def forward(self, policy):
        if self.consume_rng:
            torch.rand(1)
        alpha, beta = (torch.nn.functional.softplus(self.head(policy['state_vec'])) + 1).chunk(2, -1)
        return SimpleNamespace(alpha=alpha, beta=beta)

    def log_probability_from_output(self, output, actions):
        return torch.distributions.Beta(output.alpha, output.beta).log_prob(actions).sum(-1, keepdim=True)


@pytest.fixture
def probe():
    torch.set_num_threads(2)
    actor = Actor()
    critic = torch.nn.Linear(21, 1)
    trainer = SimpleNamespace(actor=actor, critic=critic,
                              actor_optimizer=torch.optim.Adam(actor.parameters()),
                              critic_optimizer=torch.optim.Adam(critic.parameters()))
    policy = {'lidar_grid': torch.zeros(1, 12, 59, 360).expand(512, -1, -1, -1),
              'state_vec': torch.zeros(512, 52)}
    actions = torch.full((512, 4), .5)
    with torch.no_grad():
        output = actor(policy)
        distribution = torch.distributions.Beta(output.alpha, output.beta)
        reference = {'action_mean': distribution.mean.clone(),
                     'action_std': distribution.stddev.clone(),
                     'log_probability': distribution.log_prob(actions).sum(-1, keepdim=True)}
    return trainer, policy, actions, reference


def test_matches_actual_forward_without_sampling_or_optimizer(probe):
    trainer, policy, actions, reference = probe
    before = torch.get_rng_state().clone()
    result = recalculate_actor_probe(trainer, policy, actions, reference)
    assert result['status'] == 'PASS' and result['samples'] == 512
    assert torch.equal(before, torch.get_rng_state())
    assert trainer.actor_optimizer.state_dict()['state'] == {}
    assert trainer.critic_optimizer.state_dict()['state'] == {}


def test_rejects_wrong_checkpoint_boundary(probe):
    trainer, policy, actions, reference = probe
    with torch.no_grad():
        trainer.actor.head.bias.add_(.1)
    with pytest.raises(RuntimeError, match='recalculation mismatch'):
        recalculate_actor_probe(trainer, policy, actions, reference)


def test_rejects_broadcast_log_probability(probe):
    trainer, policy, actions, reference = probe
    reference['log_probability'] = reference['log_probability'][:, 0]
    with pytest.raises(ValueError, match='shape/dtype/finite'):
        recalculate_actor_probe(trainer, policy, actions, reference)


def test_rng_consumption_fails_and_restores_rng(probe):
    trainer, policy, actions, reference = probe
    trainer.actor.consume_rng = True
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='consumed RNG'):
        recalculate_actor_probe(trainer, policy, actions, reference)
    assert torch.equal(before, torch.get_rng_state())


def test_rejects_partial_frozen_input(probe):
    trainer, policy, actions, reference = probe
    policy = {k: v[:511] for k, v in policy.items()}
    with pytest.raises(ValueError, match='all512'):
        recalculate_actor_probe(trainer, policy, actions, reference)


def test_detects_model_mutation_during_forward(probe):
    trainer, policy, actions, reference = probe
    def mutate(module, inputs, output):
        module.head.bias.add_(.1)
    handle = trainer.actor.register_forward_hook(mutate)
    try:
        with pytest.raises(RuntimeError, match='changed model or optimizer'):
            recalculate_actor_probe(trainer, policy, actions, reference)
    finally:
        handle.remove()
