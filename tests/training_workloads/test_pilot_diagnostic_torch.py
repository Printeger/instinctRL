"""Real CPU tensor regressions; no optimizer step or training checkpoint."""
from pathlib import Path
from types import SimpleNamespace as NS
import json

import pytest
import torch

from instinctlab.control.governor import decode_governor
from instinctlab.training.networks import HistoryActor, NetworkConfig, PolicyOutput
from training_workloads.unified_h1.pilot_diagnostic import FirstUpdateDiagnostic, _digest, _state


class TensorActor(torch.nn.Module):
    # Use the product probability function on deterministic real Beta tensors.
    log_probability_from_output = HistoryActor.log_probability_from_output
    def __init__(self):
        super().__init__()
        self.config = NetworkConfig()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
    def forward(self, policy):
        alpha = self.weight.expand(512, 4)
        beta = torch.full_like(alpha, 3.0)
        return PolicyOutput(alpha, beta, alpha / (alpha + beta))


def fixture():
    actor = TensorActor()
    critic = torch.nn.Linear(21, 1)
    trainer = NS(actor=actor, critic=critic, actor_optimizer=torch.optim.Adam(actor.parameters()),
                 critic_optimizer=torch.optim.Adam(critic.parameters()), update_counter=0)
    actions = torch.full((512, 4), .5)
    actions[:5] = torch.tensor([[0., .5, .5, .5], [1., .5, .5, .5],
                               [.5, .5, .5, 1.], [.5, 0., 1., 0.], [.5, 1., 0., 1.]])
    policy = {"state_vec": torch.zeros(512, 52)}
    output = actor(policy)
    batch = NS(policy=policy, privileged=torch.zeros(512, 21), actions=actions,
               old_log_probabilities=actor.log_probability_from_output(output, actions).detach().clone(),
               old_alpha=output.alpha.detach().clone(), old_beta=output.beta.detach().clone(),
               advantages=torch.ones(512, 1), returns=torch.ones(512, 1))
    return trainer, batch


def test_probe_head_units_and_product_decoder(tmp_path):
    trainer, batch = fixture()
    diagnostic = FirstUpdateDiagnostic(tmp_path, enabled=True)
    rng = torch.get_rng_state().clone()
    before = _digest(_state(trainer))
    diagnostic.before_update(trainer, batch)
    row = diagnostic._rows[0]
    assert row["issued_v_corr_b"].shape == (512, 3)
    torch.testing.assert_close(row["issued_v_corr_b"], decode_governor(batch.actions, batch.policy["state_vec"]).v_corr_b)
    torch.testing.assert_close(row["issued_v_corr_b"][:2], torch.zeros(2, 3))
    torch.testing.assert_close(row["issued_v_corr_b"][2], torch.tensor([0., 0., .5]))
    torch.testing.assert_close(row["issued_v_corr_b"][3:5], torch.tensor([[-.5, .5, -.5], [.5, -.5, .5]]))
    torch.testing.assert_close(row["mean_v_corr_b"], torch.full((512, 3), -.1))
    assert row["sample_minus_mean_action"].shape == (512, 4)
    batch.actions[2, 3] = 0.  # Probe is detached from mutable rollout storage.
    assert row["issued_v_corr_b"][2, 2] == .5
    assert _digest(_state(trainer)) == before
    assert torch.equal(rng, torch.get_rng_state())
    diagnostic.abort(RuntimeError("test interruption before any update"))
    assert not diagnostic._handles and not diagnostic._gradient_handles
    assert not trainer.actor_optimizer._optimizer_step_pre_hooks
    assert not trainer.actor.weight._backward_hooks
    partial = torch.load(tmp_path / "first_update_partial_probe.pt", weights_only=False)
    assert partial["schema_version"].endswith("/2")
    assert json.loads((tmp_path / "first_update_failure.json").read_text())["observed_actor_steps"] == 0


@pytest.mark.parametrize("field", ["old_alpha", "old_beta", "old_log_probabilities"])
def test_gate_mismatch_before_update_keeps_all_state_and_rng(tmp_path, field):
    trainer, batch = fixture()
    getattr(batch, field)[71, 0] += .2
    before = _digest(_state(trainer))
    rng = torch.get_rng_state().clone()
    diagnostic = FirstUpdateDiagnostic(tmp_path, enabled=True)
    with pytest.raises(RuntimeError, match="pre-update probability/distribution gate failed"):
        diagnostic.before_update(trainer, batch)
    assert _digest(_state(trainer)) == before
    assert torch.equal(rng, torch.get_rng_state())
    assert not diagnostic._handles and not diagnostic._gradient_handles
    assert trainer.update_counter == 0
    diagnostic.abort(RuntimeError("gate rejected"))
    assert (tmp_path / "first_update_failure.json").is_file()


def test_disabled_observer_is_noop(tmp_path):
    diagnostic = FirstUpdateDiagnostic(tmp_path, enabled=False)
    diagnostic.before_update(None, None)
    diagnostic.abort(RuntimeError("disabled"))
    assert not list(tmp_path.iterdir())
