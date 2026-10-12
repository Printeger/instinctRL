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
    assert json.loads((tmp_path / "first_update_failure.json").read_text())["completed_actor_steps"] == 0


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


def test_pre_hook_attempt_is_not_completed_update(tmp_path):
    trainer, batch = fixture()
    diagnostic = FirstUpdateDiagnostic(tmp_path, enabled=True)
    diagnostic.before_update(trainer, batch)
    before = _digest(_state(trainer))
    hook = next(iter(trainer.actor_optimizer._optimizer_step_pre_hooks.values()))
    hook(trainer.actor_optimizer, (), {})  # Callback only, deliberately no optimizer.step.
    diagnostic.abort(RuntimeError('failure after pre-hook before optimizer body'))
    evidence = json.loads((tmp_path / 'first_update_failure.json').read_text())
    assert evidence['attempted_actor_steps'] == 1
    assert evidence['completed_actor_steps'] == 0
    assert _digest(_state(trainer)) == before
    assert not trainer.actor_optimizer._optimizer_step_pre_hooks
    assert not trainer.actor.weight._backward_hooks


def test_real_product_network_collection_binding_and_no_update_gate(tmp_path):
    from instinctlab.training.networks_h1 import PrivilegedCriticH1
    torch.set_num_threads(2)
    actor = HistoryActor()
    critic = PrivilegedCriticH1()
    trainer = NS(actor=actor, critic=critic, actor_optimizer=torch.optim.Adam(actor.parameters()),
                 critic_optimizer=torch.optim.Adam(critic.parameters()), update_counter=0)
    diagnostic = FirstUpdateDiagnostic(tmp_path, enabled=True)
    original = actor.distribution
    policies=[]; actions=[]; logps=[]; expected_alpha=[]; expected_beta=[]
    lidar = torch.zeros(32, 12, 59, 360)
    lidar[:, 0::3] = 20.
    lidar[:, 1::3] = 1.
    lidar[:, 2::3] = 1.
    rng = torch.get_rng_state().clone()
    with torch.no_grad(), diagnostic.capture_rollout_distribution(trainer):
        for step in range(16):
            state = torch.zeros(32, 52)
            state[:, 0] = torch.arange(32) / 32 + step
            policy={'lidar_grid':lidar, 'state_vec':state}
            distribution, output = actor.distribution(policy)
            action = torch.full((32, 4), .4 + step / 100)
            logp=actor.log_probability_from_output(output, action)
            policies.append(state.clone());actions.append(action.clone());logps.append(logp.clone())
            expected_alpha.append(output.alpha.clone());expected_beta.append(output.beta.clone())
            # Observer captures must remain detached from original output storage.
            output.alpha.fill_(99.);output.beta.fill_(98.)
    assert actor.distribution == original
    assert torch.equal(rng, torch.get_rng_state())
    batch = NS(policy={'lidar_grid':lidar.repeat(16,1,1,1), 'state_vec':torch.cat(policies)},
               privileged=torch.zeros(512,21), actions=torch.cat(actions),
               old_log_probabilities=torch.cat(logps), advantages=torch.ones(512,1),returns=torch.ones(512,1))
    diagnostic.bind_rollout_distribution(batch)
    torch.testing.assert_close(batch.old_alpha,torch.cat(expected_alpha),rtol=0,atol=0)
    torch.testing.assert_close(batch.old_beta,torch.cat(expected_beta),rtol=0,atol=0)
    assert batch.policy['state_vec'][32,0] == 1.
    assert batch.actions[32,0] == actions[1][0,0]
    before = _digest(_state(trainer))
    diagnostic.before_update(trainer,batch)
    assert _digest(_state(trainer)) == before
    assert torch.equal(rng,torch.get_rng_state())
    diagnostic.detach_hooks()
    # Wrong input, concentration, or row order must reject without an update.
    original_state=batch.policy['state_vec'].clone()
    for corruption in ('input','order','concentration'):
        if corruption=='input': batch.policy['state_vec'][71,0]+=5
        elif corruption=='order': batch.policy['state_vec'][[71,200]]=batch.policy['state_vec'][[200,71]]
        else: batch.old_alpha[71,0]+=.2
        with pytest.raises(RuntimeError,match='pre-update probability/distribution gate failed'):
            diagnostic.before_update(trainer,batch)
        assert _digest(_state(trainer)) == before
        assert torch.equal(rng,torch.get_rng_state())
        batch.policy['state_vec'].copy_(original_state)
    assert trainer.update_counter == 0
    assert not diagnostic._handles and not diagnostic._gradient_handles
    # Collection failure also restores the original method.
    with pytest.raises(RuntimeError,match='collection interruption'):
        with diagnostic.capture_rollout_distribution(trainer):
            raise RuntimeError('collection interruption')
    assert actor.distribution == original


def test_mocked_eight_callback_boundaries_detach_normally_without_updates(tmp_path):
    trainer, batch = fixture()
    diagnostic = FirstUpdateDiagnostic(tmp_path, enabled=True)
    diagnostic.before_update(trainer, batch)
    before = _digest(_state(trainer))
    pre = next(iter(trainer.actor_optimizer._optimizer_step_pre_hooks.values()))
    post = next(iter(trainer.actor_optimizer._optimizer_step_post_hooks.values()))
    for _ in range(8):
        pre(trainer.actor_optimizer, (), {})
        post(trainer.actor_optimizer, (), {})
    diagnostic.after_update(trainer, batch, {'records':[{'actor_pre_clip_gradient_norm':0.}]*8})
    assert _digest(_state(trainer)) == before  # Callback lifecycle only, no optimizer body.
    assert not diagnostic._handles and not diagnostic._gradient_handles
    assert not trainer.actor_optimizer._optimizer_step_pre_hooks
    saved=torch.load(tmp_path/'first_update_probe.pt',weights_only=False)
    assert saved['correction_units']=='m/s' and saved['correction_heads']==[1,2,3]
    assert len(saved['minibatches'])==8
