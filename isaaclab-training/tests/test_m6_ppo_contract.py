"""Exhaustive pure-tensor contract tests for the bounded M6 PPO core."""

from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from instinctlab.training import (
    CheckpointError,
    HistoryActor,
    NetworkConfig,
    PPOBatch,
    PPOConfig,
    PPOTrainer,
    PrivilegedCritic,
    clipped_surrogate,
    compute_gae,
    load_checkpoint,
    save_checkpoint,
)
from instinctlab.training.networks import ACTION_DIM, ACTOR_KEYS, CRITIC_DIM
from instinctlab.training.ppo import deterministic_minibatches, normalize_advantages
from instinctlab.task import compute_termination


ROOT = Path(__file__).resolve().parents[1]


def _policy(batch: int = 2, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    lidar = torch.zeros(batch, 12, 59, 360, dtype=torch.float32, device=device)
    lidar[:, 0::3].fill_(4.0)
    lidar[:, 1::3].fill_(1.0)
    lidar[:, 2::3].fill_(0.8)
    state = torch.linspace(-0.4, 0.4, 52, dtype=torch.float32, device=device).repeat(batch, 1)
    return {"lidar_grid": lidar, "state_vec": state}


def _trainer(seed: int = 7, config: PPOConfig | None = None, device: str = "cpu") -> PPOTrainer:
    torch.manual_seed(seed)
    actor = HistoryActor().to(device)
    critic = PrivilegedCritic().to(device)
    return PPOTrainer(actor, critic, config or PPOConfig(minibatch_size=1, epoch_count=1))


def _batch(trainer: PPOTrainer, batch: int = 2, device: str = "cpu") -> PPOBatch:
    policy = _policy(batch, device)
    privileged = torch.linspace(-1.0, 1.0, batch * CRITIC_DIM, dtype=torch.float32, device=device).reshape(batch, CRITIC_DIM)
    with torch.no_grad():
        output = trainer.actor(policy)
        actions = output.mean_action.detach().clone()
        old_log = trainer.actor.log_probability_from_output(output, actions).detach().clone()
    advantages = torch.linspace(-1.0, 1.0, batch, dtype=torch.float32, device=device).reshape(batch, 1)
    returns = torch.linspace(0.25, 0.75, batch, dtype=torch.float32, device=device).reshape(batch, 1)
    return PPOBatch(policy, privileged, actions, old_log, advantages, returns)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lidar_channels": (8, 16)}, {"lidar_channels": (8, 0, 16)}, {"state_hidden": 0},
        {"actor_hidden": True}, {"critic_hidden": (64,)}, {"concentration_min": 0.0},
        {"concentration_max": 1.0}, {"action_log_epsilon": 0.5},
    ],
)
def test_network_config_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        NetworkConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": -0.1}, {"gae_lambda": 1.1}, {"clip_ratio": 0.0},
        {"actor_learning_rate": float("nan")}, {"maximum_gradient_norm": 0.0},
        {"minibatch_size": 0}, {"epoch_count": True}, {"permutation_seed": -1},
    ],
)
def test_ppo_config_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        PPOConfig(**kwargs)


def test_configs_are_immutable():
    with pytest.raises(FrozenInstanceError):
        NetworkConfig().actor_hidden = 1
    with pytest.raises(FrozenInstanceError):
        PPOConfig().gamma = 0.1


@pytest.mark.parametrize("kind", ["missing", "extra", "reordered"])
def test_actor_rejects_nonexact_or_reordered_keys(kind):
    actor = HistoryActor()
    policy = _policy(1)
    if kind == "missing":
        policy.pop("state_vec")
    elif kind == "extra":
        policy["critic"] = torch.zeros(1, 18)
    else:
        policy = OrderedDict((key, policy[key]) for key in reversed(ACTOR_KEYS))
    with pytest.raises(ValueError):
        actor(policy)


@pytest.mark.parametrize("kind", ["lidar_shape", "state_shape", "batch", "dtype", "nan", "inf"])
def test_actor_rejects_shape_dtype_batch_and_nonfinite(kind):
    actor = HistoryActor()
    policy = _policy(2)
    if kind == "lidar_shape":
        policy["lidar_grid"] = policy["lidar_grid"][:, :, :, :-1]
    elif kind == "state_shape":
        policy["state_vec"] = policy["state_vec"][:, :-1]
    elif kind == "batch":
        policy["state_vec"] = policy["state_vec"][:1]
    elif kind == "dtype":
        policy["state_vec"] = policy["state_vec"].double()
    elif kind == "nan":
        policy["lidar_grid"][0, 0, 0, 0] = float("nan")
    else:
        policy["state_vec"][0, 0] = float("inf")
    with pytest.raises((TypeError, ValueError)):
        actor(policy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_actor_rejects_cross_device_pair():
    actor = HistoryActor()
    policy = _policy(1)
    policy["state_vec"] = policy["state_vec"].cuda()
    with pytest.raises(ValueError):
        actor(policy)


def test_actor_manifest_distribution_shapes_bounds_and_boundary_log_prob():
    torch.manual_seed(11)
    actor = HistoryActor()
    policy = _policy(2)
    output = actor(policy)
    assert output.alpha.shape == output.beta.shape == output.mean_action.shape == (2, ACTION_DIM)
    assert torch.all(output.alpha >= actor.config.concentration_min)
    assert torch.all(output.alpha <= actor.config.concentration_max)
    assert torch.all(output.beta >= actor.config.concentration_min)
    assert torch.all(output.beta <= actor.config.concentration_max)
    assert torch.all((output.mean_action >= 0.0) & (output.mean_action <= 1.0))
    boundary = torch.tensor([[0.0, 1.0, 0.25, 0.75], [1.0, 0.0, 0.5, 0.5]], dtype=torch.float32)
    assert actor.log_probability_from_output(output, boundary).shape == (2, 1)
    assert actor.entropy_from_output(output).shape == (2, 1)
    action, log_probability, entropy = actor.sample(policy)
    assert action.shape == (2, 4) and log_probability.shape == entropy.shape == (2, 1)
    assert torch.isfinite(action).all() and torch.isfinite(log_probability).all() and torch.isfinite(entropy).all()
    manifest = actor.parameter_manifest()
    assert manifest["actor_keys"] == ACTOR_KEYS
    assert manifest["parameter_count"] == sum(parameter.numel() for parameter in actor.parameters())
    assert manifest["parameter_count"] < 500_000


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan"])
def test_critic_rejects_invalid_input(kind):
    critic = PrivilegedCritic()
    privileged = torch.zeros(2, 18)
    if kind == "shape":
        privileged = privileged[:, :-1]
    elif kind == "dtype":
        privileged = privileged.double()
    else:
        privileged[0, 0] = float("nan")
    with pytest.raises((TypeError, ValueError)):
        critic(privileged)


def test_actor_critic_are_disjoint_and_privilege_cannot_change_actor():
    torch.manual_seed(19)
    actor = HistoryActor()
    critic = PrivilegedCritic()
    assert not ({id(value) for value in actor.parameters()} & {id(value) for value in critic.parameters()})
    policy = _policy(1)
    actor_first = actor(policy)
    first_privileged = torch.zeros(1, 18)
    second_privileged = torch.linspace(-2.0, 2.0, 18).reshape(1, 18)
    first_value = critic(first_privileged)
    second_value = critic(second_privileged)
    actor_second = actor(policy)
    assert not torch.equal(first_value, second_value)
    assert torch.equal(actor_first.alpha, actor_second.alpha)
    assert torch.equal(actor_first.beta, actor_second.beta)
    assert torch.equal(actor_first.mean_action, actor_second.mean_action)
    assert torch.equal(
        actor.log_probability_from_output(actor_first, actor_first.mean_action),
        actor.log_probability_from_output(actor_second, actor_second.mean_action),
    )
    changed_policy = _policy(1)
    changed_policy["state_vec"].add_(0.7)
    assert not torch.equal(actor_first.mean_action, actor(changed_policy).mean_action)


def _single_gae(*, reward=1.0, value=2.0, next_value=3.0, terminated=False, truncated=False,
                reset=False, timeout=5.0):
    scalar = lambda item: torch.tensor([[[item]]], dtype=torch.float32)
    mask = lambda item: torch.tensor([[[item]]], dtype=torch.bool)
    return compute_gae(
        scalar(reward), scalar(value), scalar(next_value), mask(terminated), mask(truncated), mask(reset),
        timeout_bootstrap_values=scalar(timeout), gamma=0.9, gae_lambda=0.8,
    )


def test_gae_hand_truth_table_for_all_four_transition_cases():
    continuation = _single_gae()
    termination = _single_gae(terminated=True)
    truncation = _single_gae(truncated=True)
    boundary = _single_gae(reset=True)
    assert continuation.advantages.item() == pytest.approx(1.0 + 0.9 * 3.0 - 2.0)
    assert termination.advantages.item() == pytest.approx(1.0 - 2.0)
    assert truncation.advantages.item() == pytest.approx(1.0 + 0.9 * 5.0 - 2.0)
    assert boundary.advantages.item() == pytest.approx(1.0 - 2.0)
    assert torch.equal(continuation.returns, continuation.advantages + torch.tensor([[[2.0]]]))


def test_accepted_task_simultaneous_masks_use_termination_precedence_and_ignore_both_bootstraps():
    task = compute_termination(
        height_w_m=torch.tensor([[0.4]], dtype=torch.float32),
        collision=torch.tensor([[False]]),
        elapsed_steps=torch.tensor([[1000]], dtype=torch.long),
    )
    assert task.terminated.item() and task.truncated.item()
    scalar = lambda value: torch.tensor([[[value]]], dtype=torch.float32)
    first = compute_gae(
        scalar(7.0), scalar(2.0), scalar(100.0), task.terminated.unsqueeze(0),
        task.truncated.unsqueeze(0), torch.zeros(1, 1, 1, dtype=torch.bool),
        timeout_bootstrap_values=scalar(-100.0), gamma=0.9, gae_lambda=0.8,
    )
    second = compute_gae(
        scalar(7.0), scalar(2.0), scalar(-500.0), task.terminated.unsqueeze(0),
        task.truncated.unsqueeze(0), torch.zeros(1, 1, 1, dtype=torch.bool),
        timeout_bootstrap_values=scalar(900.0), gamma=0.9, gae_lambda=0.8,
    )
    without_timeout_tensor = compute_gae(
        scalar(7.0), scalar(2.0), scalar(1234.0), task.terminated.unsqueeze(0),
        task.truncated.unsqueeze(0), torch.zeros(1, 1, 1, dtype=torch.bool),
        timeout_bootstrap_values=None, gamma=0.9, gae_lambda=0.8,
    )
    for result in (first, second, without_timeout_tensor):
        assert result.advantages.item() == pytest.approx(5.0)
        assert result.returns.item() == pytest.approx(7.0)
    assert torch.equal(first.advantages, second.advantages)
    assert torch.equal(first.returns, second.returns)


def test_gae_mixed_multistep_table_distinguishes_every_mask_and_recursion_cut():
    rewards = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])
    values = torch.tensor([[[0.5], [1.0]], [[1.5], [2.0]], [[2.5], [3.0]]])
    next_values = torch.tensor([[[10.0], [20.0]], [[30.0], [40.0]], [[50.0], [60.0]]])
    timeout = torch.tensor([[[100.0], [7.0]], [[200.0], [300.0]], [[400.0], [500.0]]])
    terminated = torch.tensor([[[False], [False]], [[True], [False]], [[True], [False]]])
    truncated = torch.tensor([[[False], [True]], [[True], [False]], [[False], [False]]])
    reset = torch.tensor([[[False], [False]], [[False], [False]], [[False], [True]]])
    result = compute_gae(
        rewards, values, next_values, terminated, truncated, reset,
        timeout_bootstrap_values=timeout, gamma=0.9, gae_lambda=0.8,
    )
    expected_advantages = torch.tensor([[[10.58], [7.3]], [[1.5], [40.16]], [[2.5], [3.0]]])
    expected_returns = expected_advantages + values
    torch.testing.assert_close(result.advantages, expected_advantages, rtol=0.0, atol=1.0e-5)
    torch.testing.assert_close(result.returns, expected_returns, rtol=0.0, atol=1.0e-5)


def test_gae_recursion_and_timeout_ignore_post_reset_value():
    rewards = torch.tensor([[[1.0]], [[2.0]]])
    values = torch.tensor([[[0.5]], [[0.25]]])
    next_values = torch.tensor([[[999.0]], [[4.0]]])
    terminated = torch.zeros(2, 1, 1, dtype=torch.bool)
    truncated = torch.tensor([[[True]], [[False]]])
    reset = torch.zeros_like(truncated)
    timeout = torch.tensor([[[3.0]], [[0.0]]])
    result = compute_gae(rewards, values, next_values, terminated, truncated, reset,
                         timeout_bootstrap_values=timeout, gamma=0.9, gae_lambda=0.8)
    assert result.advantages[0].item() == pytest.approx(1.0 + 0.9 * 3.0 - 0.5)
    assert result.advantages[1].item() == pytest.approx(2.0 + 0.9 * 4.0 - 0.25)


def test_gae_rejects_missing_pure_timeout_bootstrap_and_bad_dtype():
    value = torch.zeros(1, 1, 1)
    false = torch.zeros(1, 1, 1, dtype=torch.bool)
    true = torch.ones_like(false)
    with pytest.raises(ValueError):
        compute_gae(value, value, value, false, true, false, timeout_bootstrap_values=None)
    simultaneous = compute_gae(value, value, value, true, true, false, timeout_bootstrap_values=None)
    assert torch.equal(simultaneous.advantages, value)
    assert torch.equal(simultaneous.returns, value)
    with pytest.raises(TypeError):
        compute_gae(value.double(), value, value, false, false, false, timeout_bootstrap_values=value)


def test_advantage_normalization_and_zero_variance_edge():
    values = torch.tensor([[-1.0], [0.0], [1.0]])
    normalized = normalize_advantages(values)
    assert normalized.mean().item() == pytest.approx(0.0, abs=1e-7)
    assert normalized.square().mean().item() == pytest.approx(1.0, rel=1e-6)
    assert torch.equal(normalize_advantages(torch.ones(3, 1)), torch.zeros(3, 1))
    with pytest.raises(ValueError):
        normalize_advantages(torch.tensor([[float("nan")]]))


def test_deterministic_minibatches_cover_every_sample_without_drop():
    first = deterministic_minibatches(8, 2, 2, 13)
    second = deterministic_minibatches(8, 2, 2, 13)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    for epoch in range(2):
        assert sorted(torch.cat(first[epoch * 4:(epoch + 1) * 4]).tolist()) == list(range(8))
    with pytest.raises(ValueError):
        deterministic_minibatches(7, 2, 1, 0)


def test_clipped_surrogate_matches_hand_computation():
    old = torch.zeros(2, 1)
    ratio = torch.tensor([[1.5], [0.5]])
    new = torch.log(ratio)
    advantage = torch.tensor([[2.0], [-2.0]])
    result = clipped_surrogate(new, old, advantage, 0.2)
    expected = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
    assert result["actor_loss"].item() == pytest.approx(expected.item())
    assert result["clip_fraction"].item() == 1.0
    assert result["approximate_kl"].item() == pytest.approx((-new).mean().item())


@pytest.mark.parametrize("kind", ["action_shape", "action_domain", "old_nan", "return_inf", "indivisible"])
def test_update_rejects_invalid_batch_before_mutation(kind):
    config = PPOConfig(minibatch_size=2 if kind == "indivisible" else 1, epoch_count=1)
    trainer = _trainer(config=config)
    batch = _batch(trainer, 3 if kind == "indivisible" else 2)
    if kind == "action_shape":
        batch = PPOBatch(batch.policy, batch.privileged, batch.actions[:, :-1], batch.old_log_probabilities, batch.advantages, batch.returns)
    elif kind == "action_domain":
        actions = batch.actions.clone(); actions[0, 0] = -0.1
        batch = PPOBatch(batch.policy, batch.privileged, actions, batch.old_log_probabilities, batch.advantages, batch.returns)
    elif kind == "old_nan":
        old = batch.old_log_probabilities.clone(); old[0] = float("nan")
        batch = PPOBatch(batch.policy, batch.privileged, batch.actions, old, batch.advantages, batch.returns)
    elif kind == "return_inf":
        returns = batch.returns.clone(); returns[0] = float("inf")
        batch = PPOBatch(batch.policy, batch.privileged, batch.actions, batch.old_log_probabilities, batch.advantages, returns)
    before = {name: value.detach().clone() for name, value in trainer.actor.named_parameters()}
    with pytest.raises((TypeError, ValueError)):
        trainer.update(batch)
    assert all(torch.equal(value, before[name]) for name, value in trainer.actor.named_parameters())


def test_bounded_update_is_finite_covers_samples_and_preserves_old_log_prob():
    trainer = _trainer(config=PPOConfig(minibatch_size=1, epoch_count=1, maximum_gradient_norm=1.0e-4))
    batch = _batch(trainer, 2)
    old = batch.old_log_probabilities.clone()
    actor_before = {name: value.detach().clone() for name, value in trainer.actor.named_parameters()}
    metrics = trainer.update(batch)
    assert metrics["update_counter"] == 1
    assert torch.equal(metrics["coverage"], torch.ones(1, 2, dtype=torch.int64))
    assert torch.equal(old, batch.old_log_probabilities)
    assert any(not torch.equal(value, actor_before[name]) for name, value in trainer.actor.named_parameters())
    for record in metrics["records"]:
        assert all(torch.isfinite(torch.tensor(value)) for value in record.values())
        assert record["actor_pre_clip_gradient_norm"] >= 0.0


def test_critic_only_step_cannot_mutate_actor():
    trainer = _trainer()
    actor_before = {name: value.detach().clone() for name, value in trainer.actor.named_parameters()}
    result = trainer.critic_only_step(torch.randn(2, 18), torch.randn(2, 1))
    assert result["pre_clip_gradient_norm"] >= 0.0
    assert all(torch.equal(value, actor_before[name]) for name, value in trainer.actor.named_parameters())


def test_checkpoint_round_trip_restores_exact_outputs_optimizer_config_counter_and_rng(tmp_path):
    trainer = _trainer()
    batch = _batch(trainer, 2)
    trainer.update(batch)
    policy = _policy(2)
    privileged = batch.privileged
    with torch.no_grad():
        expected_action = trainer.actor(policy).mean_action.clone()
        expected_value = trainer.critic(privileged).clone()
    path = save_checkpoint((tmp_path / "checkpoint.pt").resolve(), trainer)
    saved_rng = torch.get_rng_state().clone()
    restored = _trainer(seed=999)
    load_checkpoint(path, restored)
    with torch.no_grad():
        assert torch.equal(expected_action, restored.actor(policy).mean_action)
        assert torch.equal(expected_value, restored.critic(privileged))
    assert restored.update_counter == 1
    assert restored.config == trainer.config and restored.actor.config == trainer.actor.config
    assert restored.actor_optimizer.state_dict().keys() == trainer.actor_optimizer.state_dict().keys()
    assert torch.equal(torch.get_rng_state(), saved_rng)


@pytest.mark.parametrize("kind", ["missing", "extra", "corrupt", "nonfinite", "shape", "dtype", "config"])
def test_checkpoint_rejects_malformed_payloads(tmp_path, kind):
    trainer = _trainer()
    trainer.update(_batch(trainer, 2))
    valid = save_checkpoint((tmp_path / "valid.pt").resolve(), trainer)
    target = (tmp_path / f"{kind}.pt").resolve()
    if kind == "corrupt":
        target.write_bytes(b"not a torch checkpoint")
    else:
        payload = torch.load(valid, weights_only=True)
        if kind == "missing":
            payload.pop("rng")
        elif kind == "extra":
            payload["unexpected"] = 1
        elif kind == "nonfinite":
            next(iter(payload["actor"].values())).flatten()[0] = float("nan")
        elif kind == "shape":
            key = next(iter(payload["critic"]))
            payload["critic"][key] = payload["critic"][key].flatten()[:-1]
        elif kind == "dtype":
            key = next(iter(payload["actor"]))
            payload["actor"][key] = payload["actor"][key].double()
        else:
            payload["ppo_config"]["gamma"] = 0.5
        torch.save(payload, target)
    with pytest.raises(CheckpointError):
        load_checkpoint(target, _trainer())


def test_checkpoint_requires_explicit_existing_output_directory(tmp_path):
    trainer = _trainer()
    with pytest.raises(ValueError):
        save_checkpoint(Path("relative.pt"), trainer)
    with pytest.raises(ValueError):
        save_checkpoint((tmp_path / "missing" / "value.pt").resolve(), trainer)


def test_recursive_actor_source_leak_and_forbidden_runtime_import_scan():
    network_source = (ROOT / "instinctlab/training/networks.py").read_text()
    training_sources = "\n".join(path.read_text() for path in sorted((ROOT / "instinctlab/training").glob("*.py")))
    tree = ast.parse(network_source)
    actor_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HistoryActor")
    forward = next(node for node in actor_class.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    assert [argument.arg for argument in forward.args.args] == ["self", "policy"]
    critic_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PrivilegedCritic")
    critic_forward = next(node for node in critic_class.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    assert [argument.arg for argument in critic_forward.args.args] == ["self", "privileged"]
    identifiers = {
        value.lower()
        for node in ast.walk(actor_class)
        for value in (
            [node.id] if isinstance(node, ast.Name) else
            [node.arg] if isinstance(node, ast.arg) else
            [node.attr] if isinstance(node, ast.Attribute) else []
        )
    }
    for token in ("critic", "root_state", "actual_velocity", "clearance", "collision", "reward", "termination", "controller", "wrench", "rotor", "direction", "generator"):
        assert all(token not in identifier for identifier in identifiers)
    assert "map" not in identifiers
    imported_modules = []
    for path in sorted((ROOT / "instinctlab/training").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
    for token in ("isaaclab", "omni", "tensordict", "torchrl", "instinctlab.envs"):
        assert all(module != token and not module.startswith(token + ".") for module in imported_modules)
    for token in ("gru", "lstm", "cost_critic", "lagrange", "ppo-lagrangian", "nan_to_num"):
        assert token not in training_sources.lower()
