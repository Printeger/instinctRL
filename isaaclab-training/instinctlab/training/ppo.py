"""Fail-closed GAE and bounded PPO-Clip pure-tensor update core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .networks import ACTION_DIM, HistoryActor, PrivilegedCritic, validate_critic_input, validate_policy


def _finite_scalar(name: str, value: float, *, lower: float, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < lower or (upper is not None and result > upper):
        raise ValueError(f"{name} is outside its finite validated range")
    return result


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    maximum_gradient_norm: float = 0.5
    advantage_epsilon: float = 1.0e-8
    minibatch_size: int = 2
    epoch_count: int = 1
    permutation_seed: int = 1701

    def __post_init__(self) -> None:
        for name in ("gamma", "gae_lambda"):
            object.__setattr__(self, name, _finite_scalar(name, getattr(self, name), lower=0.0, upper=1.0))
        object.__setattr__(self, "clip_ratio", _finite_scalar("clip_ratio", self.clip_ratio, lower=1.0e-8, upper=1.0))
        object.__setattr__(self, "value_loss_coefficient", _finite_scalar("value_loss_coefficient", self.value_loss_coefficient, lower=0.0))
        object.__setattr__(self, "entropy_coefficient", _finite_scalar("entropy_coefficient", self.entropy_coefficient, lower=0.0))
        object.__setattr__(self, "actor_learning_rate", _finite_scalar("actor_learning_rate", self.actor_learning_rate, lower=1.0e-12))
        object.__setattr__(self, "critic_learning_rate", _finite_scalar("critic_learning_rate", self.critic_learning_rate, lower=1.0e-12))
        object.__setattr__(self, "maximum_gradient_norm", _finite_scalar("maximum_gradient_norm", self.maximum_gradient_norm, lower=1.0e-12))
        object.__setattr__(self, "advantage_epsilon", _finite_scalar("advantage_epsilon", self.advantage_epsilon, lower=1.0e-12))
        for name in ("minibatch_size", "epoch_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.permutation_seed, bool) or not isinstance(self.permutation_seed, int) or self.permutation_seed < 0:
            raise ValueError("permutation_seed must be a nonnegative integer")

    def manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GAEOutput:
    advantages: torch.Tensor
    returns: torch.Tensor


def _trajectory_float(name: str, value: torch.Tensor, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32")
    if value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] <= 0 or value.shape[2] != 1:
        raise ValueError(f"{name} must have shape (T, N, 1) with nonempty T and N")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must share trajectory shape {shape}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")
    return value


def _trajectory_mask(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise TypeError(f"{name} must be a boolean torch.Tensor")
    if value.shape != reference.shape or value.device != reference.device:
        raise ValueError(f"{name} must share trajectory shape and device")
    return value


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    reset_boundary: torch.Tensor,
    *,
    timeout_bootstrap_values: torch.Tensor | None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> GAEOutput:
    """Compute GAE without ever recursing or bootstrapping into a reset episode."""
    rewards = _trajectory_float("rewards", rewards)
    shape = tuple(rewards.shape)
    values = _trajectory_float("values", values, shape)
    next_values = _trajectory_float("next_values", next_values, shape)
    if values.device != rewards.device or next_values.device != rewards.device:
        raise ValueError("GAE float tensors must share one device")
    terminated = _trajectory_mask("terminated", terminated, rewards)
    truncated = _trajectory_mask("truncated", truncated, rewards)
    reset_boundary = _trajectory_mask("reset_boundary", reset_boundary, rewards)
    if bool((terminated & truncated).any().item()):
        raise ValueError("one transition cannot be both terminated and truncated")
    if bool(truncated.any().item()) and timeout_bootstrap_values is None:
        raise ValueError("truncated transitions require explicit terminal-state timeout bootstrap values")
    if timeout_bootstrap_values is None:
        timeout_bootstrap_values = torch.zeros_like(rewards)
    timeout_bootstrap_values = _trajectory_float("timeout_bootstrap_values", timeout_bootstrap_values, shape)
    if timeout_bootstrap_values.device != rewards.device:
        raise ValueError("timeout_bootstrap_values must share the trajectory device")
    gamma = _finite_scalar("gamma", gamma, lower=0.0, upper=1.0)
    gae_lambda = _finite_scalar("gae_lambda", gae_lambda, lower=0.0, upper=1.0)

    ordinary = ~(terminated | truncated | reset_boundary)
    bootstrap = torch.where(truncated, timeout_bootstrap_values, torch.where(ordinary, next_values, torch.zeros_like(values)))
    deltas = rewards + gamma * bootstrap - values
    recursion = ordinary.to(dtype=rewards.dtype)
    advantages = torch.empty_like(rewards)
    accumulator = torch.zeros_like(rewards[0])
    for step in range(rewards.shape[0] - 1, -1, -1):
        accumulator = deltas[step] + gamma * gae_lambda * recursion[step] * accumulator
        advantages[step] = accumulator
    returns = advantages + values
    if not bool(torch.isfinite(advantages).all().item()) or not bool(torch.isfinite(returns).all().item()):
        raise FloatingPointError("GAE produced nonfinite advantages or returns")
    return GAEOutput(advantages=advantages, returns=returns)


def normalize_advantages(advantages: torch.Tensor, epsilon: float = 1.0e-8) -> torch.Tensor:
    if not isinstance(advantages, torch.Tensor) or advantages.dtype != torch.float32 or advantages.numel() == 0:
        raise TypeError("advantages must be a nonempty float32 tensor")
    if not bool(torch.isfinite(advantages).all().item()):
        raise ValueError("advantages must be finite")
    epsilon = _finite_scalar("epsilon", epsilon, lower=1.0e-12)
    centered = advantages - advantages.mean()
    variance = centered.square().mean()
    if not bool(torch.isfinite(variance).item()):
        raise FloatingPointError("advantage variance is nonfinite")
    if bool((variance == 0.0).item()):
        return torch.zeros_like(advantages)
    normalized = centered / torch.sqrt(variance + epsilon)
    if not bool(torch.isfinite(normalized).all().item()):
        raise FloatingPointError("advantage normalization produced nonfinite values")
    return normalized


def deterministic_minibatches(batch_size: int, minibatch_size: int, epoch_count: int, seed: int) -> tuple[torch.Tensor, ...]:
    for name, value in (("batch_size", batch_size), ("minibatch_size", minibatch_size), ("epoch_count", epoch_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if batch_size % minibatch_size != 0:
        raise ValueError("batch_size must be exactly divisible by minibatch_size; samples may not be dropped")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    result = []
    for _ in range(epoch_count):
        permutation = torch.randperm(batch_size, generator=generator)
        result.extend(permutation.split(minibatch_size))
    return tuple(result)


@dataclass(frozen=True)
class PPOBatch:
    policy: Mapping[str, torch.Tensor]
    privileged: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


def _validate_batch(batch: PPOBatch) -> int:
    if not isinstance(batch, PPOBatch):
        raise TypeError("batch must be PPOBatch")
    lidar, _ = validate_policy(batch.policy)
    privileged = validate_critic_input(batch.privileged)
    size = lidar.shape[0]
    if privileged.shape[0] != size or privileged.device != lidar.device:
        raise ValueError("actor and critic branches must share batch size and device")
    expected = {
        "actions": (batch.actions, (size, ACTION_DIM)),
        "old_log_probabilities": (batch.old_log_probabilities, (size, 1)),
        "advantages": (batch.advantages, (size, 1)),
        "returns": (batch.returns, (size, 1)),
    }
    for name, (value, shape) in expected.items():
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value.dtype != torch.float32 or value.device != lidar.device:
            raise TypeError(f"{name} must use float32 on the batch device")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must be finite")
    if bool(((batch.actions < 0.0) | (batch.actions > 1.0)).any().item()):
        raise ValueError("actions must remain in the closed [0, 1] domain")
    return size


def clipped_surrogate(
    new_log_probability: torch.Tensor,
    old_log_probability: torch.Tensor,
    normalized_advantage: torch.Tensor,
    clip_ratio: float,
) -> dict[str, torch.Tensor]:
    if new_log_probability.shape != old_log_probability.shape or new_log_probability.shape != normalized_advantage.shape:
        raise ValueError("PPO log probabilities and advantages must have identical [N,1] shape")
    for name, value in (("new_log_probability", new_log_probability), ("old_log_probability", old_log_probability), ("normalized_advantage", normalized_advantage)):
        if value.dtype != torch.float32 or value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] != 1:
            raise TypeError(f"{name} must be nonempty float32 [N,1]")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must be finite")
    clip_ratio = _finite_scalar("clip_ratio", clip_ratio, lower=1.0e-8, upper=1.0)
    ratio = torch.exp(new_log_probability - old_log_probability)
    unclipped = ratio * normalized_advantage
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * normalized_advantage
    actor_loss = -torch.minimum(unclipped, clipped).mean()
    approximate_kl = (old_log_probability - new_log_probability).mean()
    clip_fraction = (torch.abs(ratio - 1.0) > clip_ratio).to(torch.float32).mean()
    outputs = {"ratio": ratio, "unclipped": unclipped, "clipped": clipped, "actor_loss": actor_loss,
               "approximate_kl": approximate_kl, "clip_fraction": clip_fraction}
    if any(not bool(torch.isfinite(value).all().item()) for value in outputs.values()):
        raise FloatingPointError("PPO clipped surrogate produced nonfinite output")
    return outputs


def _module_finite(module: nn.Module, label: str) -> None:
    for name, parameter in module.named_parameters():
        if not bool(torch.isfinite(parameter).all().item()):
            raise FloatingPointError(f"{label} parameter {name} is nonfinite")


def _gradients_finite(module: nn.Module, label: str) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            raise FloatingPointError(f"{label} gradient {name} is nonfinite")


def _optimizer_finite(optimizer: torch.optim.Optimizer, label: str) -> None:
    for parameter_state in optimizer.state.values():
        for name, value in parameter_state.items():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError(f"{label} optimizer state {name} is nonfinite")


class PPOTrainer:
    """One bounded deterministic PPO interface with structurally separate optimizers."""

    def __init__(self, actor: HistoryActor, critic: PrivilegedCritic, config: PPOConfig = PPOConfig()) -> None:
        if not isinstance(actor, HistoryActor) or not isinstance(critic, PrivilegedCritic):
            raise TypeError("actor and critic must use the exact training network classes")
        if not isinstance(config, PPOConfig):
            raise TypeError("config must be PPOConfig")
        if {id(parameter) for parameter in actor.parameters()} & {id(parameter) for parameter in critic.parameters()}:
            raise ValueError("actor and critic parameters must be disjoint")
        self.actor = actor
        self.critic = critic
        self.config = config
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate)
        self.update_counter = 0

    def update(self, batch: PPOBatch) -> dict[str, object]:
        size = _validate_batch(batch)
        if size % self.config.minibatch_size != 0:
            raise ValueError("batch must be divisible by configured minibatch_size")
        old_snapshot = batch.old_log_probabilities.detach().clone()
        advantages = normalize_advantages(batch.advantages, self.config.advantage_epsilon)
        indices = deterministic_minibatches(
            size, self.config.minibatch_size, self.config.epoch_count,
            self.config.permutation_seed + self.update_counter,
        )
        records: list[dict[str, float | int]] = []
        coverage = torch.zeros(self.config.epoch_count, size, dtype=torch.int64)
        minibatches_per_epoch = size // self.config.minibatch_size
        for iteration, cpu_indices in enumerate(indices):
            epoch = iteration // minibatches_per_epoch
            coverage[epoch, cpu_indices] += 1
            device_indices = cpu_indices.to(device=batch.actions.device)
            policy = {key: batch.policy[key].index_select(0, device_indices) for key in batch.policy}
            privileged = batch.privileged.index_select(0, device_indices)
            actions = batch.actions.index_select(0, device_indices)
            old_log_probability = batch.old_log_probabilities.index_select(0, device_indices).detach()
            normalized_advantage = advantages.index_select(0, device_indices)
            target_return = batch.returns.index_select(0, device_indices)

            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            output = self.actor(policy)
            new_log_probability = self.actor.log_probability_from_output(output, actions)
            entropy = self.actor.entropy_from_output(output)
            value = self.critic(privileged)
            surrogate = clipped_surrogate(new_log_probability, old_log_probability, normalized_advantage, self.config.clip_ratio)
            value_loss = 0.5 * (value - target_return).square().mean()
            entropy_mean = entropy.mean()
            total_loss = (
                surrogate["actor_loss"]
                + self.config.value_loss_coefficient * value_loss
                - self.config.entropy_coefficient * entropy_mean
            )
            for name, tensor in (("value_loss", value_loss), ("entropy", entropy_mean), ("total_loss", total_loss)):
                if not bool(torch.isfinite(tensor).all().item()):
                    raise FloatingPointError(f"{name} is nonfinite before optimizer mutation")
            total_loss.backward()
            _gradients_finite(self.actor, "actor")
            _gradients_finite(self.critic, "critic")
            actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.maximum_gradient_norm)
            critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.maximum_gradient_norm)
            if not bool(torch.isfinite(actor_norm).item()) or not bool(torch.isfinite(critic_norm).item()):
                raise FloatingPointError("pre-clip gradient norm is nonfinite")
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            _module_finite(self.actor, "actor")
            _module_finite(self.critic, "critic")
            _optimizer_finite(self.actor_optimizer, "actor")
            _optimizer_finite(self.critic_optimizer, "critic")
            records.append({
                "epoch": epoch,
                "minibatch": iteration % minibatches_per_epoch,
                "actor_loss": float(surrogate["actor_loss"].detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "entropy": float(entropy_mean.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "ratio_mean": float(surrogate["ratio"].detach().mean().item()),
                "approximate_kl": float(surrogate["approximate_kl"].detach().item()),
                "clip_fraction": float(surrogate["clip_fraction"].detach().item()),
                "actor_pre_clip_gradient_norm": float(actor_norm.detach().item()),
                "critic_pre_clip_gradient_norm": float(critic_norm.detach().item()),
            })
        if not bool((coverage == 1).all().item()):
            raise RuntimeError("deterministic minibatching did not cover every sample exactly once per epoch")
        if not torch.equal(batch.old_log_probabilities, old_snapshot):
            raise RuntimeError("old-policy log probabilities were mutated")
        self.update_counter += 1
        return {
            "update_counter": self.update_counter,
            "sample_count": size,
            "epoch_count": self.config.epoch_count,
            "coverage": coverage,
            "records": records,
        }

    def critic_only_step(self, privileged: torch.Tensor, returns: torch.Tensor) -> dict[str, float]:
        privileged = validate_critic_input(privileged)
        if not isinstance(returns, torch.Tensor) or returns.shape != (privileged.shape[0], 1):
            raise ValueError("returns must have shape (N,1)")
        if returns.dtype != torch.float32 or returns.device != privileged.device or not bool(torch.isfinite(returns).all().item()):
            raise TypeError("returns must be finite float32 on the critic device")
        actor_before = {name: parameter.detach().clone() for name, parameter in self.actor.named_parameters()}
        self.critic_optimizer.zero_grad(set_to_none=True)
        value_loss = 0.5 * (self.critic(privileged) - returns).square().mean()
        if not bool(torch.isfinite(value_loss).item()):
            raise FloatingPointError("critic-only loss is nonfinite")
        value_loss.backward()
        _gradients_finite(self.critic, "critic")
        norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.maximum_gradient_norm)
        if not bool(torch.isfinite(norm).item()):
            raise FloatingPointError("critic-only pre-clip norm is nonfinite")
        self.critic_optimizer.step()
        _module_finite(self.critic, "critic")
        _optimizer_finite(self.critic_optimizer, "critic")
        if any(not torch.equal(parameter, actor_before[name]) for name, parameter in self.actor.named_parameters()):
            raise RuntimeError("critic-only optimizer step mutated actor parameters")
        return {"value_loss": float(value_loss.detach().item()), "pre_clip_gradient_norm": float(norm.detach().item())}


__all__ = [
    "GAEOutput", "PPOBatch", "PPOConfig", "PPOTrainer", "clipped_surrogate",
    "compute_gae", "deterministic_minibatches", "normalize_advantages",
]
