"""Fail-closed bounded environment-to-PPO rollout adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from instinctlab.task.critic import build_critic_state

from .networks import ACTION_DIM, ACTOR_KEYS, CRITIC_DIM, validate_critic_input, validate_policy
from .ppo import PPOBatch, PPOConfig, PPOTrainer, compute_gae


@dataclass(frozen=True)
class RolloutConfig:
    """Immutable fixed-size rollout boundary."""

    horizon: int
    batch_size: int

    def __post_init__(self) -> None:
        for name in ("horizon", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class RolloutTransition:
    """One exact detached pre-action transition and its terminal-safe values."""

    policy: Mapping[str, torch.Tensor]
    privileged: torch.Tensor
    action: torch.Tensor
    old_log_probability: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    reset_boundary: torch.Tensor
    next_privileged: torch.Tensor
    next_value: torch.Tensor
    timeout_privileged: torch.Tensor
    timeout_value: torch.Tensor
    actor_mean_action: torch.Tensor | None = None


@dataclass(frozen=True)
class RolloutSegment:
    """Validated `[T,N,...]` tensors and their lossless flattened PPO batch."""

    config: RolloutConfig
    transitions: tuple[RolloutTransition, ...]
    policy: Mapping[str, torch.Tensor]
    privileged: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    reset_boundary: torch.Tensor
    next_privileged: torch.Tensor
    next_values: torch.Tensor
    timeout_privileged: torch.Tensor
    timeout_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    batch: PPOBatch


@dataclass(frozen=True)
class RolloutResult:
    """One complete segment plus the exact returned observation for continuation."""

    segment: RolloutSegment
    next_observation: Mapping[str, Any]


def _clone(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    if result.requires_grad or (value.numel() and result.data_ptr() == value.data_ptr()):
        raise RuntimeError("stored rollout tensor must be an independent detached clone")
    return result


def _observation(observation: Mapping[str, Any], config: RolloutConfig) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not isinstance(observation, Mapping) or tuple(observation) != ("policy", "critic"):
        raise ValueError("observation keys and order must be exactly ('policy', 'critic')")
    policy = observation["policy"]
    privileged = observation["critic"]
    lidar, _ = validate_policy(policy)
    privileged = validate_critic_input(privileged)
    if lidar.shape[0] != config.batch_size or privileged.shape[0] != config.batch_size:
        raise ValueError("observation batch does not match rollout configuration")
    if privileged.device != lidar.device:
        raise ValueError("policy and critic observations must share one device")
    return {key: policy[key] for key in ACTOR_KEYS}, privileged


def _column_float(name: str, value: torch.Tensor, config: RolloutConfig, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape == (config.batch_size,):
        value = value.unsqueeze(-1)
    if value.shape != (config.batch_size, 1):
        raise ValueError(f"{name} must have shape (N,) or (N,1)")
    if value.dtype != torch.float32 or value.device != device:
        raise TypeError(f"{name} must use float32 on the rollout device")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")
    return value


def _column_mask(name: str, value: torch.Tensor, config: RolloutConfig, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape == (config.batch_size,):
        value = value.unsqueeze(-1)
    if value.shape != (config.batch_size, 1) or value.dtype != torch.bool or value.device != device:
        raise TypeError(f"{name} must be boolean (N,) or (N,1) on the rollout device")
    return value


def _required_mapping(name: str, value: Any, keys: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(key not in value for key in keys):
        raise ValueError(f"{name} is missing required terminal fields {keys}")
    return value


def _terminal_bootstrap(
    extras: Mapping[str, Any],
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    config: RolloutConfig,
    trainer: PPOTrainer,
) -> tuple[torch.Tensor, torch.Tensor]:
    pure_timeout = truncated & ~terminated
    zeros_state = torch.zeros(config.batch_size, CRITIC_DIM, device=terminated.device, dtype=torch.float32)
    zeros_value = torch.zeros(config.batch_size, 1, device=terminated.device, dtype=torch.float32)
    if not isinstance(extras, Mapping):
        raise TypeError("environment extras must be a mapping")
    pre_reset = _required_mapping("extras.pre_reset", extras.get("pre_reset"), ("reset_mask", "root_state_w", "task"))
    reset_mask = _column_mask("pre_reset.reset_mask", pre_reset["reset_mask"], config, terminated.device)
    if not torch.equal(reset_mask, terminated | truncated):
        raise ValueError("pre-reset mask must equal the exact environment done union")
    task = _required_mapping(
        "extras.pre_reset.task", pre_reset["task"],
        ("root_state_w", "actual_velocity_b", "exact_clearance_m", "collision", "termination"),
    )
    terminal = _required_mapping(
        "extras.pre_reset.task.termination", task["termination"], ("terminated", "truncated", "done")
    )
    task_terminated = _column_mask("terminal.terminated", terminal["terminated"], config, terminated.device)
    task_truncated = _column_mask("terminal.truncated", terminal["truncated"], config, terminated.device)
    task_done = _column_mask("terminal.done", terminal["done"], config, terminated.device)
    if not torch.equal(task_terminated, terminated) or not torch.equal(task_truncated, truncated):
        raise ValueError("pre-reset task cause masks do not match returned masks")
    if not torch.equal(task_done, terminated | truncated):
        raise ValueError("pre-reset task done mask is inconsistent")
    if not bool(pure_timeout.any().item()):
        return zeros_state, zeros_value
    root = pre_reset["root_state_w"]
    if not isinstance(root, torch.Tensor) or not isinstance(task["root_state_w"], torch.Tensor):
        raise TypeError("terminal root states must be tensors")
    if not torch.equal(root, task["root_state_w"]):
        raise ValueError("pre-reset root and task root snapshots are inconsistent")
    built = build_critic_state(
        root_state_w=root,
        actual_velocity_b=task["actual_velocity_b"],
        exact_clearance_m=task["exact_clearance_m"],
        collision=task["collision"],
    ).privileged
    if built.dtype != torch.float32 or built.device != terminated.device:
        raise TypeError("terminal critic state must be float32 on the rollout device")
    with torch.no_grad():
        values = trainer.critic(built)
    selected_state = torch.where(pure_timeout.expand_as(built), built, zeros_state)
    selected_value = torch.where(pure_timeout, values, zeros_value)
    return selected_state, selected_value


def collect_transition(
    environment: Any,
    trainer: PPOTrainer,
    observation: Mapping[str, Any],
    config: RolloutConfig,
) -> tuple[RolloutTransition, Mapping[str, Any]]:
    """Collect one environment step with strict actor/critic separation."""
    if not isinstance(trainer, PPOTrainer) or not isinstance(config, RolloutConfig):
        raise TypeError("trainer and config must use exact rollout types")
    if not callable(getattr(environment, "step", None)):
        raise TypeError("environment must expose a callable step")
    policy, privileged = _observation(observation, config)
    device = privileged.device
    with torch.no_grad():
        action, old_log_probability, _ = trainer.actor.sample(policy)
        value = trainer.critic(privileged)
    if action.shape != (config.batch_size, ACTION_DIM) or action.dtype != torch.float32 or action.device != device:
        raise RuntimeError("sampled action violates the exact environment action contract")
    if action.requires_grad or old_log_probability.requires_grad or value.requires_grad:
        raise RuntimeError("collection outputs must be produced without gradients")
    staged_action = action.detach().clone()
    returned = environment.step(action)
    if not isinstance(returned, tuple) or len(returned) != 5:
        raise ValueError("environment step must return observation, reward, terminated, truncated, extras")
    next_observation, reward, terminated, truncated, extras = returned
    if not torch.equal(action, staged_action):
        raise RuntimeError("environment mutated the sampled action")
    next_policy, next_privileged = _observation(next_observation, config)
    for key in ACTOR_KEYS:
        if next_policy[key].numel() and next_policy[key].data_ptr() == policy[key].data_ptr():
            raise ValueError("returned policy observation aliases the pre-action observation")
    if next_privileged.numel() and next_privileged.data_ptr() == privileged.data_ptr():
        raise ValueError("returned critic observation aliases the pre-action observation")
    reward = _column_float("reward", reward, config, device)
    terminated = _column_mask("terminated", terminated, config, device)
    truncated = _column_mask("truncated", truncated, config, device)
    reset_boundary = terminated | truncated
    timeout_privileged, timeout_value = _terminal_bootstrap(extras, terminated, truncated, config, trainer)
    with torch.no_grad():
        next_value = trainer.critic(next_privileged)
    transition = RolloutTransition(
        policy={key: _clone(policy[key]) for key in ACTOR_KEYS},
        privileged=_clone(privileged), action=_clone(staged_action),
        old_log_probability=_clone(old_log_probability), value=_clone(value), reward=_clone(reward),
        terminated=_clone(terminated), truncated=_clone(truncated), reset_boundary=_clone(reset_boundary),
        next_privileged=_clone(next_privileged), next_value=_clone(next_value),
        timeout_privileged=_clone(timeout_privileged), timeout_value=_clone(timeout_value),
    )
    continuation = {
        "policy": {key: _clone(next_policy[key]) for key in ACTOR_KEYS},
        "critic": _clone(next_privileged),
    }
    return transition, continuation


def assemble_rollout(
    transitions: Sequence[RolloutTransition], config: RolloutConfig, ppo_config: PPOConfig
) -> RolloutSegment:
    """Stack `[T,N]`, compute terminal-safe GAE, and flatten in time-major order."""
    if not isinstance(config, RolloutConfig) or not isinstance(ppo_config, PPOConfig):
        raise TypeError("config and ppo_config must use exact immutable types")
    if not isinstance(transitions, Sequence) or len(transitions) != config.horizon:
        raise ValueError("rollout must contain exactly the configured horizon")
    items = tuple(transitions)
    if any(not isinstance(item, RolloutTransition) for item in items):
        raise TypeError("every rollout item must be RolloutTransition")
    policy = {key: torch.stack([item.policy[key] for item in items]) for key in ACTOR_KEYS}
    names = (
        "privileged", "actions", "old_log_probabilities", "values", "rewards", "terminated",
        "truncated", "reset_boundary", "next_privileged", "next_values", "timeout_privileged",
        "timeout_values",
    )
    sources = {
        "privileged": "privileged", "actions": "action", "old_log_probabilities": "old_log_probability",
        "values": "value", "rewards": "reward", "terminated": "terminated", "truncated": "truncated",
        "reset_boundary": "reset_boundary", "next_privileged": "next_privileged",
        "next_values": "next_value", "timeout_privileged": "timeout_privileged",
        "timeout_values": "timeout_value",
    }
    stacked = {name: torch.stack([getattr(item, sources[name]) for item in items]) for name in names}
    if policy["lidar_grid"].shape[:2] != (config.horizon, config.batch_size):
        raise ValueError("stored policy trajectory has inconsistent T/N dimensions")
    if stacked["privileged"].shape != (config.horizon, config.batch_size, CRITIC_DIM):
        raise ValueError("stored critic trajectory has inconsistent T/N dimensions")
    if not torch.equal(stacked["reset_boundary"], stacked["terminated"] | stacked["truncated"]):
        raise ValueError("stored reset boundary must equal exact done union")
    gae = compute_gae(
        stacked["rewards"], stacked["values"], stacked["next_values"], stacked["terminated"],
        stacked["truncated"], stacked["reset_boundary"],
        timeout_bootstrap_values=stacked["timeout_values"], gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
    )
    sample_count = config.horizon * config.batch_size
    flat_policy = {key: value.reshape(sample_count, *value.shape[2:]) for key, value in policy.items()}
    batch = PPOBatch(
        policy=flat_policy,
        privileged=stacked["privileged"].reshape(sample_count, CRITIC_DIM),
        actions=stacked["actions"].reshape(sample_count, ACTION_DIM),
        old_log_probabilities=stacked["old_log_probabilities"].reshape(sample_count, 1),
        advantages=gae.advantages.reshape(sample_count, 1), returns=gae.returns.reshape(sample_count, 1),
    )
    return RolloutSegment(
        config=config, transitions=items, policy=policy, advantages=gae.advantages, returns=gae.returns,
        batch=batch, **stacked,
    )


def collect_rollout(
    environment: Any,
    trainer: PPOTrainer,
    observation: Mapping[str, Any],
    config: RolloutConfig,
) -> RolloutResult:
    """Collect exactly one fixed horizon without performing an optimizer update."""
    current = observation
    transitions = []
    for _ in range(config.horizon):
        transition, current = collect_transition(environment, trainer, current, config)
        transitions.append(transition)
    return RolloutResult(assemble_rollout(transitions, config, trainer.config), current)


__all__ = [
    "RolloutConfig", "RolloutResult", "RolloutSegment", "RolloutTransition",
    "assemble_rollout", "collect_rollout", "collect_transition",
]
