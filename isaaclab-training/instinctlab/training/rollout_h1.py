"""Command-conditioned 21-D rollout and terminal-bootstrap wiring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from instinctlab.task.critic_h1 import CRITIC_H1_DIMENSION
from .networks import ACTION_DIM, ACTOR_KEYS, validate_policy
from .networks_h1 import PrivilegedCriticH1, validate_critic_h1_input
from .ppo import PPOBatch, PPOTrainer, compute_gae
from .rollout import RolloutConfig, RolloutResult, RolloutSegment, RolloutTransition, _clone, _column_float, _column_mask


def _observation(value: Mapping[str, Any], config: RolloutConfig) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not isinstance(value, Mapping) or tuple(value) != ("policy", "critic"):
        raise ValueError("observation keys must be exactly policy, critic")
    lidar, _ = validate_policy(value["policy"])
    critic = validate_critic_h1_input(value["critic"])
    if lidar.shape[0] != config.batch_size or critic.shape[0] != config.batch_size or lidar.device != critic.device:
        raise ValueError("H1 observation batch/device mismatch")
    return {key: value["policy"][key] for key in ACTOR_KEYS}, critic


def collect_h1_transition(environment: Any, trainer: PPOTrainer, observation: Mapping[str, Any], config: RolloutConfig):
    if (getattr(environment, '_r6_measurements', None) or {}).get('nontraining_baseline'):
        raise ValueError('evaluation-only reward mode cannot enter PPO collection')
    if not isinstance(trainer, PPOTrainer) or not isinstance(trainer.critic, PrivilegedCriticH1):
        raise TypeError("H1 rollout requires a PrivilegedCriticH1 trainer")
    policy, critic = _observation(observation, config); device = critic.device
    with torch.no_grad():
        distribution, actor_output = trainer.actor.distribution(policy)
        action = distribution.sample()
        log_probability = trainer.actor.log_probability_from_output(actor_output, action)
        value = trainer.critic(critic)
    staged = action.detach().clone()
    returned = environment.step(action)
    if not isinstance(returned, tuple) or len(returned) != 5:
        raise ValueError("environment step must return the Gymnasium five-tuple")
    next_observation, reward, terminated, truncated, extras = returned
    if isinstance(extras, Mapping):
        diagnostics = extras.get('reward_diagnostics', {})
        if extras.get('reward_valid') is False or (
            isinstance(diagnostics, Mapping) and diagnostics.get('learning_eligible') is False
        ):
            raise ValueError('unavailable evaluation reward cannot enter PPO collection')
    if not torch.equal(action, staged):
        raise RuntimeError("environment mutated action")
    next_policy, next_critic = _observation(next_observation, config)
    reward = _column_float("reward", reward, config, device)
    terminated = _column_mask("terminated", terminated, config, device)
    truncated = _column_mask("truncated", truncated, config, device)
    reset = terminated | truncated; pure_timeout = truncated & ~terminated
    timeout_critic = extras.get("pre_reset", {}).get("critic_h1") if isinstance(extras, Mapping) else None
    if bool(pure_timeout.any().item()):
        timeout_critic = validate_critic_h1_input(timeout_critic)
        if timeout_critic.shape[0] != config.batch_size:
            raise ValueError("terminal critic batch mismatch")
        with torch.no_grad(): timeout_value = trainer.critic(timeout_critic)
    else:
        timeout_critic = torch.zeros(config.batch_size, CRITIC_H1_DIMENSION, dtype=torch.float32, device=device)
        timeout_value = torch.zeros(config.batch_size, 1, dtype=torch.float32, device=device)
    timeout_critic = torch.where(pure_timeout.expand_as(timeout_critic), timeout_critic, torch.zeros_like(timeout_critic))
    timeout_value = torch.where(pure_timeout, timeout_value, torch.zeros_like(timeout_value))
    with torch.no_grad(): next_value = trainer.critic(next_critic)
    transition = RolloutTransition(
        policy={**{key: _clone(policy[key]) for key in ACTOR_KEYS}, "_actor_output_mean": _clone(actor_output.mean_action)},
        privileged=_clone(critic), action=_clone(staged),
        old_log_probability=_clone(log_probability), value=_clone(value), reward=_clone(reward),
        terminated=_clone(terminated), truncated=_clone(truncated), reset_boundary=_clone(reset),
        next_privileged=_clone(next_critic), next_value=_clone(next_value),
        timeout_privileged=_clone(timeout_critic), timeout_value=_clone(timeout_value),
    )
    return transition, {"policy": {key: _clone(next_policy[key]) for key in ACTOR_KEYS}, "critic": _clone(next_critic)}


def assemble_h1_rollout(transitions: Sequence[RolloutTransition], config: RolloutConfig, ppo_config: Any) -> RolloutSegment:
    if len(transitions) != config.horizon:
        raise ValueError("rollout must contain exactly the configured horizon")
    items = tuple(transitions)
    policy = {key: torch.stack([item.policy[key] for item in items]) for key in ACTOR_KEYS}
    mapping = {
        "privileged": "privileged", "actions": "action", "old_log_probabilities": "old_log_probability",
        "values": "value", "rewards": "reward", "terminated": "terminated", "truncated": "truncated",
        "reset_boundary": "reset_boundary", "next_privileged": "next_privileged", "next_values": "next_value",
        "timeout_privileged": "timeout_privileged", "timeout_values": "timeout_value",
    }
    stacked = {key: torch.stack([getattr(item, attr) for item in items]) for key, attr in mapping.items()}
    if stacked["privileged"].shape != (config.horizon, config.batch_size, CRITIC_H1_DIMENSION):
        raise ValueError("H1 critic trajectory shape mismatch")
    gae = compute_gae(stacked["rewards"], stacked["values"], stacked["next_values"], stacked["terminated"], stacked["truncated"], stacked["reset_boundary"], timeout_bootstrap_values=stacked["timeout_values"], gamma=ppo_config.gamma, gae_lambda=ppo_config.gae_lambda)
    count = config.horizon * config.batch_size
    batch = PPOBatch(
        policy={key: value.reshape(count, *value.shape[2:]) for key, value in policy.items()},
        privileged=stacked["privileged"].reshape(count, CRITIC_H1_DIMENSION),
        actions=stacked["actions"].reshape(count, ACTION_DIM), old_log_probabilities=stacked["old_log_probabilities"].reshape(count, 1),
        advantages=gae.advantages.reshape(count, 1), returns=gae.returns.reshape(count, 1),
    )
    return RolloutSegment(config=config, transitions=items, policy=policy, advantages=gae.advantages, returns=gae.returns, batch=batch, **stacked)


def collect_h1_rollout(environment: Any, trainer: PPOTrainer, observation: Mapping[str, Any], config: RolloutConfig) -> RolloutResult:
    current = observation; transitions = []
    for _ in range(config.horizon):
        transition, current = collect_h1_transition(environment, trainer, current, config); transitions.append(transition)
    return RolloutResult(assemble_h1_rollout(transitions, config, trainer.config), current)


__all__ = ["assemble_h1_rollout", "collect_h1_rollout", "collect_h1_transition"]
