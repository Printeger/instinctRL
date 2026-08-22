#!/usr/bin/env python3
"""Two-segment bounded real-environment PPO integration smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from isaaclab.app import AppLauncher

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--input", type=Path, required=True)
PARSER.add_argument("--output", type=Path, required=True)
PARSER.add_argument("--checkpoint", type=Path, required=True)
ARGS = PARSER.parse_args()
APP_LAUNCHER = AppLauncher(headless=True, enable_cameras=False)
SIMULATION_APP = APP_LAUNCHER.app

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instinctlab.envs import TaslabDirectEnv, TaslabDirectEnvCfg
from instinctlab.training import (
    HistoryActor, NetworkConfig, PPOConfig, PPOTrainer, PrivilegedCritic,
    RolloutConfig, collect_rollout, load_checkpoint, save_checkpoint,
)

NUM_ENVS = 2
HORIZON = 4
SEGMENTS = 2
POLICY_STEPS = HORIZON * SEGMENTS
PHYSICS_STEPS = 160


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {key: _cpu(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)): return type(value)(_cpu(child) for child in value)
    return value


def _tensor_summary(module: torch.nn.Module) -> dict[str, Any]:
    digest = hashlib.sha256(); count = 0; square_sum = 0.0; maximum = 0.0; layers = []
    for name, value in module.state_dict().items():
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode()); digest.update(cpu.numpy().tobytes())
        count += cpu.numel()
        if cpu.is_floating_point():
            square_sum += float(cpu.double().square().sum().item())
            maximum = max(maximum, float(cpu.abs().max().item()) if cpu.numel() else 0.0)
        layers.append((name, tuple(cpu.shape), str(cpu.dtype)))
    return {"sha256": digest.hexdigest(), "element_count": count, "square_sum": square_sum,
            "maximum_absolute_value": maximum, "layers": layers}


def _optimizer_summary(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict(); records = []
    for index in sorted(state["state"]):
        item = state["state"][index]
        records.append({"index": index, "step": float(item["step"].item()),
                        "exp_avg_sum": float(item["exp_avg"].double().sum().item()),
                        "exp_avg_sq_sum": float(item["exp_avg_sq"].double().sum().item()),
                        "element_count": item["exp_avg"].numel()})
    return {"parameter_groups": len(state["param_groups"]), "state_count": len(records), "records": records}


def _nested_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right): return False
    if isinstance(left, torch.Tensor): return torch.equal(left, right)
    if isinstance(left, dict): return list(left) == list(right) and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)): return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _segment_tree(segment) -> dict[str, Any]:
    return _cpu({
        "policy": segment.policy, "privileged": segment.privileged, "actions": segment.actions,
        "old_log_probabilities": segment.old_log_probabilities, "values": segment.values,
        "rewards": segment.rewards, "terminated": segment.terminated, "truncated": segment.truncated,
        "reset_boundary": segment.reset_boundary, "next_privileged": segment.next_privileged,
        "next_values": segment.next_values, "timeout_privileged": segment.timeout_privileged,
        "timeout_values": segment.timeout_values, "advantages": segment.advantages, "returns": segment.returns,
        "flat_policy": segment.batch.policy, "flat_privileged": segment.batch.privileged,
        "flat_actions": segment.batch.actions, "flat_old_log_probabilities": segment.batch.old_log_probabilities,
        "flat_advantages": segment.batch.advantages, "flat_returns": segment.batch.returns,
    })


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must select deterministic cuBLAS")
    fixture = torch.load(ARGS.input, map_location="cpu", weights_only=True)
    if not isinstance(fixture, dict) or tuple(fixture) != (
        "schema_version", "seed", "environment_seed", "num_envs", "horizon", "segments", "forced_timeout"
    ):
        raise ValueError("learning-pipeline fixture schema/order mismatch")
    expected = {"schema_version": "m6-learning-pipeline-fixture-1", "num_envs": NUM_ENVS,
                "horizon": HORIZON, "segments": SEGMENTS, "forced_timeout": {"segment": 1, "step": 1, "env": 0}}
    for key, value in expected.items():
        if fixture[key] != value: raise ValueError(f"fixture field {key} mismatch")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(fixture["seed"]); torch.cuda.manual_seed_all(fixture["seed"])
    torch.cuda.reset_peak_memory_stats()
    cfg = TaslabDirectEnvCfg(); cfg.scene.num_envs = NUM_ENVS
    environment = TaslabDirectEnv(cfg)
    observation, _ = environment.reset(seed=fixture["environment_seed"])
    network_config = NetworkConfig()
    ppo_config = PPOConfig(minibatch_size=2, epoch_count=1, permutation_seed=1701)
    trainer = PPOTrainer(HistoryActor(network_config).to(environment.device),
                         PrivilegedCritic(network_config).to(environment.device), ppo_config)
    rollout_config = RolloutConfig(HORIZON, NUM_ENVS)
    initial_observation = _cpu(observation)

    # Explicit test-fixture setup only: the next accepted public step reaches pure timeout for env0.
    environment.episode_length_buf[0] = environment.max_episode_length - 1
    first = collect_rollout(environment, trainer, observation, rollout_config)
    first_timeout = first.segment.truncated & ~first.segment.terminated
    if first_timeout.sum().item() != 1 or not bool(first_timeout[0, 0, 0].item()):
        raise AssertionError("first public step did not produce the exact forced pure-timeout row")
    if int(environment.episode_length_buf[0].item()) != HORIZON - 1:
        raise AssertionError("post-timeout live environment did not continue for three finite steps")
    torch.cuda.synchronize()
    first_metrics = trainer.update(first.segment.batch)
    if trainer.update_counter != 1 or first_metrics["sample_count"] != HORIZON * NUM_ENVS:
        raise AssertionError("first segment must produce one exact eight-sample PPO update")
    with torch.no_grad():
        expected_actor = trainer.actor(first.next_observation["policy"])
        expected_log = trainer.actor.log_probability_from_output(expected_actor, expected_actor.mean_action)
        expected_critic = trainer.critic(first.next_observation["critic"])
    actor_optimizer = _cpu(trainer.actor_optimizer.state_dict())
    critic_optimizer = _cpu(trainer.critic_optimizer.state_dict())
    ARGS.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(ARGS.checkpoint.resolve(), trainer)
    saved_cpu_rng = torch.get_rng_state().clone()
    saved_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]

    restored = PPOTrainer(HistoryActor(network_config).to(environment.device),
                          PrivilegedCritic(network_config).to(environment.device), ppo_config)
    load_checkpoint(ARGS.checkpoint.resolve(), restored)
    with torch.no_grad():
        actual_actor = restored.actor(first.next_observation["policy"])
        actual_log = restored.actor.log_probability_from_output(actual_actor, actual_actor.mean_action)
        actual_critic = restored.critic(first.next_observation["critic"])
    reload_exact = {
        "alpha": torch.equal(expected_actor.alpha, actual_actor.alpha),
        "beta": torch.equal(expected_actor.beta, actual_actor.beta),
        "mean_action": torch.equal(expected_actor.mean_action, actual_actor.mean_action),
        "log_probability": torch.equal(expected_log, actual_log),
        "critic": torch.equal(expected_critic, actual_critic),
        "actor_optimizer": _nested_equal(actor_optimizer, _cpu(restored.actor_optimizer.state_dict())),
        "critic_optimizer": _nested_equal(critic_optimizer, _cpu(restored.critic_optimizer.state_dict())),
        "update_counter": restored.update_counter == 1,
        "cpu_rng": torch.equal(saved_cpu_rng, torch.get_rng_state()),
        "cuda_rng": _nested_equal(saved_cuda_rng, torch.cuda.get_rng_state_all()),
    }
    if not all(reload_exact.values()): raise AssertionError(f"checkpoint boundary mismatch: {reload_exact}")

    second = collect_rollout(environment, restored, first.next_observation, rollout_config)
    torch.cuda.synchronize()
    second_metrics = restored.update(second.segment.batch)
    if restored.update_counter != 2 or second_metrics["sample_count"] != HORIZON * NUM_ENVS:
        raise AssertionError("second segment must produce the second exact eight-sample PPO update")
    if int(environment.common_step_counter) != POLICY_STEPS or int(environment._sim_step_counter) != PHYSICS_STEPS:
        raise AssertionError("real environment policy/physics-step count mismatch")
    tensors = (
        first.segment.actions, first.segment.old_log_probabilities, first.segment.advantages,
        first.segment.returns, second.segment.actions, second.segment.old_log_probabilities,
        second.segment.advantages, second.segment.returns,
    )
    if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
        raise FloatingPointError("bounded learning pipeline produced nonfinite tensors")
    if sum(item.config.horizon * item.config.batch_size for item in (first.segment, second.segment)) != 16:
        raise AssertionError("bounded pipeline sample count mismatch")

    result = {
        "schema_version": "m6-learning-pipeline-smoke-1",
        "fixture": fixture,
        "contract": {"num_envs": NUM_ENVS, "horizon": HORIZON, "segments": SEGMENTS,
                     "policy_steps": POLICY_STEPS, "physics_steps": PHYSICS_STEPS,
                     "sample_count": 16, "update_calls": 2, "checkpoint_reloads": 1,
                     "actor_keys": ("lidar_grid", "state_vec"), "critic_dimension": 18,
                     "action_dimension": 4, "network_config": network_config.manifest(),
                     "ppo_config": ppo_config.manifest()},
        "initial_observation": initial_observation,
        "segments": [
            {"trajectory": _segment_tree(first.segment), "update": _cpu(first_metrics)},
            {"trajectory": _segment_tree(second.segment), "update": _cpu(second_metrics)},
        ],
        "checkpoint_boundary": {"reload_exact": reload_exact, "post_reset_observation": _cpu(first.next_observation),
                                "actor_optimizer": actor_optimizer, "critic_optimizer": critic_optimizer},
        "final": {"observation": _cpu(second.next_observation), "episode_length": _cpu(environment.episode_length_buf),
                  "actor": _tensor_summary(restored.actor), "critic": _tensor_summary(restored.critic),
                  "actor_optimizer": _optimizer_summary(restored.actor_optimizer),
                  "critic_optimizer": _optimizer_summary(restored.critic_optimizer)},
        "resource": {"peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()},
        "gates": {"pure_timeout_count": int(first_timeout.sum().item()),
                  "post_reset_continuation_steps": HORIZON - 1, "all_finite": True,
                  "checkpoint_exact": True, "exact_dimensions": True, "status": "PASS"},
    }
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, ARGS.output)
    environment.close(); SIMULATION_APP.close()
    print(json.dumps({"status": "PASS", "num_envs": NUM_ENVS, "policy_steps": POLICY_STEPS,
                      "physics_steps": PHYSICS_STEPS, "samples": 16, "updates": 2,
                      "pure_timeouts": 1, "peak_cuda_memory_bytes": result["resource"]["peak_cuda_memory_bytes"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
