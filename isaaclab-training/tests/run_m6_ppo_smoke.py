#!/usr/bin/env python3
"""Two-update CUDA-only synthetic smoke for the M6 PPO tensor core."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from instinctlab.training import (
    HistoryActor, NetworkConfig, PPOBatch, PPOConfig, PPOTrainer, PrivilegedCritic,
    compute_gae, load_checkpoint, save_checkpoint,
)


def _tensor_hashes(module: torch.nn.Module) -> dict[str, object]:
    digest = hashlib.sha256()
    count = 0
    square_sum = 0.0
    maximum = 0.0
    layers = []
    for name, value in module.state_dict().items():
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(cpu.numpy().tobytes())
        count += cpu.numel()
        if cpu.is_floating_point():
            square_sum += float(cpu.double().square().sum().item())
            maximum = max(maximum, float(cpu.abs().max().item()) if cpu.numel() else 0.0)
        layers.append((name, tuple(cpu.shape), str(cpu.dtype)))
    return {"sha256": digest.hexdigest(), "element_count": count, "square_sum": square_sum,
            "maximum_absolute_value": maximum, "layers": layers}


def _optimizer_summary(optimizer: torch.optim.Optimizer) -> dict[str, object]:
    state = optimizer.state_dict()
    records = []
    for index in sorted(state["state"]):
        item = state["state"][index]
        records.append({
            "index": index,
            "step": float(item["step"].item()),
            "exp_avg_sum": float(item["exp_avg"].double().sum().item()),
            "exp_avg_sq_sum": float(item["exp_avg_sq"].double().sum().item()),
            "element_count": item["exp_avg"].numel(),
        })
    return {"parameter_groups": len(state["param_groups"]), "state_count": len(records), "records": records}


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu(child) for child in value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must select a deterministic cuBLAS workspace")
    if not torch.cuda.is_available():
        raise RuntimeError("M6 PPO smoke requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260822)
    torch.cuda.manual_seed_all(20260822)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    source = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(source, dict) or tuple(source) != (
        "schema_version", "policy", "privileged", "next_privileged", "terminal_privileged",
        "rewards", "terminated", "truncated", "reset_boundary",
    ):
        raise ValueError("synthetic input tree schema/order mismatch")
    if source["schema_version"] != "m6-ppo-synthetic-1":
        raise ValueError("synthetic input schema version mismatch")
    data = _cpu(source)
    policy = {key: value.to(device) for key, value in source["policy"].items()}
    privileged = source["privileged"].to(device)
    next_privileged = source["next_privileged"].to(device)
    terminal_privileged = source["terminal_privileged"].to(device)
    rewards = source["rewards"].to(device)
    terminated = source["terminated"].to(device)
    truncated = source["truncated"].to(device)
    reset_boundary = source["reset_boundary"].to(device)

    network_config = NetworkConfig()
    ppo_config = PPOConfig(minibatch_size=2, epoch_count=1, permutation_seed=1701)
    actor = HistoryActor(network_config).to(device)
    critic = PrivilegedCritic(network_config).to(device)
    trainer = PPOTrainer(actor, critic, ppo_config)
    with torch.no_grad():
        initial_output = actor(policy)
        action = initial_output.mean_action.clone()
        old_log_probability = actor.log_probability_from_output(initial_output, action).detach().clone()
        entropy = actor.entropy_from_output(initial_output)
        value = critic(privileged)
        next_value = critic(next_privileged)
        terminal_value = critic(terminal_privileged)
    trajectory_shape = rewards.shape
    gae = compute_gae(
        rewards,
        value.reshape(trajectory_shape),
        next_value.reshape(trajectory_shape),
        terminated,
        truncated,
        reset_boundary,
        timeout_bootstrap_values=terminal_value.reshape(trajectory_shape),
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
    )
    batch = PPOBatch(policy, privileged, action, old_log_probability, gae.advantages.reshape(-1, 1), gae.returns.reshape(-1, 1))
    update_metrics = [trainer.update(batch), trainer.update(batch)]
    if trainer.update_counter != 2:
        raise AssertionError("the bounded smoke must perform exactly two update calls")
    with torch.no_grad():
        post_output = actor(policy)
        post_log_probability = actor.log_probability_from_output(post_output, action)
        post_entropy = actor.entropy_from_output(post_output)
        post_value = critic(privileged)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.checkpoint.resolve(), trainer)

    torch.manual_seed(99)
    restored = PPOTrainer(HistoryActor(network_config).to(device), PrivilegedCritic(network_config).to(device), ppo_config)
    load_checkpoint(args.checkpoint.resolve(), restored)
    with torch.no_grad():
        reload_output = restored.actor(policy)
        reload_log_probability = restored.actor.log_probability_from_output(reload_output, action)
        reload_value = restored.critic(privileged)
    reload_exact = {
        "alpha": torch.equal(post_output.alpha, reload_output.alpha),
        "beta": torch.equal(post_output.beta, reload_output.beta),
        "mean_action": torch.equal(post_output.mean_action, reload_output.mean_action),
        "log_probability": torch.equal(post_log_probability, reload_log_probability),
        "critic_value": torch.equal(post_value, reload_value),
        "update_counter": restored.update_counter == 2,
    }
    if not all(reload_exact.values()):
        raise AssertionError("checkpoint reload did not reproduce exact actor/critic outputs")
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in (
        initial_output.alpha, initial_output.beta, action, old_log_probability, entropy, value,
        gae.advantages, gae.returns, post_output.alpha, post_output.beta,
        post_output.mean_action, post_log_probability, post_entropy, post_value,
    )):
        raise AssertionError("nonfinite tensor in bounded PPO smoke")

    result = {
        "schema_version": "m6-ppo-smoke-1",
        "contract": {
            "device": "cuda:0", "batch_size": privileged.shape[0], "trajectory_shape": tuple(trajectory_shape),
            "update_calls": 2, "actor_keys": tuple(policy), "critic_dimension": privileged.shape[1],
            "action_dimension": action.shape[1], "network_config": network_config.manifest(),
            "ppo_config": ppo_config.manifest(), "actor_manifest": actor.parameter_manifest(),
            "critic_manifest": critic.parameter_manifest(),
        },
        "inputs": data,
        "initial": {
            "alpha": _cpu(initial_output.alpha), "beta": _cpu(initial_output.beta),
            "mean_action": _cpu(initial_output.mean_action), "action": _cpu(action),
            "old_log_probability": _cpu(old_log_probability), "entropy": _cpu(entropy),
            "value": _cpu(value), "next_value": _cpu(next_value), "terminal_value": _cpu(terminal_value),
            "advantages": _cpu(gae.advantages), "returns": _cpu(gae.returns),
        },
        "updates": _cpu(update_metrics),
        "post_update": {
            "alpha": _cpu(post_output.alpha), "beta": _cpu(post_output.beta),
            "mean_action": _cpu(post_output.mean_action), "log_probability": _cpu(post_log_probability),
            "entropy": _cpu(post_entropy), "critic_value": _cpu(post_value),
            "actor_parameters": _tensor_hashes(actor), "critic_parameters": _tensor_hashes(critic),
            "actor_optimizer": _optimizer_summary(trainer.actor_optimizer),
            "critic_optimizer": _optimizer_summary(trainer.critic_optimizer),
        },
        "post_reload": {
            "alpha": _cpu(reload_output.alpha), "beta": _cpu(reload_output.beta),
            "mean_action": _cpu(reload_output.mean_action), "log_probability": _cpu(reload_log_probability),
            "critic_value": _cpu(reload_value), "exact": reload_exact,
            "actor_parameters": _tensor_hashes(restored.actor), "critic_parameters": _tensor_hashes(restored.critic),
            "actor_optimizer": _optimizer_summary(restored.actor_optimizer),
            "critic_optimizer": _optimizer_summary(restored.critic_optimizer),
        },
        "resource": {"peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()},
        "gates": {"all_finite": True, "checkpoint_exact": True, "two_updates": True, "status": "PASS"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(json.dumps({"status": "PASS", "updates": 2, "batch": privileged.shape[0],
                      "actor_parameters": result["contract"]["actor_manifest"]["parameter_count"],
                      "critic_parameters": result["contract"]["critic_manifest"]["parameter_count"],
                      "peak_cuda_memory_bytes": result["resource"]["peak_cuda_memory_bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
