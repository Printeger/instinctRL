#!/usr/bin/env python3
"""Complete deterministic pure-CUDA M6 task-contract smoke."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch

from instinctlab.control.pipeline import ControlCoreConfig, ControlCorePipeline
from instinctlab.observations.audit import audit_actor_observation
from instinctlab.task import build_critic_state, compute_reward, compute_termination

DEVICE = "cuda:0"
DTYPE = torch.float32
FIXED_SEED = 6001
BATCH = 4


def _tensor_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if is_dataclass(value):
        return {field.name: _tensor_tree(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _tensor_tree(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_tensor_tree(child) for child in value)
    return value


def _actor() -> dict[str, torch.Tensor]:
    lidar = torch.empty(BATCH, 12, 59, 360, device=DEVICE, dtype=DTYPE)
    lidar[:, 0::3] = 40.0
    lidar[:, 1::3] = 1.0
    lidar[:, 2::3] = 1.0
    state = torch.zeros(BATCH, 52, device=DEVICE, dtype=DTYPE)
    for offset in range(0, 52, 13):
        state[:, offset + 5] = 1.0
    commands = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.2, 0.0], [0.0, 0.0, 0.0]],
        device=DEVICE, dtype=DTYPE,
    )
    state[:, -7:-4] = commands
    return {"lidar_grid": lidar, "state_vec": state}


def _control(actor: dict[str, torch.Tensor]):
    pipeline = ControlCorePipeline(ControlCoreConfig(use_b0=True))
    zeros = torch.zeros(BATCH, dtype=torch.bool, device=DEVICE)
    action = torch.zeros(BATCH, 4, device=DEVICE, dtype=DTYPE)
    quaternion = torch.zeros(BATCH, 4, device=DEVICE, dtype=DTYPE)
    quaternion[:, 0] = 1.0
    return pipeline(actor["lidar_grid"], actor["state_vec"], action, 0.1, zeros, zeros, quaternion)


def _reward_inputs(control, privileged_variant: int) -> dict[str, torch.Tensor]:
    command = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.2, 0.0], [0.0, 0.0, 0.0]],
        device=DEVICE, dtype=DTYPE,
    )
    actual = torch.tensor(
        [[0.7, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.0, 0.0, 0.0]],
        device=DEVICE, dtype=DTYPE,
    )
    clearance = torch.tensor([[2.0], [0.9], [0.6], [2.0]], device=DEVICE, dtype=DTYPE)
    collision = torch.tensor([[False], [False], [False], [False]], device=DEVICE)
    if privileged_variant == 1:
        actual = actual + torch.tensor([[0.3, 0.1, 0.0]], device=DEVICE)
        clearance = clearance.clone()
        clearance[2, 0] = 0.1
        collision = collision.clone()
        collision[2, 0] = True
    return {
        "v_cmd_b": command,
        "v_gov_b": control.v_gov_b,
        "v_final_b": control.v_final_b,
        "previous_issued_v_final_b": torch.zeros_like(command),
        "actual_velocity_b": actual,
        "anchor_loss": control.anchor_metrics["huber_loss"],
        "anchor_active": control.anchor_metrics["active"],
        "anchor_usable_fraction": control.anchor_metrics["usable_fraction"],
        "ics_beta": control.ics_metrics["beta"],
        "ics_emergency": control.ics_metrics["emergency"],
        "exact_clearance_m": clearance,
        "collision": collision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 is required by the M6 smoke contract")
    actor = _actor()
    audit_actor_observation(actor)
    control_a = _control(actor)
    control_b = _control(actor)
    rewards_a = compute_reward(**_reward_inputs(control_a, 0))
    rewards_b = compute_reward(**_reward_inputs(control_b, 1))
    root_a = torch.zeros(BATCH, 13, device=DEVICE, dtype=DTYPE)
    root_a[:, 2] = 2.0
    root_a[:, 3] = 1.0
    root_b = root_a.clone()
    root_b[:, 0] = 3.0
    inputs_a, inputs_b = _reward_inputs(control_a, 0), _reward_inputs(control_b, 1)
    critic_a = build_critic_state(
        root_state_w=root_a, actual_velocity_b=inputs_a["actual_velocity_b"],
        exact_clearance_m=inputs_a["exact_clearance_m"], collision=inputs_a["collision"],
    )
    critic_b = build_critic_state(
        root_state_w=root_b, actual_velocity_b=inputs_b["actual_velocity_b"],
        exact_clearance_m=inputs_b["exact_clearance_m"], collision=inputs_b["collision"],
    )
    termination = compute_termination(
        height_w_m=torch.tensor([[2.0], [0.4], [4.1], [2.0]], device=DEVICE),
        collision=inputs_b["collision"],
        elapsed_steps=torch.tensor([[1], [1], [1000], [1000]], device=DEVICE, dtype=torch.long),
    )
    actor_equal = all(torch.equal(actor[key], actor[key].clone()) for key in actor)
    control_equal = all(
        torch.equal(getattr(control_a, name), getattr(control_b, name))
        for name in ("alpha", "v_corr_b", "v_gov_b", "v_final_b", "v_final_w")
    )
    gates = {
        "actor_equal_under_privilege": actor_equal,
        "control_equal_under_privilege": control_equal,
        "reward_changes_under_privilege": not torch.equal(rewards_a.total, rewards_b.total),
        "critic_changes_under_privilege": not torch.equal(critic_a.privileged, critic_b.privileged),
        "all_finite": all(torch.isfinite(value).all().item() for value in rewards_a.components.values())
        and torch.isfinite(critic_a.privileged).all().item(),
    }
    gates["all_gates_pass"] = all(gates.values())
    payload = {
        "contract": {"schema": "m6-task-contract-smoke-1", "device": DEVICE, "dtype": str(DTYPE), "seed": FIXED_SEED, "batch": BATCH},
        "actor": actor,
        "control_a": control_a,
        "control_b": control_b,
        "reward_inputs_a": inputs_a,
        "reward_inputs_b": inputs_b,
        "rewards_a": rewards_a,
        "rewards_b": rewards_b,
        "critic_a": critic_a,
        "critic_b": critic_b,
        "termination": termination,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_tensor_tree(payload), args.output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps({"contract": payload["contract"], "gates": gates}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if gates["all_gates_pass"] else "FAIL", "batch": BATCH}, sort_keys=True))
    raise SystemExit(0 if gates["all_gates_pass"] else 2)


if __name__ == "__main__":
    main()
