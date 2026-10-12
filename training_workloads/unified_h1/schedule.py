"""Exact global curriculum clock, command roles and reset distribution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

NUM_ENVS = 32
ROLLOUT_HORIZON = 16
WARMUP_STEPS = 1024
ROLE_BLOCK_STEPS = 128
R4_STEPS = 4096
R5_MAX_STEPS = 32768
DIRECTIONS = (
    (0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, -0.2, 0.0),
    (0.15, 0.15, 0.0), (-0.15, 0.15, 0.0), (0.15, -0.15, 0.0), (-0.15, -0.15, 0.0),
)


@dataclass(frozen=True)
class StepPlan:
    global_policy_step: int
    command_b: torch.Tensor
    next_command_b: torch.Tensor
    zero_mask: torch.Tensor
    nonzero_mask: torch.Tensor
    block: int | None


def command_at(global_policy_step: int, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(global_policy_step, bool) or not isinstance(global_policy_step, int) or global_policy_step < 0:
        raise ValueError("global_policy_step must be a nonnegative integer")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("schedule dtype must be float32 or float64")
    command = torch.zeros(NUM_ENVS, 3, device=device, dtype=dtype)
    if global_policy_step < WARMUP_STEPS:
        return command
    block = (global_policy_step - WARMUP_STEPS) // ROLE_BLOCK_STEPS
    nonzero_group = block % 2
    direction_round = block // 2
    directions = torch.tensor(DIRECTIONS, device=device, dtype=dtype)
    envs = torch.arange(NUM_ENVS, device=device)
    selected = (envs // 16) == nonzero_group
    command[selected] = directions[(envs[selected] % 8 + direction_round) % 8]
    return command


def plan_step(global_policy_step: int, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> StepPlan:
    current = command_at(global_policy_step, device=device, dtype=dtype)
    following = command_at(global_policy_step + 1, device=device, dtype=dtype)
    zero = torch.linalg.vector_norm(current, dim=-1) <= 1.0e-3
    block = None if global_policy_step < WARMUP_STEPS else (global_policy_step - WARMUP_STEPS) // ROLE_BLOCK_STEPS
    return StepPlan(global_policy_step, current, following, zero, ~zero, block)


def reset_local_positions(global_policy_step: int, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[str, ...]]:
    ids = torch.arange(NUM_ENVS, dtype=torch.long) if env_ids is None else env_ids
    if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long:
        raise TypeError("env_ids must be rank-one torch.long")
    positions = torch.zeros(ids.numel(), 3, dtype=torch.float32, device=ids.device); positions[:, 2] = 2.0
    labels = ["center"] * ids.numel()
    if global_policy_step >= WARMUP_STEPS:
        for row, env in enumerate(ids.detach().cpu().tolist()):
            if env % 8 == 6:
                positions[row, 0] = 4.6; labels[row] = "near_x_wall"
            elif env % 8 == 7:
                positions[row, 1] = 4.6; labels[row] = "near_y_wall"
    return positions, tuple(labels)


class ScheduleClock:
    def __init__(self, completed_policy_steps: int = 0) -> None:
        if isinstance(completed_policy_steps, bool) or not isinstance(completed_policy_steps, int) or completed_policy_steps < 0:
            raise ValueError("completed_policy_steps must be nonnegative")
        self.completed_policy_steps = completed_policy_steps

    def plan(self, *, device="cpu", dtype=torch.float32) -> StepPlan:
        return plan_step(self.completed_policy_steps, device=device, dtype=dtype)

    def complete_step(self) -> None:
        self.completed_policy_steps += 1

    def reset_lane(self, env_id: int) -> None:
        if isinstance(env_id, bool) or not isinstance(env_id, int) or not 0 <= env_id < NUM_ENVS:
            raise ValueError("invalid env_id")
        # Lane reset intentionally does not alter the global curriculum clock.

    def state_dict(self) -> dict[str, int | str]:
        return {"version": "curriculum-h1-v1", "completed_policy_steps": self.completed_policy_steps}

    def load_state_dict(self, value: dict[str, int | str]) -> None:
        if set(value) != {"version", "completed_policy_steps"} or value["version"] != "curriculum-h1-v1":
            raise ValueError("curriculum state schema mismatch")
        restored = value["completed_policy_steps"]
        if isinstance(restored, bool) or not isinstance(restored, int) or restored < 0:
            raise ValueError("invalid restored curriculum clock")
        self.completed_policy_steps = restored


__all__ = [
    "DIRECTIONS", "NUM_ENVS", "R4_STEPS", "R5_MAX_STEPS", "ROLE_BLOCK_STEPS", "ROLLOUT_HORIZON",
    "ScheduleClock", "StepPlan", "WARMUP_STEPS", "command_at", "plan_step", "reset_local_positions",
]
