#!/usr/bin/env python3
"""Two-environment, 32-policy-step fixed-action DirectRLEnv CUDA rollout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

from isaaclab.app import AppLauncher

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--output", type=Path, required=True)
ARGS = PARSER.parse_args()
APP_LAUNCHER = AppLauncher(headless=True, enable_cameras=False)
SIMULATION_APP = APP_LAUNCHER.app

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instinctlab.envs import TaslabDirectEnv, TaslabDirectEnvCfg

NUM_ENVS = 2
POLICY_STEPS = 32
PHYSICS_STEPS = 640
SELECTIVE_RESET_STEP = 16
FIXED_ACTION = (1.0, 0.5, 0.5, 0.5)


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _mapping(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: _cpu(value) for key, value in values.items()}


def _capture(
    env: TaslabDirectEnv,
    step: int,
    action: torch.Tensor,
    observation: dict[str, Any],
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    reset_marker: torch.Tensor,
) -> dict[str, Any]:
    if env._control is None or env._lee is None or env._wrench is None or env._reward is None:
        raise RuntimeError("complete control/task capture is unavailable")
    task = env.extras["task"]
    return {
        "step": step,
        "actions": _cpu(action),
        "policy": _mapping(observation["policy"]),
        "critic": _cpu(observation["critic"]),
        "control": {
            "alpha": _cpu(env._control.alpha),
            "v_corr_b": _cpu(env._control.v_corr_b),
            "v_gov_b": _cpu(env._control.v_gov_b),
            "v_final_b": _cpu(env._control.v_final_b),
            "v_final_w": _cpu(env._control.v_final_w),
            "anchor_metrics": _mapping(env._control.anchor_metrics),
            "ics_metrics": _mapping(env._control.ics_metrics),
        },
        "reward": {
            "total": _cpu(reward),
            "components": _mapping(env._reward.components),
            "diagnostics": _mapping(env._reward.diagnostics),
        },
        "done": {
            "terminated": _cpu(terminated),
            "truncated": _cpu(truncated),
            "reason_code": _cpu(task["reason_code"]),
            "collision": _cpu(task["collision"]),
            "exact_clearance_m": _cpu(task["exact_clearance_m"]),
        },
        "root_state_w": _cpu(env._robot.data.root_state_w),
        "forces_body_n": _cpu(env._wrench.forces_body_n),
        "torques_body_nm": _cpu(env._wrench.torques_body_nm),
        "rotor": {
            "normalized_commands": _cpu(env._lee.normalized_rotor_commands),
            "target_throttle": _cpu(env._wrench.rotor.target_throttle),
            "throttle": _cpu(env._wrench.rotor.throttle),
            "thrusts_n": _cpu(env._wrench.rotor.thrusts_n),
            "reaction_moments_nm": _cpu(env._wrench.rotor.reaction_moments_nm),
        },
        "reset_marker": _cpu(reset_marker),
        "episode_length": _cpu(env.episode_length_buf),
    }


def _negative_checks(env: TaslabDirectEnv) -> dict[str, bool]:
    valid = torch.tensor(FIXED_ACTION, device=env.device, dtype=env._actions.dtype).repeat(NUM_ENVS, 1)
    malformed = {
        "shape": valid[:, :3],
        "dtype": valid.to(torch.float64),
        "nonfinite": valid.clone(),
        "below_range": valid.clone(),
        "above_range": valid.clone(),
        "device": valid.cpu(),
    }
    malformed["nonfinite"][0, 0] = float("nan")
    malformed["below_range"][0, 0] = -0.01
    malformed["above_range"][0, 0] = 1.01
    results: dict[str, bool] = {}
    for name, action in malformed.items():
        before = (
            env._actions.clone(), env._robot.data.root_state_w.clone(),
            env.episode_length_buf.clone(), env._wrench_adapter.rotors.throttle,
        )
        try:
            env.step(action)
        except (TypeError, ValueError):
            after = (
                env._actions, env._robot.data.root_state_w,
                env.episode_length_buf, env._wrench_adapter.rotors.throttle,
            )
            results[name] = all(torch.equal(left, right) for left, right in zip(before, after, strict=True))
        else:
            results[name] = False
    before_root = env._robot.data.root_state_w.clone()
    before_episode = env.episode_length_buf.clone()
    try:
        env._reset_idx(torch.tensor([NUM_ENVS], dtype=torch.long, device=env.device))
    except IndexError:
        results["reset_index"] = torch.equal(before_root, env._robot.data.root_state_w) and torch.equal(
            before_episode, env.episode_length_buf
        )
    else:
        results["reset_index"] = False
    if not all(results.values()):
        raise AssertionError(f"negative atomicity failure: {results}")
    return results


def main() -> None:
    cfg = TaslabDirectEnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    env = TaslabDirectEnv(cfg)
    observation, _ = env.reset(seed=0)
    initial_policy = _mapping(observation["policy"])
    negative_checks = _negative_checks(env)
    action = torch.tensor(FIXED_ACTION, device=env.device, dtype=env._actions.dtype).repeat(NUM_ENVS, 1)
    captures: list[dict[str, Any]] = []
    selective_before: dict[str, torch.Tensor] | None = None
    selective_after: dict[str, torch.Tensor] | None = None
    for step in range(1, POLICY_STEPS + 1):
        reset_marker = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.device)
        if step == SELECTIVE_RESET_STEP:
            selective_before = {
                "root_state_w_env1": _cpu(env._robot.data.root_state_w[1]),
                "policy_cache_env1": _cpu(env._policy["state_vec"][1]),
                "episode_env1": _cpu(env.episode_length_buf[1]),
            }
            reset_ids = torch.tensor([0], dtype=torch.long, device=env.device)
            env._reset_idx(reset_ids)
            selective_after_reset = {
                "root_state_w_env1": _cpu(env._robot.data.root_state_w[1]),
                "policy_cache_env1": _cpu(env._policy["state_vec"][1]),
                "episode_env1": _cpu(env.episode_length_buf[1]),
            }
            for key in selective_before:
                if not torch.equal(selective_before[key], selective_after_reset[key]):
                    raise AssertionError(f"selective reset changed unaffected {key}")
            env.scene.write_data_to_sim()
            env.sim.forward()
            observation = env._get_observations()
            reset_marker[0] = True
            selective_after = {
                "root_state_w_env1": _cpu(env._robot.data.root_state_w[1]),
                "policy_cache_env1": _cpu(env._policy["state_vec"][1]),
                "episode_env1": _cpu(env.episode_length_buf[1]),
            }
        observation, reward, terminated, truncated, _ = env.step(action)
        captures.append(_capture(env, step, action, observation, reward, terminated, truncated, reset_marker))

    if selective_before is None or selective_after is None:
        raise RuntimeError("selective reset evidence was not captured")
    if not torch.equal(selective_before["episode_env1"], selective_after["episode_env1"]):
        raise AssertionError("selective reset changed unaffected episode counter")
    all_finite = all(
        torch.isfinite(capture["reward"]["total"]).all()
        and torch.isfinite(capture["policy"]["lidar_grid"]).all()
        and torch.isfinite(capture["policy"]["state_vec"]).all()
        and torch.isfinite(capture["critic"]).all()
        and torch.isfinite(capture["root_state_w"]).all()
        for capture in captures
    )
    if not all_finite:
        raise FloatingPointError("rollout contains non-finite state")
    output = {
        "schema_version": "m6-environment-smoke-1",
        "contract": {
            "num_envs": NUM_ENVS,
            "policy_steps": POLICY_STEPS,
            "physics_steps": PHYSICS_STEPS,
            "physics_dt_s": 0.005,
            "decimation": 20,
            "policy_dt_s": 0.1,
            "device": "cuda:0",
            "fixed_action": FIXED_ACTION,
            "selective_reset_step": SELECTIVE_RESET_STEP,
        },
        "initial_policy": initial_policy,
        "selective_reset": {"before": selective_before, "after": selective_after},
        "negative_checks": negative_checks,
        "captures": captures,
        "gates": {
            "all_finite": all_finite,
            "capture_count": len(captures),
            "selective_reset_count": sum(int(capture["reset_marker"].sum().item()) for capture in captures),
            "all_gates_pass": all_finite and len(captures) == POLICY_STEPS,
        },
    }
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, ARGS.output)
    print({"status": "PASS", "captures": len(captures), "physics_steps": PHYSICS_STEPS})
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
