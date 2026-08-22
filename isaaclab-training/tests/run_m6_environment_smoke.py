#!/usr/bin/env python3
"""Two-environment, 32-policy-step fixed-action DirectRLEnv CUDA rollout."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
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


def _nested_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _cpu(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _nested_cpu(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _nested_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_nested_cpu(item) for item in value)
    if isinstance(value, list):
        return [_nested_cpu(item) for item in value]
    return value


def _env_slice(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return _cpu(value[index])
        return _cpu(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _env_slice(getattr(value, field.name), index, batch_size) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _env_slice(item, index, batch_size) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_env_slice(item, index, batch_size) for item in value)
    return value


def _assert_nested_equal(left: Any, right: Any, path: str = "root") -> None:
    if type(left) is not type(right):
        raise AssertionError(f"type mismatch at {path}")
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise AssertionError(f"tensor mismatch at {path}")
        return
    if isinstance(left, dict):
        if list(left) != list(right):
            raise AssertionError(f"key mismatch at {path}")
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f"length mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_nested_equal(left_item, right_item, f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"value mismatch at {path}")


def _all_tensor_zero(value: Any) -> bool:
    tensors: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            if not _all_tensor_zero(item):
                return False
    elif isinstance(value, (tuple, list)):
        for item in value:
            if not _all_tensor_zero(item):
                return False
    return all(torch.count_nonzero(tensor) == 0 for tensor in tensors)


def _reset_state(env: TaslabDirectEnv, index: int) -> dict[str, Any]:
    forces, torques = env._wrench_adapter.pending_wrenches()
    history = env._history
    caches = {
        "control": env._control,
        "lee": env._lee,
        "wrench": env._wrench,
        "reward": env._reward,
        "task": env._task,
    }
    return {
        "articulation": {
            "root_state_w": _cpu(env._robot.data.root_state_w[index]),
            "joint_pos": _cpu(env._robot.data.joint_pos[index]),
            "joint_vel": _cpu(env._robot.data.joint_vel[index]),
        },
        "sensors": {
            "ray_drift": _cpu(env._ray_caster.drift[index]),
            "ray_cast_drift": _cpu(env._ray_caster.ray_cast_drift[index]),
            "ray_timestamp": _cpu(env._ray_caster._timestamp[index]),
            "ray_last_update": _cpu(env._ray_caster._timestamp_last_update[index]),
            "ray_outdated": _cpu(env._ray_caster._is_outdated[index]),
            "contact_force_w": _cpu(env._contact_sensor.data.net_forces_w[index]),
            "contact_force_history_w": _cpu(env._contact_sensor.data.net_forces_w_history[index]),
            "contact_timestamp": _cpu(env._contact_sensor._timestamp[index]),
            "contact_last_update": _cpu(env._contact_sensor._timestamp_last_update[index]),
            "contact_outdated": _cpu(env._contact_sensor._is_outdated[index]),
        },
        "rotor_wrench": {
            "throttle": _cpu(env._wrench_adapter.rotors.throttle[index]),
            "pending_forces": _cpu(forces[index]),
            "pending_torques": _cpu(torques[index]),
        },
        "history": {
            "lidar": _cpu(history._lidar_history[index]),
            "state": _cpu(history._state_history[index]),
            "last_sensor_time": _cpu(history._last_sensor_time_s[index]),
            "last_policy_time": _cpu(history._last_policy_time_s[index]),
            "has_time": _cpu(history._has_time[index]),
        },
        "pipeline": _nested_cpu(env._control_pipelines[index].internal_snapshot()),
        "buffers": {
            "elapsed_steps": _cpu(env.episode_length_buf[index]),
            "action": _cpu(env._actions[index]),
            "previous_issued": _cpu(env._previous_issued_v_final_b[index]),
            "reward_previous": _cpu(env._reward_previous_v_final_b[index]),
            "episode_reset": _cpu(env._episode_reset_pending[index]),
            "explicit_reset": _cpu(env._explicit_reset_pending[index]),
            "policy_lidar": _cpu(env._policy["lidar_grid"][index]),
            "policy_state": _cpu(env._policy["state_vec"][index]),
        },
        "caches": {key: _env_slice(value, index, NUM_ENVS) if value is not None else None
                   for key, value in caches.items()},
    }


def _seed_discriminating_reset_state(env: TaslabDirectEnv) -> None:
    device = env._actions.device
    dtype = env._actions.dtype
    root = env._robot.data.root_state_w.clone()
    root[:, 7:10] = torch.tensor([[0.11, 0.12, 0.13], [0.21, 0.22, 0.23]], device=device, dtype=dtype)
    root[:, 10:13] = torch.tensor([[0.31, 0.32, 0.33], [0.41, 0.42, 0.43]], device=device, dtype=dtype)
    joint_pos = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], device=device, dtype=dtype)
    joint_vel = joint_pos + 1.0
    env._robot.write_root_velocity_to_sim(root[:, 7:])
    env._robot.write_joint_state_to_sim(joint_pos, joint_vel)
    env._wrench_adapter.apply(
        env._robot,
        torch.tensor([[-0.5, -0.25, 0.0, 0.25], [0.25, 0.5, 0.75, 1.0]], device=device, dtype=dtype),
    )
    env._ray_caster.drift[:] = torch.tensor([[0.11, 0.12, 0.13], [0.21, 0.22, 0.23]], device=device)
    env._ray_caster.ray_cast_drift[:] = torch.tensor([[0.31, 0.32, 0.33], [0.41, 0.42, 0.43]], device=device)
    env._ray_caster._timestamp[:] = torch.tensor([3.0, 4.0], device=device)
    env._ray_caster._timestamp_last_update[:] = torch.tensor([2.0, 3.0], device=device)
    env._ray_caster._is_outdated.zero_()
    env._contact_sensor.data.net_forces_w[0] = 1.25
    env._contact_sensor.data.net_forces_w[1] = 2.5
    env._contact_sensor.data.net_forces_w_history[0] = 1.5
    env._contact_sensor.data.net_forces_w_history[1] = 3.0
    env._contact_sensor._timestamp[:] = torch.tensor([5.0, 6.0], device=device)
    env._contact_sensor._timestamp_last_update[:] = torch.tensor([4.0, 5.0], device=device)
    env._contact_sensor._is_outdated.zero_()
    env._history._lidar_history[0].fill_(0.125)
    env._history._lidar_history[1].fill_(0.25)
    env._history._state_history[0].fill_(0.375)
    env._history._state_history[1].fill_(0.5)
    env._history._last_sensor_time_s[:] = torch.tensor([[7.0], [8.0]], device=device)
    env._history._last_policy_time_s[:] = torch.tensor([[7.5], [8.5]], device=device)
    env._history._has_time.fill_(True)
    env.episode_length_buf[:] = torch.tensor([7, 11], device=device)
    env._actions[:] = torch.tensor([[0.1] * 4, [0.9] * 4], device=device, dtype=dtype)
    env._previous_issued_v_final_b[:] = torch.tensor([[0.1] * 3, [0.2] * 3], device=device, dtype=dtype)
    env._reward_previous_v_final_b[:] = torch.tensor([[0.3] * 3, [0.4] * 3], device=device, dtype=dtype)
    env._episode_reset_pending[:] = torch.tensor([False, True], device=device)
    env._explicit_reset_pending[:] = torch.tensor([True, False], device=device)
    env._policy["lidar_grid"][0].fill_(0.625)
    env._policy["lidar_grid"][1].fill_(0.75)
    env._policy["state_vec"][0].fill_(0.825)
    env._policy["state_vec"][1].fill_(0.95)


def _reset_qualification(env: TaslabDirectEnv, action: torch.Tensor) -> dict[str, Any]:
    env.step(action)
    _seed_discriminating_reset_state(env)
    before = [_reset_state(env, index) for index in range(NUM_ENVS)]
    env._reset_idx(torch.tensor([0], device=env.device, dtype=torch.long))
    selected_sensor_outdated = {
        "ray": bool(env._ray_caster._is_outdated[0].item()),
        "contact": bool(env._contact_sensor._is_outdated[0].item()),
    }
    after = [_reset_state(env, index) for index in range(NUM_ENVS)]
    _assert_nested_equal(before[1], after[1], "selective.unaffected_env1")
    default_root = env._robot.data.default_root_state[0].clone()
    default_root[:3] += env.scene.env_origins[0]
    if not torch.equal(after[0]["articulation"]["root_state_w"], _cpu(default_root)):
        raise AssertionError("selected root state did not reset to the public default")
    if torch.count_nonzero(after[0]["articulation"]["joint_pos"]) != 0 or torch.count_nonzero(
        after[0]["articulation"]["joint_vel"]
    ) != 0:
        raise AssertionError("selected joint state did not reset")
    for section in ("rotor_wrench", "history"):
        if not _all_tensor_zero(after[0][section]):
            raise AssertionError(f"selected {section} did not clear")
    if after[0]["pipeline"] != {"anchor": {}, "ics": {}}:
        raise AssertionError("selected M5 pipeline/anchor was not replaced")
    for key in ("elapsed_steps", "action", "previous_issued", "reward_previous", "policy_lidar", "policy_state"):
        if not _all_tensor_zero(after[0]["buffers"][key]):
            raise AssertionError(f"selected buffer {key} did not clear")
    if not bool(after[0]["buffers"]["episode_reset"]) or not bool(after[0]["buffers"]["explicit_reset"]):
        raise AssertionError("selected reset flags were not staged")
    sensor = after[0]["sensors"]
    for key in ("ray_drift", "ray_cast_drift", "ray_timestamp", "ray_last_update",
                "contact_force_w", "contact_force_history_w", "contact_timestamp", "contact_last_update"):
        if not _all_tensor_zero(sensor[key]):
            raise AssertionError(f"selected sensor state {key} did not reset")
    if selected_sensor_outdated != {"ray": True, "contact": True}:
        raise AssertionError("selected sensors were not marked outdated")
    if not all(value is not None and _all_tensor_zero(value) for value in after[0]["caches"].values()):
        raise AssertionError("selected environment cache slices did not clear")

    env._reset_idx(torch.arange(NUM_ENVS, device=env.device, dtype=torch.long))
    full_after = [_reset_state(env, index) for index in range(NUM_ENVS)]
    if not all(all(value is None for value in state["caches"].values()) for state in full_after):
        raise AssertionError("full reset did not invalidate all whole-batch caches")
    return {"selective_before": before, "selective_after": after,
            "selected_sensor_outdated_before_read": selected_sensor_outdated, "full_after": full_after,
            "unaffected_exact": True, "selected_exact": True, "full_exact": True}


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
    if "pre_reset" not in env.extras:
        raise RuntimeError("terminal-safe pre-reset capture is unavailable")
    task = env.extras["task"]
    pre_reset = env.extras["pre_reset"]
    return {
        "step": step,
        "actions": _cpu(action),
        "policy": _mapping(observation["policy"]),
        "critic": _cpu(observation["critic"]),
        "control": _nested_cpu(pre_reset["control"]),
        "reward": {
            "total": _cpu(reward),
            "components": _nested_cpu(pre_reset["reward"]["components"]),
            "diagnostics": _nested_cpu(pre_reset["reward"]["diagnostics"]),
        },
        "done": {
            "terminated": _cpu(terminated),
            "truncated": _cpu(truncated),
            "reason_code": _cpu(task["reason_code"]),
            "collision": _cpu(task["collision"]),
            "exact_clearance_m": _cpu(task["exact_clearance_m"]),
        },
        "pre_reset": _nested_cpu(pre_reset),
        "root_state_w": _cpu(env._robot.data.root_state_w),
        "forces_body_n": _nested_cpu(pre_reset["wrench"]["forces_body_n"]),
        "torques_body_nm": _nested_cpu(pre_reset["wrench"]["torques_body_nm"]),
        "rotor": {
            "normalized_commands": _nested_cpu(pre_reset["lee"]["normalized_rotor_commands"]),
            "target_throttle": _nested_cpu(pre_reset["wrench"]["rotor"]["target_throttle"]),
            "throttle": _nested_cpu(pre_reset["wrench"]["rotor"]["throttle"]),
            "thrusts_n": _nested_cpu(pre_reset["wrench"]["rotor"]["thrusts_n"]),
            "reaction_moments_nm": _nested_cpu(pre_reset["wrench"]["rotor"]["reaction_moments_nm"]),
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
    action = torch.tensor(FIXED_ACTION, device=env.device, dtype=env._actions.dtype).repeat(NUM_ENVS, 1)
    reset_qualification = _reset_qualification(env, action)
    observation, _ = env.reset(seed=0)
    initial_policy = _mapping(observation["policy"])
    negative_checks = _negative_checks(env)
    captures: list[dict[str, Any]] = []
    auto_reset: dict[str, Any] | None = None
    for step in range(1, POLICY_STEPS + 1):
        if step == SELECTIVE_RESET_STEP:
            env.episode_length_buf[0] = env.max_episode_length - 1
        observation, reward, terminated, truncated, _ = env.step(action)
        reset_marker = terminated | truncated
        captures.append(_capture(env, step, action, observation, reward, terminated, truncated, reset_marker))
        if step == SELECTIVE_RESET_STEP:
            pre_reset = env.extras["pre_reset"]
            if terminated.tolist() != [False, False] or truncated.tolist() != [True, False]:
                raise AssertionError("public step did not produce the exact selected timeout edge")
            if env.extras["task"]["reason_code"].flatten().tolist() != [1, 0]:
                raise AssertionError("public step did not preserve accepted timeout reason precedence")
            if int(pre_reset["episode_length"][0].item()) != env.max_episode_length:
                raise AssertionError("terminal capture does not contain the timeout-edge elapsed step")
            if int(env.episode_length_buf[0].item()) != 0:
                raise AssertionError("automatic reset did not restart the selected episode counter")
            if not torch.equal(reward, pre_reset["reward"]["total"][:, 0]):
                raise AssertionError("returned reward is not the terminal pre-reset reward")
            default_root = env._robot.data.default_root_state[0].clone()
            default_root[:3] += env.scene.env_origins[0]
            if not torch.equal(env._robot.data.root_state_w[0], default_root):
                raise AssertionError("returned observation was not built from the post-reset root state")
            if torch.count_nonzero(observation["policy"]["lidar_grid"][0, :9]) != 0 or torch.count_nonzero(
                observation["policy"]["state_vec"][0, :39]
            ) != 0:
                raise AssertionError("terminal history leaked into the post-reset policy branch")
            for name, cache in (("control", env._control), ("lee", env._lee), ("wrench", env._wrench),
                                ("reward", env._reward), ("task", env._task)):
                _assert_nested_equal(
                    _env_slice(cache, 1, NUM_ENVS),
                    _env_slice(pre_reset[name], 1, NUM_ENVS),
                    f"auto_reset.unaffected_cache.{name}",
                )
            auto_reset = {
                "step": step,
                "terminal_pre_reset": _nested_cpu(pre_reset),
                "returned_reward": _cpu(reward),
                "returned_terminated": _cpu(terminated),
                "returned_truncated": _cpu(truncated),
                "returned_reason": _cpu(env.extras["task"]["reason_code"]),
                "post_reset_policy": _mapping(observation["policy"]),
                "post_reset_critic": _cpu(observation["critic"]),
                "post_reset_root": _cpu(env._robot.data.root_state_w),
                "post_reset_episode": _cpu(env.episode_length_buf),
                "unaffected_cache_exact": True,
            }

    if auto_reset is None:
        raise RuntimeError("automatic done-to-reset evidence was not captured")
    if any(not torch.isfinite(captures[index]["reward"]["total"]).all() for index in (16, 17)):
        raise FloatingPointError("post-reset continuation steps were not finite")
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
            "automatic_timeout_step": SELECTIVE_RESET_STEP,
        },
        "initial_policy": initial_policy,
        "reset_qualification": reset_qualification,
        "auto_reset": auto_reset,
        "negative_checks": negative_checks,
        "captures": captures,
        "gates": {
            "all_finite": all_finite,
            "capture_count": len(captures),
            "automatic_reset_count": sum(int(capture["reset_marker"].sum().item()) for capture in captures),
            "post_reset_continuation_steps": POLICY_STEPS - SELECTIVE_RESET_STEP,
            "all_gates_pass": all_finite and len(captures) == POLICY_STEPS and auto_reset is not None,
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
