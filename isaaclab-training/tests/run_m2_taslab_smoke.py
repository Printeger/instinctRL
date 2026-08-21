#!/usr/bin/env python3
"""One-articulation deterministic CUDA gravity-only smoke for M2."""

from __future__ import annotations

import argparse
import json
import math
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
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instinctlab.assets.taslab_uav.articulation import TASLAB_ARTICULATION_CFG

DEVICE = "cuda:0"
DT = 0.005
STEPS_PER_TRAJECTORY = 32
TOTAL_PHYSICS_STEPS = 64
TOLERANCE = 1.0e-5
EXPECTED_BODY_NAMES = ("base_link", "rotor_0", "rotor_1", "rotor_2", "rotor_3")
EXPECTED_JOINT_NAMES = ("rotor_0_joint", "rotor_1_joint", "rotor_2_joint", "rotor_3_joint")
CAPTURE_STEPS = {0, 1, 8, 16, 24, 32}


def _tensor(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def _normalized_root(value: torch.Tensor) -> torch.Tensor:
    normalized = value.clone()
    normalized[..., 3:7] *= torch.where(normalized[..., 3:4] < 0.0, -1.0, 1.0)
    return normalized


def _snapshot(robot: Articulation, step: int) -> dict[str, Any]:
    return {"step": step, "root_state_w": _tensor(_normalized_root(robot.data.root_state_w)),
            "joint_pos": _tensor(robot.data.joint_pos), "joint_vel": _tensor(robot.data.joint_vel)}


def _restore(robot: Articulation) -> dict[str, Any]:
    root = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    robot.update(0.0)
    return _snapshot(robot, 0)


def _trajectory(sim: SimulationContext, robot: Articulation) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    states = [_snapshot(robot, 0)]
    selected = [states[0]]
    for step in range(1, STEPS_PER_TRAJECTORY + 1):
        sim.step(render=False)
        robot.update(DT)
        state = _snapshot(robot, step)
        states.append(state)
        if step in CAPTURE_STEPS:
            selected.append(state)
    return states, selected


def _flatten(state: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for key in ("root_state_w", "joint_pos", "joint_vel"):
        stack = [state[key]]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(reversed(item))
            else:
                values.append(float(item))
    return values


def _trajectory_error(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> float:
    errors = [abs(a - b) for left, right in zip(first, second, strict=True)
              for a, b in zip(_flatten(left), _flatten(right), strict=True)]
    return max(errors, default=0.0)


def _sanity(states: list[dict[str, Any]]) -> dict[str, Any]:
    initial = states[0]["root_state_w"][0]
    final = states[-1]["root_state_w"][0]
    quaternions = [state["root_state_w"][0][3:7] for state in states]
    quat_error = max(abs(math.sqrt(sum(value * value for value in quat)) - 1.0) for quat in quaternions)
    attitude = max(2.0 * math.acos(min(1.0, abs(quat[0]))) for quat in quaternions)
    horizontal = math.hypot(final[0] - initial[0], final[1] - initial[1])
    drop = initial[2] - final[2]
    finite = all(math.isfinite(value) for state in states for value in _flatten(state))
    return {"finite": finite, "quaternion_norm_error": quat_error,
            "horizontal_displacement_m": horizontal, "attitude_angle_drift_rad": attitude,
            "vertical_drop_m": drop, "quaternion_norm_pass": quat_error <= 1.0e-5,
            "horizontal_pass": horizontal <= 1.0e-4, "attitude_pass": attitude <= 1.0e-3,
            "vertical_pass": 0.05 <= drop <= 0.30}


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=DEVICE))
    cfg = TASLAB_ARTICULATION_CFG.replace(prim_path="/World/TASLAB")
    cfg.spawn.func(cfg.prim_path, cfg.spawn, translation=cfg.init_state.pos, orientation=cfg.init_state.rot)
    robot = Articulation(cfg)
    sim.reset()
    robot.update(0.0)
    body_names = tuple(robot.body_names)
    joint_names = tuple(robot.joint_names)
    topology_pass = body_names == EXPECTED_BODY_NAMES and joint_names == EXPECTED_JOINT_NAMES
    masses = robot.root_physx_view.get_masses()
    inertias = robot.root_physx_view.get_inertias()
    centers = robot.root_physx_view.get_coms()
    defaults = {"root_state_w": _tensor(_normalized_root(robot.data.default_root_state)),
                "joint_pos": _tensor(robot.data.default_joint_pos),
                "joint_vel": _tensor(robot.data.default_joint_vel)}
    first_restore = _restore(robot)
    first_all, first_selected = _trajectory(sim, robot)
    replay_restore = _restore(robot)
    replay_all, replay_selected = _trajectory(sim, robot)
    replay_error = _trajectory_error(first_all, replay_all)
    sanity = _sanity(first_all)
    properties_finite = all(torch.isfinite(value).all().item() for value in (masses, inertias, centers))
    all_state_finite = sanity["finite"] and all(math.isfinite(value) for state in replay_all for value in _flatten(state))
    all_gates = (topology_pass and properties_finite and all_state_finite and replay_error <= TOLERANCE
                 and sanity["quaternion_norm_pass"] and sanity["horizontal_pass"]
                 and sanity["attitude_pass"] and sanity["vertical_pass"])
    result = {
        "schema_version": "m2-taslab-smoke-1",
        "contract": {"device": DEVICE, "dt_s": DT, "articulations": 1,
                     "steps_per_trajectory": STEPS_PER_TRAJECTORY, "trajectories": 2,
                     "total_physics_steps": TOTAL_PHYSICS_STEPS, "applied_control": False,
                     "tolerance": TOLERANCE},
        "topology": {"articulation_prim_path": cfg.prim_path,
                     "root_physx_prim_paths": list(robot.root_physx_view.prim_paths),
                     "body_count": robot.num_bodies, "body_names": list(body_names),
                     "joint_count": robot.num_joints, "joint_names": list(joint_names),
                     "expected_match": topology_pass},
        "tensor_contract": {"root_device": str(robot.data.root_state_w.device),
                            "root_dtype": str(robot.data.root_state_w.dtype),
                            "joint_device": str(robot.data.joint_pos.device),
                            "joint_dtype": str(robot.data.joint_pos.dtype)},
        "effective_properties": {"body_masses_kg": _tensor(masses),
                                 "base_mass_kg": float(masses[0, body_names.index("base_link")].item()),
                                 "total_articulated_mass_kg": float(masses.sum().item()),
                                 "body_centers_of_mass": _tensor(centers),
                                 "body_inertias": _tensor(inertias), "finite": properties_finite},
        "states": {"default": defaults, "first_restore": first_restore, "replay_restore": replay_restore,
                   "first_selected": first_selected, "replay_selected": replay_selected},
        "gates": {"within_process_max_absolute_state_error": replay_error,
                  "within_process_replay_pass": replay_error <= TOLERANCE,
                  **sanity, "all_state_finite": all_state_finite, "all_gates_pass": all_gates},
        "operation_audit": {"app_launcher": "headless=True, enable_cameras=False",
                            "simulation_contexts": 1, "articulations_spawned": 1,
                            "ground_planes": 0, "lights": 0, "cameras": 0,
                            "force_calls": 0, "torque_calls": 0, "wrench_calls": 0,
                            "effort_target_calls": 0, "position_target_calls": 0,
                            "velocity_target_calls": 0, "controller_calls": 0,
                            "physics_steps": TOTAL_PHYSICS_STEPS,
                            "motion_source": "gravity and free articulation dynamics only",
                            "shutdown": "os._exit after normalized JSON and streams are flushed"}}
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    ARGS.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if all_gates else "FAIL"}, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if all_gates else 2)


if __name__ == "__main__":
    main()
