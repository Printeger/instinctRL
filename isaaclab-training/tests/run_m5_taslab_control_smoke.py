#!/usr/bin/env python3
"""Bounded one-environment end-to-end TASLAB CUDA control smoke."""

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
from isaaclab.sensors import RayCaster
from isaaclab.sim import SimulationContext
from pxr import UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instinctlab.assets.taslab_uav.articulation import TASLAB_ARTICULATION_CFG, TASLAB_BODY_NAMES
from instinctlab.assets.taslab_uav.config import TASLAB_UAV
from instinctlab.control.lee_velocity_controller import GRAVITY_M_S2, rotate_world_to_body_wxyz, taslab_velocity_control
from instinctlab.control.pipeline import ControlCoreConfig, ControlCorePipeline
from instinctlab.control.runtime_adapter import TaslabArticulationWrenchAdapter
from instinctlab.observations.actor import ActorObservationHistory
from instinctlab.sensors.mid360 import (
    MID360_RAY_CASTER_CFG,
    UPDATE_PERIOD_S,
    compose_mounted_sensor_pose,
    ranges_and_validity,
)

DEVICE = "cuda:0"
DT = 0.005
HISTORY_DT_S = 0.1
PHYSICS_STEPS_PER_SCENARIO = 64
SCENARIO_COUNT = 4
TOTAL_PHYSICS_STEPS = 256
POLICY_INTERVAL_STEPS = 20
TOLERANCE = 1.0e-5
FIXED_SEED = 5002
SCENARIOS = (
    ("zero_b0", (0.0, 0.0, 0.0), 0.0),
    ("positive_x", (0.35, 0.0, 0.0), 0.0),
    ("yaw_positive_x", (0.35, 0.0, 0.0), 90.0),
    ("reset_replay", (0.35, 0.0, 0.0), 0.0),
)


def _spawn_static_plane() -> None:
    minimum, maximum, spacing = -42, 42, 2
    coordinates = list(range(minimum, maximum + spacing, spacing))
    points = [(float(x), float(y), 0.0) for y in coordinates for x in coordinates]
    width = len(coordinates)
    indices: list[int] = []
    for row in range(width - 1):
        for column in range(width - 1):
            lower_left = row * width + column
            lower_right = lower_left + 1
            upper_left = lower_left + width
            upper_right = upper_left + 1
            indices.extend((lower_left, lower_right, upper_right, lower_left, upper_right, upper_left))
    sim_utils.create_prim("/World/MID360StaticScene", "Xform")
    mesh = UsdGeom.Mesh.Define(sim_utils.get_current_stage(), "/World/MID360StaticScene/PlaneMesh")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")


def _yaw_quaternion(degrees: float, reference: torch.Tensor) -> torch.Tensor:
    half = math.radians(degrees) * 0.5
    return torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]], device=reference.device, dtype=reference.dtype)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    return value


def _stack_records(records: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([record[key] for record in records]) for key in records[0]}


def _restore(
    robot: Articulation,
    sensor: RayCaster,
    adapter: TaslabArticulationWrenchAdapter,
    yaw_degrees: float,
) -> None:
    adapter.reset(robot)
    root = robot.data.default_root_state.clone()
    root[:, 3:7] = _yaw_quaternion(yaw_degrees, root)
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()
    robot.update(0.0)
    sensor.reset()
    sensor.update(UPDATE_PERIOD_S, force_recompute=True)


def _measurement(sensor: RayCaster) -> tuple[torch.Tensor, torch.Tensor]:
    parent_position = sensor.data.pos_w
    parent_quaternion = sensor.data.quat_w
    mount_position = torch.tensor(sensor.cfg.offset.pos, device=parent_position.device, dtype=parent_position.dtype)
    mount_quaternion = torch.tensor(sensor.cfg.offset.rot, device=parent_quaternion.device, dtype=parent_quaternion.dtype)
    origin, _ = compose_mounted_sensor_pose(parent_position, parent_quaternion, mount_position, mount_quaternion)
    _, ranges, validity = ranges_and_validity(sensor.data.ray_hits_w, origin)
    return ranges, validity


def _scenario(
    sim: SimulationContext,
    robot: Articulation,
    sensor: RayCaster,
    command_body_values: tuple[float, float, float],
    yaw_degrees: float,
) -> dict[str, Any]:
    dtype = robot.data.root_state_w.dtype
    adapter = TaslabArticulationWrenchAdapter(1, DEVICE, dtype)
    _restore(robot, sensor, adapter, yaw_degrees)
    observation_history = ActorObservationHistory()
    control = ControlCorePipeline(ControlCoreConfig(use_b0=True))
    command_body = torch.tensor([command_body_values], device=DEVICE, dtype=dtype)
    fixed_action = torch.zeros(1, 4, device=DEVICE, dtype=dtype)
    reset_mask = torch.zeros(1, dtype=torch.bool, device=DEVICE)
    previous_issued = torch.zeros(1, 3, device=DEVICE, dtype=dtype)
    target_velocity_w = torch.zeros(1, 3, device=DEVICE, dtype=dtype)
    physics_records: list[dict[str, torch.Tensor]] = []
    policy_records: list[dict[str, torch.Tensor]] = []

    for step in range(PHYSICS_STEPS_PER_SCENARIO):
        if step % POLICY_INTERVAL_STEPS == 0:
            sensor.update(HISTORY_DT_S, force_recompute=True)
            ranges, validity = _measurement(sensor)
            root = robot.data.root_state_w.clone()
            body_angular_velocity = rotate_world_to_body_wxyz(root[:, 3:7], root[:, 10:13])
            gravity_world = torch.tensor([[0.0, 0.0, -1.0]], device=DEVICE, dtype=dtype)
            gravity_body = rotate_world_to_body_wxyz(root[:, 3:7], gravity_world)
            timestamp = torch.full((1, 1), step * DT, device=DEVICE, dtype=dtype)
            actor = observation_history.update(
                ranges,
                validity,
                timestamp,
                timestamp,
                body_angular_velocity,
                gravity_body,
                command_body,
                previous_issued,
            )
            control_result = control(
                actor["lidar_grid"],
                actor["state_vec"],
                fixed_action,
                HISTORY_DT_S,
                reset_mask,
                reset_mask,
                root[:, 3:7],
            )
            previous_issued = control_result.v_final_b.detach().clone()
            target_velocity_w = control_result.v_final_w.detach().clone()
            policy_records.append(
                {
                    "step": torch.tensor(step, device=DEVICE, dtype=torch.long),
                    "ranges_m": ranges[0].clone(),
                    "validity": validity[0].clone(),
                    "lidar_grid": actor["lidar_grid"][0].clone(),
                    "state_vec": actor["state_vec"][0].clone(),
                    "fixed_action": fixed_action[0].clone(),
                    "alpha": control_result.alpha[0].clone(),
                    "v_corr_b": control_result.v_corr_b[0].clone(),
                    "v_gov_b": control_result.v_gov_b[0].clone(),
                    "v_final_b": control_result.v_final_b[0].clone(),
                    "v_final_w": control_result.v_final_w[0].clone(),
                    "beta": control_result.ics_metrics["beta"][0].clone(),
                    "anchor_active": control_result.anchor_metrics["active"][0].clone(),
                }
            )

        root_before = robot.data.root_state_w.clone()
        lee = taslab_velocity_control(root_before, target_velocity_w)
        wrench = adapter.apply(robot, lee.normalized_rotor_commands)
        adapter.write_data_to_sim(robot)
        sim.step(render=False)
        robot.update(DT)
        sensor.update(DT)
        root_after = robot.data.root_state_w.clone()
        physics_records.append(
            {
                "step": torch.tensor(step + 1, device=DEVICE, dtype=torch.long),
                "reset_marker": torch.tensor(step == 0, device=DEVICE, dtype=torch.bool),
                "root_state_before_w": root_before[0].clone(),
                "root_state_after_w": root_after[0].clone(),
                "target_velocity_w": target_velocity_w[0].clone(),
                "controller_coordinates": lee.normalized_rotor_commands[0].clone(),
                "throttle": wrench.rotor.throttle[0].clone(),
                "thrusts_n": wrench.rotor.thrusts_n[0].clone(),
                "reaction_moments_nm": wrench.rotor.reaction_moments_nm[0].clone(),
                "forces_body_n": wrench.forces_body_n[0].clone(),
                "torques_body_nm": wrench.torques_body_nm[0].clone(),
            }
        )
    return {"physics": _stack_records(physics_records), "policy": _stack_records(policy_records)}


def _maximum_error(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> float:
    maximum = 0.0
    for section in ("physics", "policy"):
        for key in first[section]:
            left, right = first[section][key], second[section][key]
            if left.is_floating_point():
                maximum = max(maximum, float(torch.abs(left - right).max().item()))
            elif not torch.equal(left, right):
                return float("inf")
    return maximum


def main() -> None:
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=DEVICE))
    _spawn_static_plane()
    robot = Articulation(TASLAB_ARTICULATION_CFG.replace(prim_path="/World/TASLAB"))
    sensor = RayCaster(MID360_RAY_CASTER_CFG.replace(prim_path="/World/TASLAB/base_link"))
    sim.reset()
    robot.update(0.0)
    sensor.reset()
    traces = {
        name: _scenario(sim, robot, sensor, command, yaw)
        for name, command, yaw in SCENARIOS
    }

    zero_initial = traces["zero_b0"]["physics"]["root_state_before_w"][0]
    zero_final = traces["zero_b0"]["physics"]["root_state_after_w"][-1]
    positive_initial = traces["positive_x"]["physics"]["root_state_before_w"][0]
    positive_final = traces["positive_x"]["physics"]["root_state_after_w"][-1]
    yaw_initial = traces["yaw_positive_x"]["physics"]["root_state_before_w"][0]
    yaw_final = traces["yaw_positive_x"]["physics"]["root_state_after_w"][-1]
    freefall_drop = 0.5 * GRAVITY_M_S2 * (PHYSICS_STEPS_PER_SCENARIO * DT) ** 2
    zero_drop = float((zero_initial[2] - zero_final[2]).item())
    positive_x = float((positive_final[0] - positive_initial[0]).item())
    yaw_positive_y = float((yaw_final[1] - yaw_initial[1]).item())
    replay_error = _maximum_error(traces["positive_x"], traces["reset_replay"])
    quaternion_error = max(
        float(torch.abs(torch.linalg.vector_norm(trace["physics"]["root_state_after_w"][:, 3:7], dim=-1) - 1.0).max().item())
        for trace in traces.values()
    )
    maximum_thrust = max(
        float(trace["physics"]["thrusts_n"].max().item()) for trace in traces.values()
    )
    theoretical_maximum_thrust = TASLAB_UAV.rotors.force_constants[0] * TASLAB_UAV.rotors.max_angular_velocities[0] ** 2
    all_finite = all(
        bool(value.isfinite().all().item())
        for trace in traces.values()
        for section in trace.values()
        for value in section.values()
        if value.is_floating_point()
    )
    topology_pass = tuple(robot.body_names) == TASLAB_BODY_NAMES
    gates = {
        "all_finite": all_finite,
        "topology_pass": topology_pass,
        "quaternion_norm_error": quaternion_error,
        "quaternion_pass": quaternion_error <= TOLERANCE,
        "zero_command_drop_m": zero_drop,
        "freefall_reference_drop_m": freefall_drop,
        "zero_command_not_freefall_pass": zero_drop < 0.8 * freefall_drop,
        "positive_x_displacement_m": positive_x,
        "positive_x_sign_pass": positive_x > 0.0,
        "yaw_positive_y_displacement_m": yaw_positive_y,
        "yaw_boundary_sign_pass": yaw_positive_y > 0.0,
        "within_process_replay_max_error": replay_error,
        "within_process_replay_pass": replay_error <= TOLERANCE,
        "maximum_per_rotor_thrust_n": maximum_thrust,
        "theoretical_maximum_per_rotor_thrust_n": theoretical_maximum_thrust,
        "force_bound_pass": maximum_thrust <= theoretical_maximum_thrust + TOLERANCE,
    }
    gates["all_gates_pass"] = all(
        bool(gates[key]) for key in (
            "all_finite", "topology_pass", "quaternion_pass", "zero_command_not_freefall_pass",
            "positive_x_sign_pass", "yaw_boundary_sign_pass", "within_process_replay_pass", "force_bound_pass",
        )
    )
    payload = {
        "contract": {
            "schema": "m5-taslab-runtime-smoke-1",
            "device": DEVICE,
            "dt_s": DT,
            "history_dt_s": HISTORY_DT_S,
            "physics_steps_per_scenario": PHYSICS_STEPS_PER_SCENARIO,
            "scenario_count": SCENARIO_COUNT,
            "total_physics_steps": TOTAL_PHYSICS_STEPS,
            "fixed_seed": FIXED_SEED,
            "tolerance": TOLERANCE,
            "body_names": tuple(robot.body_names),
            "rotor_names": TASLAB_BODY_NAMES[1:],
        },
        "traces": traces,
        "gates": gates,
    }
    output_pt = ARGS.output.with_suffix(".pt")
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cpu_tree(payload), output_pt)
    summary = {"contract": payload["contract"], "gates": gates, "trace_path": output_pt.name}
    ARGS.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if gates["all_gates_pass"] else "FAIL", "steps": TOTAL_PHYSICS_STEPS}, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if gates["all_gates_pass"] else 2)


if __name__ == "__main__":
    main()
