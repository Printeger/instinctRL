#!/usr/bin/env python3
"""Bounded one-articulation headless CUDA B0 controller-handoff smoke."""

from __future__ import annotations

import argparse
import hashlib
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
from instinctlab.assets.taslab_uav.articulation import TASLAB_ARTICULATION_CFG, TASLAB_BODY_NAMES
from instinctlab.assets.taslab_uav.config import TASLAB_UAV
from instinctlab.control.baseline_b0 import B0DirectConfig
from instinctlab.control.runtime_adapter import TaslabArticulationWrenchAdapter
from instinctlab.evaluation.baseline_controller_execution import execute_ready_b0_controller
from instinctlab.evaluation.baselines import ActorAuditStatus, CapabilityBinding, ExecutionStatus, build_run_manifest, resolve_baseline

DEVICE = "cuda:0"
DT_S = 0.005
STEPS_PER_FIXTURE = 64
FIXTURE_COUNT = 4
TOTAL_STEPS = 256
FIXED_SEED = 7009
VELOCITY_LIMIT_M_S = 2.0
FIXTURES = (
    ("zero_identity", (0.0, 0.0, 0.0), 0.0),
    ("below_identity", (0.35, -0.10, 0.05), 0.0),
    ("over_yaw", (3.0, 4.0, 0.0), 90.0),
    ("reset_replay", (0.35, -0.10, 0.05), 0.0),
)

def _yaw_quaternion(degrees: float, reference: torch.Tensor) -> torch.Tensor:
    half = math.radians(degrees) * 0.5
    return torch.tensor(((math.cos(half), 0.0, 0.0, math.sin(half)),), device=reference.device, dtype=reference.dtype)

def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {key: _cpu(child) for key, child in value.items()}
    return value

def _stack(records: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([record[key] for record in records]) for key in records[0]}

def _manifest(product_commit: str):
    row = resolve_baseline("B0")
    bindings = tuple(CapabilityBinding(item.name, item.accepted_provider, item.accepted_evidence_sha256) for item in row.capability_requirements)
    return build_run_manifest(product_commit=product_commit, baseline_id="B0", config_switch=row.config_switch, capability_bindings=bindings, scenario_id="M7-009.constant_command", seed=FIXED_SEED, actor_audit_status=ActorAuditStatus.PASS, execution_status=ExecutionStatus.READY, skip_reason=None)

def _restore(robot: Articulation, adapter: TaslabArticulationWrenchAdapter, yaw_degrees: float) -> None:
    adapter.reset(robot)
    root = robot.data.default_root_state.clone()
    root[:, 3:7] = _yaw_quaternion(yaw_degrees, root)
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset(); robot.update(0.0)

def _fixture(sim: SimulationContext, robot: Articulation, adapter: TaslabArticulationWrenchAdapter, manifest, command_values, yaw_degrees):
    _restore(robot, adapter, yaw_degrees)
    dtype = robot.data.root_state_w.dtype
    command = torch.tensor((command_values,), device=DEVICE, dtype=dtype)
    config = B0DirectConfig(VELOCITY_LIMIT_M_S)
    records = []
    executed_json = None
    for step in range(STEPS_PER_FIXTURE):
        root_before = robot.data.root_state_w.clone()
        envelope = execute_ready_b0_controller(manifest, command, root_before, config)
        if executed_json is None: executed_json = envelope.b0_execution.executed_manifest.canonical_json
        elif envelope.b0_execution.executed_manifest.canonical_json != executed_json: raise AssertionError("EXECUTED manifest changed within fixture")
        wrench = adapter.apply(robot, envelope.lee_result.normalized_rotor_commands)
        adapter.write_data_to_sim(robot)
        sim.step(render=False); robot.update(DT_S)
        records.append({
            "step": torch.tensor(step + 1, device=DEVICE, dtype=torch.long),
            "reset_marker": torch.tensor(step == 0, device=DEVICE, dtype=torch.bool),
            "root_before_w": root_before[0].clone(), "root_after_w": robot.data.root_state_w[0].clone(),
            "command_body_m_s": command[0].clone(),
            "issued_body_m_s": envelope.b0_execution.result.issued_body_velocity_m_s[0].clone(),
            "issued_world_m_s": envelope.b0_execution.result.issued_world_velocity_m_s[0].clone(),
            "lee_normalized_rotor": envelope.lee_result.normalized_rotor_commands[0].clone(),
            "lee_held_position_w_m": envelope.lee_result.held_position_w_m[0].clone(),
            "lee_held_yaw_rad": envelope.lee_result.held_yaw_rad[0].clone(),
            "lee_desired_rotation_w": envelope.lee_result.desired_rotation_w[0].clone(),
            "lee_collective_thrust_n": envelope.lee_result.collective_thrust_n[0].clone(),
            "rotor_target_throttle": wrench.rotor.target_throttle[0].clone(),
            "rotor_throttle": wrench.rotor.throttle[0].clone(),
            "rotor_thrusts_n": wrench.rotor.thrusts_n[0].clone(),
            "rotor_reaction_moments_nm": wrench.rotor.reaction_moments_nm[0].clone(),
            "forces_body_n": wrench.forces_body_n[0].clone(),
            "torques_body_nm": wrench.torques_body_nm[0].clone(),
        })
    return {"ready_json": manifest.canonical_json, "executed_json": executed_json, "records": _stack(records)}

def main() -> None:
    if ARGS.output.exists() or ARGS.output.with_suffix(".pt").exists(): raise FileExistsError(ARGS.output)
    torch.manual_seed(FIXED_SEED); torch.cuda.manual_seed_all(FIXED_SEED)
    product_commit = os.environ["M7_009_PRODUCT_COMMIT"]
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT_S, device=DEVICE))
    robot = Articulation(TASLAB_ARTICULATION_CFG.replace(prim_path="/World/TASLAB"))
    sim.reset(); robot.update(0.0)
    adapter = TaslabArticulationWrenchAdapter(1, DEVICE, robot.data.root_state_w.dtype)
    manifest = _manifest(product_commit)
    traces = {name: _fixture(sim, robot, adapter, manifest, command, yaw) for name, command, yaw in FIXTURES}
    floating = [value for trace in traces.values() for value in trace["records"].values() if value.is_floating_point()]
    max_thrust = TASLAB_UAV.rotors.force_constants[0] * TASLAB_UAV.rotors.max_angular_velocities[0] ** 2
    gates = {
        "all_finite": all(bool(value.isfinite().all().item()) for value in floating),
        "topology_exact": tuple(robot.body_names) == TASLAB_BODY_NAMES,
        "unit_quaternion": all(bool((torch.abs(torch.linalg.vector_norm(trace["records"]["root_before_w"][:, 3:7], dim=-1) - 1.0) <= 1e-5).all().item()) for trace in traces.values()),
        "body_clip_exact": all(bool((torch.linalg.vector_norm(trace["records"]["issued_body_m_s"], dim=-1) <= VELOCITY_LIMIT_M_S + 1e-6).all().item()) for trace in traces.values()),
        "throttle_domain": all(bool(((trace["records"]["rotor_throttle"] >= 0.0) & (trace["records"]["rotor_throttle"] <= 1.0)).all().item()) for trace in traces.values()),
        "target_throttle_domain": all(bool(((trace["records"]["rotor_target_throttle"] >= 0.0) & (trace["records"]["rotor_target_throttle"] <= 1.0)).all().item()) for trace in traces.values()),
        "physical_thrust_bound": all(bool((trace["records"]["rotor_thrusts_n"] <= max_thrust + 1e-5).all().item()) for trace in traces.values()),
        "reset_replay_exact": all(torch.equal(traces["below_identity"]["records"][key], traces["reset_replay"]["records"][key]) for key in traces["below_identity"]["records"]),
        "manifest_constant": len({trace["executed_json"] for trace in traces.values()}) == 1,
        "complete_steps": sum(trace["records"]["step"].numel() for trace in traces.values()) == TOTAL_STEPS,
    }
    gates["all_gates_pass"] = all(gates.values())
    contract = {"schema": "m7-009-b0-controller-smoke-v1", "device": DEVICE, "dt_s": DT_S, "steps_per_fixture": STEPS_PER_FIXTURE, "fixture_count": FIXTURE_COUNT, "total_steps": TOTAL_STEPS, "fixed_seed": FIXED_SEED, "body_names": tuple(robot.body_names), "gpu_name": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__)}
    payload = {"contract": contract, "traces": traces, "gates": gates}
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cpu(payload), ARGS.output.with_suffix(".pt"))
    bundle_sha = hashlib.sha256(ARGS.output.with_suffix(".pt").read_bytes()).hexdigest()
    ARGS.output.write_text(json.dumps({"contract": contract, "gates": gates, "bundle": ARGS.output.with_suffix(".pt").name, "bundle_sha256": bundle_sha}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS" if gates["all_gates_pass"] else "FAIL", "steps": TOTAL_STEPS}, sort_keys=True)); sys.stdout.flush(); sys.stderr.flush(); os._exit(0 if gates["all_gates_pass"] else 2)

if __name__ == "__main__": main()
