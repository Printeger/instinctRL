#!/usr/bin/env python3
"""Two-environment, eight-policy-step exact B0 TASLAB headless CUDA smoke."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
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
from instinctlab.assets.taslab_uav.config import TASLAB_UAV
from instinctlab.control.baseline_b0 import B0DirectConfig
from instinctlab.evaluation.baseline_environment import (
    B0EnvironmentConfig,
    begin_ready_b0_environment,
)
from instinctlab.evaluation.baselines import (
    ActorAuditStatus,
    CapabilityBinding,
    ExecutionStatus,
    build_run_manifest,
    resolve_baseline,
)
from instinctlab.envs.taslab_direct_env import DECIMATION, TaslabDirectEnvCfg

NUM_ENVS = 2
POLICY_STEPS = 8
PHYSICS_STEPS = 160
ENVIRONMENT_SAMPLES = 320
FIXED_SEED = 7011
VELOCITY_LIMIT_M_S = 2.0
COMMAND_SCHEDULE = (
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((0.25, -0.10, 0.05), (-0.20, 0.15, 0.10)),
    ((3.0, 4.0, 0.0), (-4.0, 0.0, 3.0)),
    ((0.35, 0.10, -0.05), (0.10, -0.25, 0.15)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-3.0, -4.0, 0.0), (0.0, 4.0, -3.0)),
    ((0.10, 0.20, 0.30), (-0.30, 0.20, 0.10)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
)


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _cpu(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    return value


def _manifest(product_commit: str):
    row = resolve_baseline("B0")
    bindings = tuple(
        CapabilityBinding(item.name, item.accepted_provider, item.accepted_evidence_sha256)
        for item in row.capability_requirements
    )
    return build_run_manifest(
        product_commit=product_commit,
        baseline_id="B0",
        config_switch=row.config_switch,
        capability_bindings=bindings,
        scenario_id="M7-011.environment-smoke",
        seed=FIXED_SEED,
        actor_audit_status=ActorAuditStatus.PASS,
        execution_status=ExecutionStatus.READY,
        skip_reason=None,
    )


def _stack_values(values: list[Any]) -> Any:
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    if isinstance(values[0], dict):
        return {key: _stack_values([value[key] for value in values]) for key in values[0]}
    return values


def _stack(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: _stack_values([record[key] for record in records]) for key in records[0]}


def _all_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    return True


def main() -> None:
    bundle_path = ARGS.output.with_suffix(".pt")
    if ARGS.output.exists() or bundle_path.exists():
        raise FileExistsError(ARGS.output)
    product_commit = os.environ["M7_011_PRODUCT_COMMIT"]
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)
    cfg = TaslabDirectEnvCfg()
    cfg.seed = FIXED_SEED
    manifest = _manifest(product_commit)
    env = begin_ready_b0_environment(
        manifest,
        B0EnvironmentConfig(POLICY_STEPS, B0DirectConfig(VELOCITY_LIMIT_M_S), cfg),
        None,
    )
    dtype = env._robot.data.root_state_w.dtype
    pipeline_calls = [0]

    class ForbiddenPipeline:
        def __call__(self, *args, **kwargs):
            pipeline_calls[0] += 1
            raise AssertionError("ControlCorePipeline must not execute in B0 environment mode")

    env._control_pipelines = [ForbiddenPipeline() for _ in range(NUM_ENVS)]
    reset_calls = [0]
    accepted_reset = env._reset_idx

    def traced_reset(env_ids):
        reset_calls[0] += 1
        return accepted_reset(env_ids)

    env._reset_idx = traced_reset
    physics_records: list[dict[str, Any]] = []
    accepted_apply = env._apply_action
    current_policy = [0]

    def traced_apply():
        command = env._b0_current_command.detach().clone()
        root_before = env._robot.data.root_state_w.detach().clone()
        accepted_apply()
        control = env._control
        lee = env._lee
        wrench = env._wrench
        physics_records.append({
            "physics_step": torch.tensor(env.physics_steps, device=env.device, dtype=torch.long),
            "policy_step": torch.tensor(current_policy[0], device=env.device, dtype=torch.long),
            "command_body_m_s": command,
            "root_before_w": root_before,
            "issued_body_m_s": control.v_final_b.detach().clone(),
            "issued_world_m_s": control.v_final_w.detach().clone(),
            "alpha": control.alpha.detach().clone(),
            "correction_body_m_s": control.v_corr_b.detach().clone(),
            "governed_body_m_s": control.v_gov_b.detach().clone(),
            "anchor_metrics": _cpu(control.anchor_metrics),
            "ics_metrics": _cpu(control.ics_metrics),
            "lee": _cpu(lee),
            "rotor": _cpu(wrench.rotor),
            "forces_body_n": wrench.forces_body_n.detach().clone(),
            "torques_body_nm": wrench.torques_body_nm.detach().clone(),
            "body_identity_mismatch": torch.count_nonzero(control.v_gov_b != control.v_final_b),
            "lee_wrench_mismatch": torch.count_nonzero(
                lee.normalized_rotor_commands != wrench.rotor.normalized_commands
            ),
        })

    env._apply_action = traced_apply
    policy_records = []
    for policy_index, values in enumerate(COMMAND_SCHEDULE):
        current_policy[0] = policy_index
        command = torch.tensor(values, device=env.device, dtype=dtype)
        observation, reward, terminated, truncated, extras = env.step(command)
        if bool(terminated.any().item()) or bool(truncated.any().item()):
            raise AssertionError("bounded B0 smoke must not terminate or reset")
        task = extras["task"]
        policy_records.append({
            "policy_step": torch.tensor(policy_index, device=env.device, dtype=torch.long),
            "command_body_m_s": command.detach().clone(),
            "root_after_w": env._robot.data.root_state_w.detach().clone(),
            "reward": reward.detach().clone(),
            "terminated": terminated.detach().clone(),
            "truncated": truncated.detach().clone(),
            "collision": task["collision"].detach().clone(),
            "reason_code": task["reason_code"].detach().clone(),
            "exact_clearance_m": task["exact_clearance_m"].detach().clone(),
            "episode_length": env.episode_length_buf.detach().clone(),
            "policy_lidar_shape": tuple(observation["policy"]["lidar_grid"].shape),
            "policy_state_shape": tuple(observation["policy"]["state_vec"].shape),
        })
    result = env.finalize()
    physics = _stack(physics_records)
    policies = _stack(policy_records)
    rotor = physics["rotor"]
    max_thrust = TASLAB_UAV.rotors.force_constants[0] * TASLAB_UAV.rotors.max_angular_velocities[0] ** 2
    gates = {
        "exact_policy_count": result.coverage.policy_steps == POLICY_STEPS,
        "exact_physics_count": result.coverage.physics_steps == PHYSICS_STEPS,
        "exact_lifecycle_count": result.coverage.lifecycle_calls == PHYSICS_STEPS,
        "exact_wrench_count": result.coverage.wrench_applications == PHYSICS_STEPS,
        "exact_sample_count": result.coverage.environment_samples == ENVIRONMENT_SAMPLES,
        "exact_receipt_count": result.receipt.step_count == PHYSICS_STEPS,
        "one_ready_one_executed": result.receipt.ready_manifest.manifest_sha256 != result.receipt.executed_manifest.manifest_sha256,
        "no_pipeline_calls": pipeline_calls[0] == 0,
        "no_reset_calls": reset_calls[0] == 0,
        "no_collision": not bool(policies["collision"].any().item()),
        "no_done": not bool(policies["terminated"].any().item()) and not bool(policies["truncated"].any().item()),
        "all_finite": _all_finite({"physics": physics, "policies": policies}),
        "body_identity_exact": int(physics["body_identity_mismatch"].sum().item()) == 0,
        "lee_wrench_identity_exact": int(physics["lee_wrench_mismatch"].sum().item()) == 0,
        "alpha_one": bool((physics["alpha"] == 1.0).all().item()),
        "correction_zero": int(torch.count_nonzero(physics["correction_body_m_s"]).item()) == 0,
        "anchor_inactive": int(torch.count_nonzero(physics["anchor_metrics"]["active"]).item()) == 0,
        "ics_disabled": bool((physics["ics_metrics"]["beta"] == 1.0).all().item()) and int(torch.count_nonzero(physics["ics_metrics"]["emergency"]).item()) == 0,
        "rotor_command_domain": bool(((rotor["normalized_commands"] >= -1.0) & (rotor["normalized_commands"] <= 1.0)).all().item()),
        "throttle_domain": bool(((rotor["throttle"] >= 0.0) & (rotor["throttle"] <= 1.0)).all().item()),
        "physical_thrust_bound": bool(((rotor["thrusts_n"] >= 0.0) & (rotor["thrusts_n"] <= max_thrust)).all().item()),
        "state_domain_bound": bool((physics["root_before_w"][..., :3].abs() <= 100.0).all().item()) and bool((physics["root_before_w"][..., 7:].abs() <= 1000.0).all().item()),
        "final_tensor_caches_cleared": env._b0_current_command is None and env._control is None and env._lee is None and env._wrench is None,
    }
    gates["all_gates_pass"] = all(gates.values())
    receipt = {
        "schema_version": result.receipt.schema_version,
        "ready_manifest_sha256": result.receipt.ready_manifest.manifest_sha256,
        "executed_manifest_sha256": result.receipt.executed_manifest.manifest_sha256,
        "step_count": result.receipt.step_count,
        "trajectory_sha256": result.receipt.trajectory_sha256,
        "ready_manifest_json": result.receipt.ready_manifest.canonical_json,
        "executed_manifest_json": result.receipt.executed_manifest.canonical_json,
    }
    contract = {
        "schema": "m7-011-b0-environment-smoke-v1",
        "product_commit": product_commit,
        "num_envs": NUM_ENVS,
        "policy_steps": POLICY_STEPS,
        "decimation": DECIMATION,
        "physics_steps": PHYSICS_STEPS,
        "environment_samples": ENVIRONMENT_SAMPLES,
        "fixed_seed": FIXED_SEED,
        "velocity_limit_m_s": VELOCITY_LIMIT_M_S,
        "device": str(env.device),
        "dtype": str(dtype),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
    }
    payload = {
        "contract": contract,
        "command_schedule": COMMAND_SCHEDULE,
        "physics": physics,
        "policies": policies,
        "receipt": receipt,
        "coverage": _cpu(result.coverage),
        "call_counts": {
            "pipeline": pipeline_calls[0],
            "reset": reset_calls[0],
            "lifecycle": result.coverage.lifecycle_calls,
            "wrench": result.coverage.wrench_applications,
        },
        "gates": gates,
    }
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cpu(payload), bundle_path)
    bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    summary = {
        "contract": contract,
        "receipt": receipt,
        "coverage": _cpu(result.coverage),
        "call_counts": payload["call_counts"],
        "gates": gates,
        "bundle": bundle_path.name,
        "bundle_sha256": bundle_sha,
    }
    ARGS.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS" if gates["all_gates_pass"] else "FAIL", "policy_steps": result.coverage.policy_steps, "physics_steps": result.coverage.physics_steps, "trajectory_sha256": result.receipt.trajectory_sha256}, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if gates["all_gates_pass"] else 2)


if __name__ == "__main__":
    main()
