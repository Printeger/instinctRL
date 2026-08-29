#!/usr/bin/env python3
"""Exact two-environment M7-012 B0 environment-to-evaluation CUDA smoke."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, fields, is_dataclass
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
PARSER.add_argument("--product-commit", required=True)
ARGS = PARSER.parse_args()
APP_LAUNCHER = AppLauncher(headless=True, enable_cameras=False)
SIMULATION_APP = APP_LAUNCHER.app

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instinctlab.control.baseline_b0 import B0DirectConfig
from instinctlab.evaluation.baseline_environment import (
    B0EnvironmentConfig,
    begin_ready_b0_environment,
)
from instinctlab.evaluation.baseline_runtime import (
    B0RuntimeEvaluationConfig,
    B0RuntimeEvaluationProvenance,
    begin_ready_b0_runtime_evaluation,
)
from instinctlab.evaluation.baselines import (
    ActorAuditStatus,
    CapabilityBinding,
    ExecutionStatus,
    build_run_manifest,
    resolve_baseline,
)
from instinctlab.evaluation.metrics import (
    EvaluationMetricConfig,
    aggregate_metrics,
    compute_step_metrics,
)
from instinctlab.evaluation.raycaster_observability import (
    acquire_exact_observability_from_source,
    frozen_raycaster_provenance,
)
from instinctlab.evaluation.runtime import distill_exact_runtime_observability
from instinctlab.envs.taslab_direct_env import DECIMATION, TaslabDirectEnvCfg


NUM_ENVS = 2
POLICY_STEPS = 8
METRIC_SAMPLES = 16
PHYSICS_STEPS = 160
ENVIRONMENT_SAMPLES = 320
RAYS_PER_ENVIRONMENT = 21240
RAY_SAMPLES = 339840
FIXED_SEED = 7012
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
actor_calls = [0]
training_calls = [0]


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return OrderedDict((field.name, _cpu(getattr(value, field.name))) for field in fields(value))
    if isinstance(value, dict):
        return OrderedDict((key, _cpu(item)) for key, item in value.items())
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
        scenario_id="M7-012.runtime-evaluation-smoke",
        seed=FIXED_SEED,
        actor_audit_status=ActorAuditStatus.PASS,
        execution_status=ExecutionStatus.READY,
        skip_reason=None,
    )


def _metric_mapping(metric: Any) -> OrderedDict[str, object]:
    return OrderedDict((field.name, getattr(metric, field.name)) for field in fields(metric))


def _reduce(values: torch.Tensor, mask: torch.Tensor, reduction: str) -> tuple[bool, float | None, float | None, int, int]:
    selected = values[mask]
    count = int(selected.numel())
    if count == 0:
        return False, None, None, 0, 0
    if reduction == "minimum":
        numerator = float(selected.min().item())
        return True, numerator, numerator, 1, count
    numerator = float(selected.sum().item())
    value = numerator / count
    if reduction == "root_mean_square":
        value = math.sqrt(value)
    return True, value, numerator, count, count


def _independent_metric_oracle(
    evaluation: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, OrderedDict[str, object]]:
    command = evaluation["commanded_velocity_b_m_s"]
    actual = evaluation["actual_velocity_b_m_s"]
    issued = evaluation["issued_velocity_b_m_s"]
    command_norm = torch.linalg.vector_norm(command, dim=-1)
    null_command = command_norm <= 1.0e-3
    intervention = (evaluation["ics_beta"] < 0.999) | evaluation["ics_emergency"]
    preservation_mask = (~null_command) & (~intervention)
    preservation = torch.zeros_like(command_norm)
    preservation[preservation_mask] = (
        torch.linalg.vector_norm(issued[preservation_mask], dim=-1)
        / command_norm[preservation_mask]
    )
    all_samples = torch.ones((8, 2), dtype=torch.bool, device=command.device)
    specs = (
        ("tracking_rmse_m_s", (actual - command).square().sum(dim=-1), all_samples, "root_mean_square", "m/s"),
        ("null_station_drift_mean_m", torch.linalg.vector_norm(evaluation["position_w_m"] - evaluation["station_reference_position_w_m"], dim=-1), null_command, "mean", "m"),
        ("null_station_speed_mean_m_s", torch.linalg.vector_norm(actual, dim=-1), null_command, "mean", "m/s"),
        ("anchor_error_mean_m", evaluation["anchor_error_m"], evaluation["anchor_valid"], "mean", "m"),
        ("minimum_clearance_m", evaluation["minimum_clearance_m"], evaluation["clearance_valid"], "minimum", "m"),
        ("collision_rate", evaluation["collision"].to(command.dtype), all_samples, "mean", "fraction"),
        ("ics_violation_rate", (evaluation["ics_worst_margin_m_s"] < 0.0).to(command.dtype), all_samples, "mean", "fraction"),
        ("command_preservation_ratio", preservation, preservation_mask, "mean", "ratio"),
        ("intervention_frequency", intervention.to(command.dtype), all_samples, "mean", "fraction"),
        ("observability_mean", evaluation["observability_value"], evaluation["observability_valid"], "mean", "source-defined"),
    )
    output = OrderedDict()
    for name, values, mask, reduction, unit in specs:
        available, value, numerator, denominator, valid_count = _reduce(values, mask, reduction)
        output[name] = OrderedDict((
            ("name", name), ("reduction", reduction), ("unit", unit),
            ("available", available), ("value", value), ("numerator", numerator),
            ("denominator", denominator), ("valid_count", valid_count),
            ("total_count", METRIC_SAMPLES),
        ))
    return output


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    bundle_path = ARGS.output.with_suffix(".pt")
    if ARGS.output.exists() or bundle_path.exists():
        raise FileExistsError(ARGS.output)
    if len(ARGS.product_commit) != 40:
        raise ValueError("product commit must be one full Git SHA")
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)
    torch.use_deterministic_algorithms(True)
    cfg = TaslabDirectEnvCfg()
    cfg.seed = FIXED_SEED
    manifest = _manifest(ARGS.product_commit)
    env = begin_ready_b0_environment(
        manifest,
        B0EnvironmentConfig(POLICY_STEPS, B0DirectConfig(VELOCITY_LIMIT_M_S), cfg),
        None,
    )
    initial_observation = env._get_observations()
    collector = begin_ready_b0_runtime_evaluation(
        manifest,
        initial_observation["critic"].detach().clone(),
        B0RuntimeEvaluationConfig(
            POLICY_STEPS, NUM_ENVS, RAYS_PER_ENVIRONMENT,
            EvaluationMetricConfig(),
            B0RuntimeEvaluationProvenance(
                ARGS.product_commit,
                "m7-012-b0-two-env-eight-step-v1",
                "run_m7_b0_runtime_evaluation_smoke.py",
                FIXED_SEED,
            ),
        ),
    )

    class ForbiddenPipeline:
        def __call__(self, *args, **kwargs):
            actor_calls[0] += 1
            raise AssertionError("learned/control pipeline must not execute in B0 collection")

    env._control_pipelines = [ForbiddenPipeline() for _ in range(NUM_ENVS)]
    reset_calls = [0]
    accepted_reset = env._reset_idx

    def traced_reset(env_ids):
        reset_calls[0] += 1
        return accepted_reset(env_ids)

    env._reset_idx = traced_reset
    environment_step_calls = [0]
    staged_identity_passes = [0]
    source_steps = []
    for step, values in enumerate(COMMAND_SCHEDULE, 1):
        original = torch.tensor(values, device=env.device, dtype=torch.float32)
        original_snapshot = original.detach().clone()
        staged = collector.prepare_command(original)
        if staged is original or staged.untyped_storage().data_ptr() == original.untyped_storage().data_ptr():
            raise AssertionError("collector must own one independent staged command")
        if not torch.equal(original, original_snapshot) or not torch.equal(staged, original_snapshot):
            raise AssertionError("prepare_command changed command values")
        environment_step_calls[0] += 1
        passed_to_environment = staged
        observation, reward, terminated, truncated, extras = env.step(passed_to_environment)
        if passed_to_environment is not staged:
            raise AssertionError("the collector-returned staged command was not the sole environment input")
        staged_identity_passes[0] += 1
        if bool(terminated.any().item()) or bool(truncated.any().item()):
            raise AssertionError("bounded B0 collection must not terminate or reset")
        source = extras["evaluation_ray_source"]
        acquisition = acquire_exact_observability_from_source(source)
        exact = distill_exact_runtime_observability(source, acquisition)
        source_steps.append(exact.source_step.detach().cpu().clone())
        del acquisition
        collector.commit_transition(
            staged_command=staged,
            pre_reset=extras["pre_reset"],
            exact_observability=exact,
            returned_critic=observation["critic"].detach().clone(),
            returned_episode_length=env.episode_length_buf.detach().clone(),
        )
    environment_result = env.finalize()
    result = collector.finalize(environment_result)
    evaluation = OrderedDict(result.evaluation_inputs)
    details = OrderedDict(result.observability_details)
    step_metrics = compute_step_metrics(evaluation, EvaluationMetricConfig())
    aggregate = aggregate_metrics(step_metrics)
    independent = _independent_metric_oracle(evaluation)
    accepted = OrderedDict((metric.name, _metric_mapping(metric)) for metric in aggregate.metrics)
    if independent != accepted:
        raise AssertionError("independent metric oracle disagrees with accepted unchanged formulas")
    record_mapping = result.record.to_mapping()
    retention = OrderedDict((
        ("result_field_names", tuple(field.name for field in fields(result))),
        ("evaluation_input_names", tuple(evaluation)),
        ("observability_detail_names", tuple(details)),
        ("forbidden_public_fields", ()),
        ("active_row_cache_count", len(collector._rows)),
        ("active_detail_cache_count", len(collector._details)),
        ("station_reference_cleared", collector._station_reference is None),
        ("pending_command_cleared", not collector.has_pending_command),
    ))
    forbidden = {
        "ray_starts_w", "ray_directions_w", "accepted_ray_hits_w", "accepted_beam_valid",
        "surface_normals_w", "raw_surface_normals_w", "mesh", "lidar_grid", "history",
    }
    exposed = set(retention["result_field_names"]) | set(evaluation) | set(details)
    retention["forbidden_public_fields"] = tuple(sorted(exposed & forbidden))
    call_counts = OrderedDict((
        ("actor", actor_calls[0]), ("training", training_calls[0]),
        ("environment_step", environment_step_calls[0]),
        ("staged_identity", staged_identity_passes[0]),
        ("reset", reset_calls[0]), ("policy", result.contract.policy_steps),
        ("metric_samples", result.contract.metric_samples),
        ("physics", result.contract.physics_steps),
        ("lifecycle", result.contract.lifecycle_calls),
        ("wrench", result.contract.wrench_applications),
        ("environment_samples", result.contract.environment_samples),
        ("rays", result.contract.ray_samples),
    ))
    gates = OrderedDict((
        ("exact_counts", tuple(call_counts.values()) == (0, 0, 8, 8, 0, 8, 16, 160, 160, 160, 320, 339840)),
        ("source_steps_exact", torch.equal(torch.stack(source_steps), torch.arange(1, 9).unsqueeze(1).expand(-1, 2))),
        ("command_schedule_exact", torch.equal(evaluation["commanded_velocity_b_m_s"].cpu(), torch.tensor(COMMAND_SCHEDULE))),
        ("metric_oracle_exact", independent == accepted),
        ("canonical_round_trip", json.dumps(record_mapping, ensure_ascii=True, allow_nan=False, separators=(",", ":")) == result.record.canonical_json),
        ("receipt_exact", result.ready_manifest_sha256 == environment_result.receipt.ready_manifest.manifest_sha256 and result.executed_manifest_sha256 == environment_result.receipt.executed_manifest.manifest_sha256),
        ("no_reset_or_done", not bool(evaluation["terminated"].any().item()) and not bool(evaluation["truncated"].any().item())),
        ("observability_exact", bool(evaluation["observability_source"].eq(2).all().item())),
        ("privacy_retention", retention["forbidden_public_fields"] == ()),
        ("caches_cleared", retention["active_row_cache_count"] == 0 and retention["active_detail_cache_count"] == 0 and retention["station_reference_cleared"] and retention["pending_command_cleared"]),
        ("all_finite", all(bool(torch.isfinite(value).all().item()) for value in evaluation.values() if value.is_floating_point()) and all(bool(torch.isfinite(value).all().item()) for value in details.values() if value.is_floating_point())),
    ))
    gates["all_gates_pass"] = all(gates.values())
    contract = OrderedDict((
        ("schema", "m7-012-b0-runtime-evaluation-smoke-v1"),
        ("product_commit", ARGS.product_commit), ("num_envs", NUM_ENVS),
        ("policy_steps", POLICY_STEPS), ("metric_samples", METRIC_SAMPLES),
        ("physics_steps", PHYSICS_STEPS), ("environment_samples", ENVIRONMENT_SAMPLES),
        ("rays_per_environment", RAYS_PER_ENVIRONMENT), ("ray_samples", RAY_SAMPLES),
        ("fixed_seed", FIXED_SEED), ("device", str(env.device)),
        ("gpu_name", torch.cuda.get_device_name(0)), ("torch_version", str(torch.__version__)),
        ("collector_source_sha256", _hash(Path(__file__).resolve().parents[1] / "instinctlab/evaluation/baseline_runtime.py")),
        ("raycaster_provenance", frozen_raycaster_provenance()),
    ))
    receipt = OrderedDict((
        ("schema_version", environment_result.receipt.schema_version),
        ("ready_manifest_sha256", result.ready_manifest_sha256),
        ("executed_manifest_sha256", result.executed_manifest_sha256),
        ("trajectory_sha256", result.trajectory_sha256),
        ("step_count", environment_result.receipt.step_count),
        ("coverage", OrderedDict(asdict(result.contract).items())),
    ))
    payload = OrderedDict((
        ("contract", contract), ("command_schedule", COMMAND_SCHEDULE),
        ("environment_receipt", receipt),
        ("evaluation_inputs", _cpu(evaluation)),
        ("observability_details", _cpu(details)),
        ("command_identity_sha256", _cpu(result.command_identity_sha256)),
        ("source_identity_sha256", _cpu(result.source_identity_sha256)),
        ("result_identity_sha256", _cpu(result.result_identity_sha256)),
        ("metric_oracle", independent), ("canonical_record", result.record.canonical_json),
        ("record_mapping", record_mapping), ("call_counts", call_counts),
        ("retention", retention), ("gates", gates),
    ))
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, bundle_path)
    summary = OrderedDict((
        ("status", "PASS" if gates["all_gates_pass"] else "FAIL"),
        ("contract", contract), ("environment_receipt", receipt),
        ("metric_oracle", independent), ("call_counts", call_counts),
        ("retention", retention), ("gates", gates), ("bundle", bundle_path.name),
        ("bundle_sha256", _hash(bundle_path)),
        ("record_sha256", hashlib.sha256(result.record.canonical_json.encode("ascii")).hexdigest()),
    ))
    ARGS.output.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")
    print(json.dumps({
        "status": summary["status"], "policy_steps": POLICY_STEPS,
        "metric_samples": METRIC_SAMPLES, "physics_steps": PHYSICS_STEPS,
        "ray_samples": RAY_SAMPLES,
    }, sort_keys=True))
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if gates["all_gates_pass"] else 2)


if __name__ == "__main__":
    main()
