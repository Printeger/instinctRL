"""Two-case deterministic evaluation orchestration with no performance gate."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import struct
from typing import Any, Protocol

from .config import load_config
from .semantic import strict_canonical_record, validate_evaluation_preflight


TRACKING_COMMANDS = (
    (0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.2, 0.0),
    (0.0, -0.2, 0.0), (0.15, 0.15, 0.0), (-0.15, -0.15, 0.0), (0.0, 0.0, 0.0),
)


def _float32_command_at(case: str, policy_step: int) -> list[float]:
    """Return the exact JSON-visible values produced by the product float32 tensor."""
    return [
        struct.unpack("<f", struct.pack("<f", component))[0]
        for component in evaluation_command_at(case, policy_step)
    ]


def evaluation_command_at(case: str, policy_step: int) -> tuple[float, float, float]:
    if case not in {"station", "tracking"} or isinstance(policy_step, bool) or not isinstance(policy_step, int) or not 0 <= policy_step <= 256:
        raise ValueError("evaluation case/global step mismatch")
    if case == "station":
        return (0.0, 0.0, 0.0)
    return TRACKING_COMMANDS[min(policy_step, 255) // 32]


class EvaluationBackend(Protocol):
    def bootstrap(self, seed: int) -> None: ...
    def create_environment(self, case: str) -> None: ...
    def reset_environment(self, seed: int) -> None: ...
    def load_checkpoint(self, path: Path, counter: int) -> None: ...
    def run_initial_preflight(self) -> Mapping[str, Any]: ...
    def disable_gradients(self) -> None: ...
    def evaluate_step(self, case: str, index: int, command: tuple[float, float, float]) -> Mapping[str, Any]: ...
    def finalize(self) -> Mapping[str, Any]: ...
    def preserve_partial(self, error: BaseException) -> None: ...


def run_evaluation(case: str, checkpoint: Path, backend: EvaluationBackend, progress: Any) -> dict[str, Any]:
    config = load_config()
    evaluation = config["evaluation"]
    if case not in evaluation["cases"]:
        raise ValueError("unknown evaluation case")
    completed = 0
    rows: list[dict[str, Any]] = []
    failure_stage = "bootstrap"
    try:
        backend.bootstrap(evaluation["environment_seed"])
        failure_stage = "environment_creation"
        backend.create_environment(case)
        failure_stage = "environment_reset"
        backend.reset_environment(evaluation["environment_seed"])
        failure_stage = "checkpoint_load"
        backend.load_checkpoint(checkpoint, evaluation["target_checkpoint_counter"])
        failure_stage = "initial_preflight"
        preflight = dict(backend.run_initial_preflight())
        failure_stage = "gradient_disable"
        backend.disable_gradients()
        for index in range(256):
            command = evaluation_command_at(case, index)
            progress.write(status="RUNNING", case=case, attempted_transitions=(index + 1) * 2, completed_transitions=index * 2, completed_policy_steps=index)
            failure_stage = "evaluation_step"
            row = dict(backend.evaluate_step(case, index, command))
            if row.get("global_index") != index or tuple(row.get("command_b_m_s", ())) != command or row.get("actor_keys") != ["lidar_grid", "state_vec"]:
                raise ValueError("evaluation row violates command/actor alignment")
            rows.append(row)
            completed += 2
        failure_stage = "evaluation_finalize"
        summary = dict(backend.finalize())
        result = {
            "schema_version": "scale-003-evaluation-semantic.v1", "status": "PASS", "case": case,
            "checkpoint_counter": 1954, "num_envs": 2, "policy_steps": 256,
            "environment_transitions": completed, "action_mode": "deterministic_mean",
            "fixture": "EVAL-002 global floor-plus-corner at 4m spacing",
            "performance_gate": False, "training_performed": False, "optimizer_updates": 0,
            "initial_preflight": preflight, "rows": rows, "summary": summary,
        }
        validate_evaluation_semantic(result, case=case)
        progress.write(status="PASS", case=case, attempted_transitions=512, completed_transitions=512, completed_policy_steps=256)
        return result
    except BaseException as error:
        latest = progress.latest if hasattr(progress, "latest") and isinstance(progress.latest, Mapping) else {}
        accounting: dict[str, Any] = {}
        failure_accounting = getattr(backend, "failure_accounting", None)
        if callable(failure_accounting):
            try:
                observed = failure_accounting()
                if isinstance(observed, Mapping): accounting = dict(observed)
            except BaseException:
                accounting = {"accounting_status": "UNAVAILABLE_DURING_FAILURE_AUDIT"}
        actual_completed = max(completed, int(latest.get("completed_transitions", completed)), int(accounting.get("completed_transitions", completed)))
        attempted = max(actual_completed, int(latest.get("attempted_transitions", completed)))
        progress.fail(error, case=case, failure_stage=failure_stage,
                      attempted_transitions=attempted, completed_transitions=actual_completed,
                      completed_policy_steps=int(accounting.get("completed_policy_steps", actual_completed // 2)),
                      accounting_status=accounting.get("accounting_status", "CONFIRMED_BY_BACKEND" if accounting else "LAST_CONFIRMED_BOUNDARY"))
        try:
            backend.preserve_partial(error)
        except BaseException:
            pass
        raise


def validate_evaluation_semantic(value: Mapping[str, Any], *, case: str) -> None:
    required = {"schema_version", "status", "case", "checkpoint_counter", "num_envs", "policy_steps", "environment_transitions", "action_mode", "fixture", "performance_gate", "training_performed", "optimizer_updates", "initial_preflight", "rows", "summary"}
    if set(value) != required or value.get("schema_version") != "scale-003-evaluation-semantic.v1" or value.get("status") != "PASS" or value.get("case") != case:
        raise ValueError("evaluation semantic schema/status mismatch")
    exact = {"checkpoint_counter": 1954, "num_envs": 2, "policy_steps": 256, "environment_transitions": 512, "action_mode": "deterministic_mean", "fixture": "EVAL-002 global floor-plus-corner at 4m spacing", "performance_gate": False, "training_performed": False, "optimizer_updates": 0}
    if any(value.get(key) != expected for key, expected in exact.items()) or len(value.get("rows", ())) != 256:
        raise ValueError("evaluation semantic exact campaign mismatch")
    validate_evaluation_preflight(value["initial_preflight"])
    terminal_count = 0
    for index, row in enumerate(value["rows"]):
        required_row = {"global_index", "command_b_m_s", "actor_keys", "terminal_environment_count", "visibility"}
        if not isinstance(row, Mapping) or set(row) != required_row:
            raise ValueError("evaluation semantic row schema mismatch")
        if row.get("global_index") != index or tuple(row.get("command_b_m_s", ())) != evaluation_command_at(case, index) or row.get("actor_keys") != ["lidar_grid", "state_vec"]:
            raise ValueError("evaluation semantic schedule mismatch")
        if type(row["terminal_environment_count"]) is not int or not 0 <= row["terminal_environment_count"] <= 2:
            raise ValueError("evaluation terminal row count mismatch")
        visibility = row["visibility"]
        if not isinstance(visibility, Mapping) or set(visibility) != {"floor", "x_wall", "y_wall"} or any(not isinstance(items, list) or len(items) != 2 or any(type(item) is not int or item < 0 for item in items) for items in visibility.values()):
            raise ValueError("evaluation row visibility evidence mismatch")
        terminal_count += row["terminal_environment_count"]
    summary = value["summary"]
    required_summary = {"canonical_metrics_available", "terminal_samples_preserved", "zero_denominator_is_unavailable", "runtime_evaluation", "canonical_record", "reset_count", "checkpoint_state_unchanged"}
    if not isinstance(summary, Mapping) or set(summary) != required_summary:
        raise ValueError("evaluation summary schema mismatch")
    if summary.get("canonical_metrics_available") is not True or summary.get("terminal_samples_preserved") is not True or summary.get("zero_denominator_is_unavailable") is not True or summary.get("checkpoint_state_unchanged") is not True:
        raise ValueError("evaluation metric/terminal semantics mismatch")
    runtime = summary["runtime_evaluation"]
    runtime_keys = {"schema_version", "contract", "source_manifest", "policy_hashes", "action_hashes", "station_reference_before_w_m", "station_reference_after_w_m", "reset_masks", "returned_episode_lengths", "evaluation_inputs", "observability_details", "canonical_record", "gates"}
    if not isinstance(runtime, Mapping) or set(runtime) != runtime_keys or runtime["schema_version"] != "m7.runtime-evaluation.v2":
        raise ValueError("runtime evaluation evidence schema mismatch")
    if runtime["contract"] != {"num_envs": 2, "policy_steps": 256, "samples": 512, "rays_per_environment": 21240, "ray_samples": 10874880}:
        raise ValueError("runtime evaluation coverage contract mismatch")
    if not isinstance(runtime["policy_hashes"], list) or len(runtime["policy_hashes"]) != 256 or not isinstance(runtime["action_hashes"], list) or len(runtime["action_hashes"]) != 256:
        raise ValueError("runtime policy/action hash coverage mismatch")
    inputs = runtime["evaluation_inputs"]
    from .semantic import EVALUATION_INPUT_KEYS, METRIC_KEYS, OBSERVABILITY_DETAIL_KEYS, SOURCE_METADATA_KEYS, require_sha256
    if not isinstance(inputs, Mapping) or set(inputs) != set(EVALUATION_INPUT_KEYS):
        raise ValueError("runtime evaluation input keys mismatch")
    if any(not isinstance(items, list) or len(items) != 256 or any(not isinstance(row, list) or len(row) != 2 for row in items) for items in inputs.values()):
        raise ValueError("runtime evaluation input dimensions mismatch")
    vector_inputs = {"commanded_velocity_b_m_s", "actual_velocity_b_m_s", "issued_velocity_b_m_s", "position_w_m", "station_reference_position_w_m"}
    boolean_inputs = {"anchor_valid", "clearance_valid", "collision", "ics_emergency", "terminated", "truncated", "termination_collision", "termination_below_height", "termination_above_height", "termination_timeout", "observability_valid"}
    integer_inputs = {"termination_reason", "observability_source"}
    for name, items in inputs.items():
        if name in vector_inputs:
            if any(not isinstance(environment, list) or len(environment) != 3 for row in items for environment in row):
                raise ValueError(f"runtime vector input {name} shape mismatch")
            scalars = [scalar for row in items for environment in row for scalar in environment]
        else:
            scalars = [environment for row in items for environment in row]
        if name in boolean_inputs:
            valid = all(type(scalar) is bool for scalar in scalars)
        elif name in integer_inputs:
            valid = all(type(scalar) is int for scalar in scalars)
        else:
            valid = all(isinstance(scalar, (int, float)) and not isinstance(scalar, bool) for scalar in scalars)
        if not valid:
            raise ValueError(f"runtime input {name} scalar type mismatch")
    expected_commands = [[_float32_command_at(case, index) for _ in range(2)] for index in range(256)]
    if inputs["commanded_velocity_b_m_s"] != expected_commands:
        raise ValueError("runtime command inputs do not match the diagnostic schedule")
    source_manifest = runtime["source_manifest"]
    if not isinstance(source_manifest, Mapping) or set(source_manifest) != set(EVALUATION_INPUT_KEYS) or any(not isinstance(row, Mapping) or set(row) != set(SOURCE_METADATA_KEYS) for row in source_manifest.values()):
        raise ValueError("runtime source manifest identity/schema mismatch")
    from instinctlab.evaluation.runtime import SOURCE_MANIFEST
    if source_manifest != SOURCE_MANIFEST:
        raise ValueError("runtime source manifest differs from the pinned product contract")
    for digest in (*runtime["policy_hashes"], *runtime["action_hashes"]): require_sha256(digest, "runtime row hash")
    reset_masks = runtime["reset_masks"]
    if not isinstance(reset_masks, list) or len(reset_masks) != 256 or any(not isinstance(row, list) or len(row) != 2 or any(type(item) is not bool for item in row) for row in reset_masks):
        raise ValueError("runtime reset mask dimensions/type mismatch")
    reset_count = sum(sum(row) for row in reset_masks)
    episode_lengths = runtime["returned_episode_lengths"]
    if not isinstance(episode_lengths, list) or len(episode_lengths) != 256 or any(not isinstance(row, list) or len(row) != 2 or any(type(item) is not int or item < 0 for item in row) for row in episode_lengths):
        raise ValueError("runtime returned episode-length evidence mismatch")
    station_before = runtime["station_reference_before_w_m"]
    station_after = runtime["station_reference_after_w_m"]
    if any(
        not isinstance(rows, list) or len(rows) != 256
        or any(not isinstance(row, list) or len(row) != 2 or any(not isinstance(point, list) or len(point) != 3 for point in row) for row in rows)
        for rows in (station_before, station_after)
    ):
        raise ValueError("runtime station-reference dimensions mismatch")
    if station_before != inputs["station_reference_position_w_m"] or any(station_after[index] != station_before[index + 1] for index in range(255)):
        raise ValueError("runtime station-reference ownership mismatch")
    gates = runtime["gates"]
    if gates != {"coverage_exact": True, "all_finite": True, "observability_exact": True, "reset_count": reset_count}:
        raise ValueError("runtime gates/reset count mismatch")
    if summary["reset_count"] != reset_count or terminal_count != reset_count:
        raise ValueError("terminal rows/runtime reset count mismatch")
    details = runtime["observability_details"]
    if not isinstance(details, Mapping) or set(details) != set(OBSERVABILITY_DETAIL_KEYS):
        raise ValueError("runtime exact-observability detail schema mismatch")
    for name, items in details.items():
        if not isinstance(items, list) or len(items) != 256 or any(not isinstance(row, list) or len(row) != 2 for row in items):
            raise ValueError(f"runtime observability detail {name} coverage mismatch")
    for name in ("source_identity_sha256", "result_identity_sha256"):
        if any(not isinstance(digest, list) or len(digest) != 32 or any(type(byte) is not int or not 0 <= byte <= 255 for byte in digest) for row in details[name] for digest in row):
            raise ValueError(f"runtime observability detail {name} identity mismatch")
    canonical = strict_canonical_record(runtime["canonical_record"])
    if summary["canonical_record"] != canonical:
        raise ValueError("canonical record mirror mismatch")
    if sum(canonical["observability_source_counts"].values()) != 512 or set(canonical["metrics"]) != set(METRIC_KEYS):
        raise ValueError("canonical record source/metric coverage mismatch")
    provenance = canonical["provenance"]
    if provenance != {"product_commit": load_config()["pins"]["product_commit"],
                      "fixture_id": f"scale-003-eval-002-{case}",
                      "generator_id": "training_workloads/scale_003/formal_evaluation.py", "seed": 20260905}:
        raise ValueError("canonical record source provenance mismatch")
    _validate_metric_replay(inputs, canonical)


def _validate_metric_replay(inputs: Mapping[str, Any], canonical: Mapping[str, Any]) -> None:
    """Replay metric counts and exact record arithmetic using the frozen validator policy."""
    from collections import OrderedDict
    import math
    import torch
    from .semantic import EVALUATION_INPUT_KEYS, METRIC_KEYS
    from instinctlab.evaluation.metrics import (
        ObservabilitySource, aggregate_metrics, compute_step_metrics,
    )
    from instinctlab.task.termination import (
        TERMINATION_ABOVE_HEIGHT, TERMINATION_BELOW_HEIGHT,
        TERMINATION_COLLISION, TERMINATION_NONE, TERMINATION_TIMEOUT,
    )

    vector = {"commanded_velocity_b_m_s", "actual_velocity_b_m_s", "issued_velocity_b_m_s", "position_w_m", "station_reference_position_w_m"}
    boolean = {"anchor_valid", "clearance_valid", "collision", "ics_emergency", "terminated", "truncated", "termination_collision", "termination_below_height", "termination_above_height", "termination_timeout", "observability_valid"}
    integer = {"termination_reason", "observability_source"}
    tensors = OrderedDict()
    for name in EVALUATION_INPUT_KEYS:
        rows = inputs[name]
        dtype = torch.bool if name in boolean else torch.long if name in integer else torch.float32
        tensor = torch.tensor(rows, dtype=dtype)
        expected_shape = (256, 2, 3) if name in vector else (256, 2)
        if tensor.shape != expected_shape:
            raise ValueError(f"runtime input {name} shape mismatch")
        tensors[name] = tensor
    replay = aggregate_metrics(compute_step_metrics(tensors))
    actual = {metric.name: metric for metric in replay.metrics}
    exact_numeric_replay = {
        "minimum_clearance_m",
        "collision_rate",
        "ics_violation_rate",
        "intervention_frequency",
    }
    float_reduction_self_consistency = {
        "tracking_rmse_m_s",
        "null_station_drift_mean_m",
        "null_station_speed_mean_m_s",
        "anchor_error_mean_m",
        "command_preservation_ratio",
        "observability_mean",
    }
    if exact_numeric_replay | float_reduction_self_consistency != set(METRIC_KEYS) or exact_numeric_replay & float_reduction_self_consistency:
        raise RuntimeError("metric replay classification does not cover the frozen metric set")
    for name in METRIC_KEYS:
        row = canonical["metrics"][name]
        metric = actual[name]
        if (
            row["reduction"] != metric.reduction
            or row["unit"] != metric.unit
            or row["available"] != metric.available
            or row["denominator"] != metric.denominator
            or row["valid_count"] != metric.valid_count
            or row["total_count"] != metric.total_count
        ):
            raise ValueError(f"canonical metric identity/count replay mismatch: {name}")
        if not metric.available:
            if (row["value"], row["numerator"], row["denominator"], row["valid_count"]) != (None, None, 0, 0):
                raise ValueError(f"canonical metric UNAVAILABLE representation mismatch: {name}")
            continue
        if name in exact_numeric_replay and (row["numerator"], row["value"]) != (metric.numerator, metric.value):
            raise ValueError(f"canonical exact metric facts do not match source samples: {name}")
        numerator, value, denominator = row["numerator"], row["value"], row["denominator"]
        if type(numerator) is not float or type(value) is not float or not math.isfinite(numerator) or not math.isfinite(value):
            raise ValueError(f"canonical metric must contain finite float facts: {name}")
        if row["reduction"] == "minimum":
            consistent = denominator == 1 and value == numerator
        elif row["reduction"] == "root_mean_square":
            consistent = denominator == row["valid_count"] and value == math.sqrt(numerator / denominator)
        else:
            consistent = denominator == row["valid_count"] and value == numerator / denominator
        if not consistent:
            raise ValueError(f"canonical metric aggregation facts mismatch: {name}")

    source = tensors["observability_source"]
    expected_sources = {
        label.name: int((source == int(label)).sum().item())
        for label in (ObservabilitySource.UNAVAILABLE, ObservabilitySource.PROXY, ObservabilitySource.EXACT)
    }
    reason = tensors["termination_reason"]
    expected_termination = {
        "collision": int(tensors["termination_collision"].sum().item()),
        "below_height": int(tensors["termination_below_height"].sum().item()),
        "above_height": int(tensors["termination_above_height"].sum().item()),
        "timeout": int(tensors["termination_timeout"].sum().item()),
        "terminated": int(tensors["terminated"].sum().item()),
        "truncated": int(tensors["truncated"].sum().item()),
        "done": int((tensors["terminated"] | tensors["truncated"]).sum().item()),
        "simultaneous_terminated_truncated": int((tensors["terminated"] & tensors["truncated"]).sum().item()),
        "reason_none": int((reason == TERMINATION_NONE).sum().item()),
        "reason_timeout": int((reason == TERMINATION_TIMEOUT).sum().item()),
        "reason_above_height": int((reason == TERMINATION_ABOVE_HEIGHT).sum().item()),
        "reason_below_height": int((reason == TERMINATION_BELOW_HEIGHT).sum().item()),
        "reason_collision": int((reason == TERMINATION_COLLISION).sum().item()),
    }
    if canonical["observability_source_counts"] != expected_sources:
        raise ValueError("canonical observability source counts do not match replay")
    if canonical["termination_counts"] != expected_termination:
        raise ValueError("canonical termination counts do not match replay")
