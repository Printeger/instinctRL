"""Strict CPU-safe validators shared by training and evaluation evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .config import loads_strict_json, sha256_file
from .geometry import MESH_PRIM_PATH, validate_exact_observability_gate
from .resources import RUNTIME_MIN_FREE_MEMORY_BYTES, validate_coexistence_window


SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def require_finite_tree(value: Any, label: str = "evidence") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite value")
    if isinstance(value, Mapping):
        for key, item in value.items():
            require_finite_tree(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_tree(item, f"{label}[{index}]")


def validate_prelaunch_resource_gate(value: Any, *, mode: str | None = None) -> None:
    required = {
        "schema_version", "status", "sample_count", "span_ns", "samples", "mode", "attempt_id", "pid",
        "authorization_sha256", "frozen_gate_source_sha256", "nvidia_smi_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("prelaunch resource gate schema mismatch")
    if value["schema_version"] != "scale-003-coexistence-window.v1" or value["status"] != "PASS":
        raise ValueError("prelaunch resource gate did not pass")
    if mode is not None and value["mode"] != mode:
        raise ValueError("prelaunch resource gate mode mismatch")
    if not isinstance(value["attempt_id"], str) or not value["attempt_id"] or type(value["pid"]) is not int or value["pid"] <= 0:
        raise ValueError("prelaunch resource gate process binding mismatch")
    for field in ("authorization_sha256", "frozen_gate_source_sha256", "nvidia_smi_sha256"):
        require_sha256(value[field], field)
    validate_coexistence_window({key: value[key] for key in ("sample_count", "span_ns", "samples")})


def validate_runtime_resource(value: Any, *, expected_stage: str | None = None) -> None:
    required = {"stage", "ordinal", "free_memory_bytes", "total_memory_bytes", "allocated_memory_bytes", "minimum_free_memory_bytes"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("runtime resource evidence schema mismatch")
    if expected_stage is not None and value["stage"] != expected_stage:
        raise ValueError("runtime resource stage mismatch")
    for field in required - {"stage"}:
        if type(value[field]) is not int or value[field] < 0:
            raise ValueError("runtime resource values must be nonnegative integers")
    if value["minimum_free_memory_bytes"] != RUNTIME_MIN_FREE_MEMORY_BYTES or value["free_memory_bytes"] < RUNTIME_MIN_FREE_MEMORY_BYTES:
        raise ValueError("runtime resource evidence is below the 1 GiB safety line")
    if value["free_memory_bytes"] > value["total_memory_bytes"] or value["allocated_memory_bytes"] > value["total_memory_bytes"]:
        raise ValueError("runtime resource memory accounting mismatch")


def _verify_payload_hash(value: Mapping[str, Any], label: str) -> None:
    digest = require_sha256(value.get("sha256"), f"{label}.sha256")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if actual != digest:
        raise ValueError(f"{label} payload hash mismatch")


def validate_training_preflight(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"geometry", "observability", "prelaunch_resource_gate", "runtime_resource_initial"}:
        raise ValueError("training preflight evidence schema mismatch")
    geometry = value["geometry"]
    required_geometry = {"mesh_prim_path", "points", "triangles", "wall_colliders", "env_origins", "inventory", "coverage_limit", "sha256"}
    if not isinstance(geometry, Mapping) or set(geometry) != required_geometry:
        raise ValueError("training geometry preflight schema mismatch")
    if geometry["mesh_prim_path"] != MESH_PRIM_PATH or (geometry["points"], geometry["triangles"], geometry["wall_colliders"]) != (59424, 113024, 64):
        raise ValueError("training geometry preflight topology mismatch")
    if not isinstance(geometry["env_origins"], list) or len(geometry["env_origins"]) != 32 or any(not isinstance(row, list) or len(row) != 3 for row in geometry["env_origins"]):
        raise ValueError("training geometry preflight origin coverage mismatch")
    if not isinstance(geometry["inventory"], list) or len([row for row in geometry["inventory"] if isinstance(row, Mapping) and row.get("collision") is True and str(row.get("path", "")).startswith("/World/Scale003WallColliders/")]) != 64:
        raise ValueError("training geometry preflight collision inventory mismatch")
    _verify_payload_hash(geometry, "training geometry")
    observability = value["observability"]
    required_observability = {"status", "algorithm", "environments", "surface_hits", "numerical_rank", "sigma_min", "singular_values", "source"}
    if not isinstance(observability, Mapping) or set(observability) != required_observability or observability["source"] != "actual_raw_raycaster_acquisition":
        raise ValueError("training exact observability evidence schema/source mismatch")
    validate_exact_observability_gate(
        surface_hits=observability["surface_hits"], numerical_rank=observability["numerical_rank"],
        sigma_min=observability["sigma_min"], num_envs=32,
    )
    if not isinstance(observability["singular_values"], list) or len(observability["singular_values"]) != 32 or any(not isinstance(row, list) or len(row) != 3 for row in observability["singular_values"]):
        raise ValueError("training exact singular-value evidence mismatch")
    validate_prelaunch_resource_gate(value["prelaunch_resource_gate"], mode="training")
    validate_runtime_resource(value["runtime_resource_initial"], expected_stage="before_environment")
    require_finite_tree(value)


EVALUATION_INPUT_KEYS = (
    "commanded_velocity_b_m_s", "actual_velocity_b_m_s", "issued_velocity_b_m_s", "position_w_m",
    "station_reference_position_w_m", "anchor_error_m", "anchor_valid", "minimum_clearance_m",
    "clearance_valid", "collision", "ics_beta", "ics_emergency", "ics_worst_margin_m_s", "terminated",
    "truncated", "termination_reason", "termination_collision", "termination_below_height",
    "termination_above_height", "termination_timeout", "observability_value", "observability_valid", "observability_source",
)
METRIC_KEYS = (
    "tracking_rmse_m_s", "null_station_drift_mean_m", "null_station_speed_mean_m_s", "anchor_error_mean_m",
    "minimum_clearance_m", "collision_rate", "ics_violation_rate", "command_preservation_ratio",
    "intervention_frequency", "observability_mean",
)
SOURCE_METADATA_KEYS = ("source", "source_shape", "record_shape", "dtype", "units", "frame", "validity", "ownership", "privilege")
OBSERVABILITY_DETAIL_KEYS = (
    "numerical_rank", "singular_values", "condition_number", "condition_valid", "weak_direction_w",
    "weak_direction_valid", "valid_ray_count", "total_ray_count", "valid_beam_fraction",
    "hit_max_error_m", "distance_max_error_m", "source_step", "source_reset_mask",
    "source_identity_sha256", "result_identity_sha256",
)


def strict_canonical_record(text: Any) -> dict[str, Any]:
    if not isinstance(text, str) or not text:
        raise ValueError("canonical record must be a nonempty JSON string")
    value = loads_strict_json(text, source="canonical_record")
    canonical = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    if canonical != text:
        raise ValueError("canonical record bytes are not canonical")
    required = {"schema_version", "platform", "sensor", "claim", "actor_audit", "input_dimensions", "evaluation_input_order", "metric_config", "observability_source_counts", "metrics", "termination_counts", "provenance"}
    if (
        set(value) != required
        or value["schema_version"] != "m7.evaluation.v1"
        or value["platform"] != "TASLAB_UAV"
        or value["sensor"] != "Livox MID360"
        or value["claim"] != "pure-tensor contract record; no simulator evaluation or policy-performance claim"
    ):
        raise ValueError("canonical evaluation record schema mismatch")
    if value["input_dimensions"] != {"time_steps": 256, "environments": 2, "samples": 512} or value["evaluation_input_order"] != list(EVALUATION_INPUT_KEYS):
        raise ValueError("canonical evaluation record dimensions/order mismatch")
    actor = value["actor_audit"]
    actor_fields = {"status", "keys", "lidar_grid_shape", "state_vec_shape", "dtype", "device", "lidar_grid_sha256", "state_vec_sha256"}
    if (
        not isinstance(actor, Mapping) or set(actor) != actor_fields
        or actor["status"] != "PASS" or actor["keys"] != ["lidar_grid", "state_vec"]
        or actor["lidar_grid_shape"] != [2, 12, 59, 360] or actor["state_vec_shape"] != [2, 52]
        or actor["dtype"] != "torch.float32" or not isinstance(actor["device"], str) or not actor["device"]
    ):
        raise ValueError("canonical actor audit mismatch")
    require_sha256(actor["lidar_grid_sha256"], "canonical lidar_grid hash")
    require_sha256(actor["state_vec_sha256"], "canonical state_vec hash")
    if value["metric_config"] != {"schema_version": "m7.evaluation.v1", "null_command_threshold_m_s": 0.001, "full_compliance_beta": 0.999}:
        raise ValueError("canonical metric config mismatch")
    source_counts = value["observability_source_counts"]
    if not isinstance(source_counts, Mapping) or set(source_counts) != {"UNAVAILABLE", "PROXY", "EXACT"} or any(type(item) is not int or item < 0 for item in source_counts.values()):
        raise ValueError("canonical observability source counts mismatch")
    metrics = value["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_KEYS):
        raise ValueError("canonical metric definition keys mismatch")
    for name, metric in metrics.items():
        fields = {"reduction", "unit", "available", "value", "numerator", "denominator", "valid_count", "total_count"}
        if not isinstance(metric, Mapping) or set(metric) != fields or type(metric["available"]) is not bool:
            raise ValueError(f"canonical metric {name} schema mismatch")
        if type(metric["denominator"]) is not int or type(metric["valid_count"]) is not int or metric["total_count"] != 512 or not 0 <= metric["valid_count"] <= 512:
            raise ValueError(f"canonical metric {name} count mismatch")
        if metric["available"]:
            if metric["denominator"] <= 0 or not all(isinstance(metric[field], (int, float)) and not isinstance(metric[field], bool) and math.isfinite(float(metric[field])) for field in ("value", "numerator")):
                raise ValueError(f"canonical metric {name} available-value mismatch")
        elif (metric["value"], metric["numerator"], metric["denominator"], metric["valid_count"]) != (None, None, 0, 0):
            raise ValueError(f"canonical metric {name} UNAVAILABLE semantics mismatch")
    termination_fields = {
        "collision", "below_height", "above_height", "timeout", "terminated", "truncated", "done",
        "simultaneous_terminated_truncated", "reason_none", "reason_timeout", "reason_above_height",
        "reason_below_height", "reason_collision",
    }
    termination = value["termination_counts"]
    if not isinstance(termination, Mapping) or set(termination) != termination_fields or any(type(item) is not int or item < 0 for item in termination.values()):
        raise ValueError("canonical termination count schema mismatch")
    require_finite_tree(value)
    return value


def validate_evaluation_preflight(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"geometry", "observability", "prelaunch_resource_gate", "runtime_resource_initial"}:
        raise ValueError("evaluation preflight evidence schema mismatch")
    geometry = value["geometry"]
    required_geometry = {"mesh_path", "points", "indices", "topology", "inventory", "env_origins", "initial_root_state_w", "sha256"}
    if not isinstance(geometry, Mapping) or set(geometry) != required_geometry or geometry["mesh_path"] != "/World/MID360StaticScene/Eval002CombinedMesh":
        raise ValueError("evaluation geometry evidence schema/path mismatch")
    topology = geometry["topology"]
    if not isinstance(topology, Mapping) or (topology.get("points"), topology.get("triangles"), topology.get("indices")) != (1857, 3532, 10596):
        raise ValueError("evaluation geometry topology mismatch")
    if not isinstance(geometry["points"], list) or len(geometry["points"]) != 1857 or not isinstance(geometry["indices"], list) or len(geometry["indices"]) != 10596:
        raise ValueError("evaluation geometry readback coverage mismatch")
    if not isinstance(geometry["inventory"], list) or not any(isinstance(row, Mapping) and row.get("path") == "/World/MID360StaticScene/Eval002CombinedMesh" and row.get("collision") is False for row in geometry["inventory"]):
        raise ValueError("evaluation sensed-mesh inventory mismatch")
    if not isinstance(geometry["env_origins"], list) or len(geometry["env_origins"]) != 2:
        raise ValueError("evaluation geometry origin coverage mismatch")
    _verify_payload_hash(geometry, "evaluation geometry")
    observability = value["observability"]
    if not isinstance(observability, Mapping) or set(observability) != {"surface_hits", "numerical_rank", "singular_values", "sigma_min"}:
        raise ValueError("evaluation initial observability schema mismatch")
    validate_exact_observability_gate(surface_hits=observability["surface_hits"], numerical_rank=observability["numerical_rank"], sigma_min=observability["sigma_min"], num_envs=2)
    validate_prelaunch_resource_gate(value["prelaunch_resource_gate"], mode="evaluation")
    validate_runtime_resource(value["runtime_resource_initial"], expected_stage="before_environment")
    require_finite_tree(value)
