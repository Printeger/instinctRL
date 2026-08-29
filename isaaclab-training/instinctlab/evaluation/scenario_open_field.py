"""Exact open-field B0 scenario fixture and qualification envelope."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import re
from typing import Final

from instinctlab.control import B0DirectConfig
from instinctlab.envs.taslab_direct_env import DECIMATION, TaslabDirectEnvCfg

from .baseline_environment import B0EnvironmentConfig, B0EnvironmentResult
from .baseline_run import B0_RUN_RECEIPT_SCHEMA_VERSION
from .baseline_runtime import (
    B0RuntimeEvaluationRecord,
    B0RuntimeEvaluationConfig,
    B0RuntimeEvaluationProvenance,
    B0RuntimeEvaluationResult,
    B0_SOURCE_MANIFEST,
    RECORD_KEYS,
)
from .baselines import (
    ActorAuditStatus,
    BaselineRunManifest,
    CapabilityBinding,
    ExecutionStatus,
    build_run_manifest,
    resolve_baseline,
)
from .metrics import EVALUATION_INPUT_KEYS, METRIC_KEYS, EvaluationMetricConfig
from .scenarios import (
    ScenarioId,
    ScenarioRunPlan,
    ScenarioSelection,
    bind_scenario_run_plan,
    build_scenario_catalog,
    resolve_scenario,
    select_scenario,
)


QUALIFICATION_SCHEMA_VERSION: Final = "m7.open-field-b0-qualification.v1"
FIXTURE_ID: Final = "m7-014-open-field-b0-two-env-eight-step-v1"
GENERATOR_ID: Final = "instinctlab.evaluation.scenario_open_field"
SCENARIO_ID: Final = "open_field"
BASELINE_ID: Final = "B0"
FIXED_SEED: Final = 7014
NUM_ENVS: Final = 2
POLICY_STEPS: Final = 8
PHYSICS_STEPS: Final = 160
LIFECYCLE_CALLS: Final = 160
WRENCH_APPLICATIONS: Final = 160
ENVIRONMENT_SAMPLES: Final = 320
RAYS_PER_ENVIRONMENT: Final = 21_240
RAY_SAMPLES: Final = 339_840
VELOCITY_LIMIT_M_S: Final = 2.0
GROUND_PRIM_PATH: Final = "/World/ground"
RAY_SCENE_PRIM_PATH: Final = "/World/MID360StaticScene"
RAY_MESH_PRIM_PATH: Final = "/World/MID360StaticScene/PlaneMesh"
RAYCASTER_MESH_PRIM_PATHS: Final = (RAY_SCENE_PRIM_PATH,)
CANONICAL_SCENARIO_PRIMS: Final = (
    (GROUND_PRIM_PATH, "Xform"),
    (RAY_SCENE_PRIM_PATH, "Xform"),
    (RAY_MESH_PRIM_PATH, "Mesh"),
)
GRID_MIN_M: Final = -42.0
GRID_MAX_M: Final = 42.0
GRID_SPACING_M: Final = 2.0
POINT_COUNT: Final = 1_849
TRIANGLE_COUNT: Final = 3_528
FACE_INDEX_COUNT: Final = 10_584
COMMAND_SCHEDULE: Final = (
    ((0.00, 0.00, 0.00), (0.00, 0.00, 0.00)),
    ((0.25, 0.00, 0.00), (0.25, 0.00, 0.00)),
    ((0.00, 0.25, 0.00), (0.00, -0.25, 0.00)),
    ((-0.25, 0.00, 0.00), (0.25, 0.00, 0.00)),
    ((0.00, 0.00, 0.00), (0.00, 0.00, 0.00)),
    ((0.15, 0.15, 0.00), (0.15, -0.15, 0.00)),
    ((0.00, 0.00, 0.10), (0.00, 0.00, -0.10)),
    ((0.00, 0.00, 0.00), (0.00, 0.00, 0.00)),
)

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA_PATTERN = re.compile(r"[0-9a-f]{64}")
_QUALIFICATION_KEYS = (
    "schema_version", "catalog_sha256", "fixture", "fixture_source_sha256",
    "run_plan", "manifest", "geometry", "commands", "evaluation_record",
    "environment_receipt", "coverage", "metrics", "termination_counts",
    "gates", "claim",
)
_FIXTURE_KEYS = ("fixture_id", "generator_id", "scenario_id", "baseline_id", "seed")
_HASHED_MAPPING_KEYS = ("sha256", "mapping")
_GEOMETRY_KEYS = (
    "prim_paths_and_types", "forbidden_scenario_geometry_paths", "ground_prim_path",
    "ray_scene_prim_path", "ray_mesh_prim_path", "raycaster_mesh_prim_paths",
    "point_count", "triangle_count", "face_index_count", "minimum_xyz_m",
    "maximum_xyz_m", "unique_x_m", "unique_y_m", "unique_z_m",
)
_RECEIPT_KEYS = (
    "schema_version", "ready_manifest_sha256", "executed_manifest_sha256",
    "trajectory_sha256", "step_count",
)
_COVERAGE_KEYS = (
    "num_envs", "policy_steps", "metric_samples", "physics_steps",
    "lifecycle_calls", "wrench_applications", "environment_samples",
    "rays_per_environment", "ray_samples",
)
_METRIC_KEYS = (
    "name", "reduction", "unit", "available", "value", "numerator",
    "denominator", "valid_count", "total_count",
)
_TERMINATION_COUNT_KEYS = ("terminated", "truncated", "reset")
_QUALIFICATION_GATE_KEYS = (
    "catalog_open_field_only", "run_plan_executable", "geometry_exact",
    "coverage_exact", "receipt_bound", "evaluation_record_bound",
    "no_reset_or_done", "actor_command_only",
)
_CLAIM = "exact open-field B0 migration wiring and scenario identity only; no performance or comparison claim"
_B0_CLAIM = "exact command-only wiring and aggregation evidence; no policy-performance claim"
_B0_ACTOR_SURFACE_KEYS = ("kind", "input_fields", "learned_actor_called", "normalized_action_present")
_B0_MANIFEST_KEYS = ("ready_sha256", "executed_sha256")
_B0_RECEIPT_KEYS = ("schema_version", "step_count", "trajectory_sha256")
_B0_METRIC_KEYS = _METRIC_KEYS[1:]
_B0_OBSERVABILITY_COUNT_KEYS = ("UNAVAILABLE", "PROXY", "EXACT")
_B0_TERMINATION_COUNT_KEYS = (
    "collision", "below_height", "above_height", "timeout", "terminated",
    "truncated", "done", "reason_none",
)
_B0_PROVENANCE_KEYS = ("product_commit", "fixture_id", "generator_id", "seed")
_B0_GATE_KEYS = (
    "coverage_exact", "metrics_unchanged", "command_only_actor_surface",
    "observability_normal_free", "no_reset", "all_finite",
)
_EXPECTED_COVERAGE = (NUM_ENVS, POLICY_STEPS, NUM_ENVS * POLICY_STEPS, PHYSICS_STEPS, LIFECYCLE_CALLS, WRENCH_APPLICATIONS, ENVIRONMENT_SAMPLES, RAYS_PER_ENVIRONMENT, RAY_SAMPLES)
_METRIC_SPECS = (
    ("tracking_rmse_m_s", "root_mean_square", "m/s"),
    ("null_station_drift_mean_m", "mean", "m"),
    ("null_station_speed_mean_m_s", "mean", "m/s"),
    ("anchor_error_mean_m", "mean", "m"),
    ("minimum_clearance_m", "minimum", "m"),
    ("collision_rate", "mean", "fraction"),
    ("ics_violation_rate", "mean", "fraction"),
    ("command_preservation_ratio", "mean", "ratio"),
    ("intervention_frequency", "mean", "fraction"),
    ("observability_mean", "mean", "source-defined"),
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> OrderedDict[str, object]:
    output: OrderedDict[str, object] = OrderedDict()
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        output[key] = value
    return output


def _decode_canonical(canonical_json: object) -> OrderedDict[str, object]:
    if type(canonical_json) is not str or not canonical_json or any(ord(char) > 127 for char in canonical_json):
        raise TypeError("canonical_json must be one exact nonempty ASCII string")
    decoded = json.loads(
        canonical_json, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
    )
    if not isinstance(decoded, OrderedDict) or _canonical(decoded) != canonical_json:
        raise ValueError("qualification must be one canonical JSON object")
    return decoded


def _ordered(name: str, value: object, keys: tuple[str, ...]) -> OrderedDict[str, object]:
    if not isinstance(value, OrderedDict) or tuple(value) != keys:
        raise ValueError(f"{name} keys must be exactly ordered as {keys}")
    return value


def _exact_sha(name: str, value: object) -> str:
    if type(value) is not str or _SHA_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _json_tuple(value: object) -> tuple[object, ...]:
    if type(value) is not list:
        raise TypeError("canonical JSON sequence must be an array")
    return tuple(_json_tuple(item) if type(item) is list else item for item in value)


def _exact_schedule(value: object) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    if type(value) is not tuple or len(value) != POLICY_STEPS:
        raise TypeError("command_schedule must be one exact immutable eight-row tuple")
    for row in value:
        if type(row) is not tuple or len(row) != NUM_ENVS:
            raise TypeError("each command row must be one exact two-environment tuple")
        for command in row:
            if type(command) is not tuple or len(command) != 3 or any(type(item) is not float for item in command):
                raise TypeError("each body command must be one exact float triple")
    if value != COMMAND_SCHEDULE:
        raise ValueError("command_schedule must equal the sole M7-014 schedule")
    return value


@dataclass(frozen=True, slots=True)
class OpenFieldGeometryEvidence:
    prim_paths_and_types: tuple[tuple[str, str], ...]
    forbidden_scenario_geometry_paths: tuple[str, ...]
    ground_prim_path: str
    ray_scene_prim_path: str
    ray_mesh_prim_path: str
    raycaster_mesh_prim_paths: tuple[str, ...]
    point_count: int
    triangle_count: int
    face_index_count: int
    minimum_xyz_m: tuple[float, float, float]
    maximum_xyz_m: tuple[float, float, float]
    unique_x_m: tuple[float, ...]
    unique_y_m: tuple[float, ...]
    unique_z_m: tuple[float, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("OpenFieldGeometryEvidence must not be subclassed")

    def __post_init__(self) -> None:
        if type(self.prim_paths_and_types) is not tuple or any(
            type(row) is not tuple or len(row) != 2 or any(type(item) is not str for item in row)
            for row in self.prim_paths_and_types
        ):
            raise TypeError("prim_paths_and_types must be an exact immutable string-pair tuple")
        if self.prim_paths_and_types != CANONICAL_SCENARIO_PRIMS:
            raise ValueError("geometry prim inventory must equal the exact ordered plane-only tuple")
        if type(self.forbidden_scenario_geometry_paths) is not tuple or self.forbidden_scenario_geometry_paths != ():
            raise ValueError("forbidden scenario geometry paths must be one exact empty tuple")
        if any(type(value) is not int for value in (self.point_count, self.triangle_count, self.face_index_count)):
            raise TypeError("plane mesh topology counts must be exact integers")
        for name in ("minimum_xyz_m", "maximum_xyz_m", "unique_x_m", "unique_y_m", "unique_z_m"):
            value = getattr(self, name)
            if type(value) is not tuple or any(type(item) is not float for item in value):
                raise TypeError(f"{name} must be one exact immutable float tuple")
        if self.ground_prim_path != GROUND_PRIM_PATH or self.ray_scene_prim_path != RAY_SCENE_PRIM_PATH or self.ray_mesh_prim_path != RAY_MESH_PRIM_PATH:
            raise ValueError("open-field prim identities are not exact")
        if type(self.raycaster_mesh_prim_paths) is not tuple or self.raycaster_mesh_prim_paths != RAYCASTER_MESH_PRIM_PATHS:
            raise ValueError("RayCaster mesh paths are not exact")
        if (self.point_count, self.triangle_count, self.face_index_count) != (POINT_COUNT, TRIANGLE_COUNT, FACE_INDEX_COUNT):
            raise ValueError("plane mesh topology counts are not exact")
        expected_axis = tuple(float(value) for value in range(-42, 43, 2))
        if self.unique_x_m != expected_axis or self.unique_y_m != expected_axis or self.unique_z_m != (0.0,):
            raise ValueError("plane mesh grid coordinates are not exact")
        if self.minimum_xyz_m != (GRID_MIN_M, GRID_MIN_M, 0.0) or self.maximum_xyz_m != (GRID_MAX_M, GRID_MAX_M, 0.0):
            raise ValueError("plane mesh bounds are not exact")


@dataclass(frozen=True, slots=True)
class OpenFieldScenarioFixture:
    manifest: BaselineRunManifest
    selection: ScenarioSelection
    run_plan: ScenarioRunPlan
    environment_config: B0EnvironmentConfig
    evaluation_config: B0RuntimeEvaluationConfig
    command_schedule: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("OpenFieldScenarioFixture must not be subclassed")

    def __post_init__(self) -> None:
        if type(self.manifest) is not BaselineRunManifest or type(self.selection) is not ScenarioSelection or type(self.run_plan) is not ScenarioRunPlan:
            raise TypeError("fixture manifest, selection and run_plan require exact public types")
        if type(self.environment_config) is not B0EnvironmentConfig or type(self.evaluation_config) is not B0RuntimeEvaluationConfig:
            raise TypeError("fixture configs require exact accepted types")
        manifest = BaselineRunManifest.from_json(self.manifest.canonical_json)
        selection = select_scenario(scenario_id=SCENARIO_ID, seed=FIXED_SEED)
        plan = bind_scenario_run_plan(selection=selection, manifest=manifest)
        mapping = manifest.to_mapping()
        if self.selection != selection or self.run_plan.canonical_json != plan.canonical_json or not plan.to_mapping()["executable"]:
            raise ValueError("fixture must bind the exact executable READY open-field plan")
        if mapping["baseline_id"] != BASELINE_ID or mapping["scenario_id"] != SCENARIO_ID or mapping["seed"] != FIXED_SEED or mapping["actor_audit_status"] != "PASS" or mapping["execution_status"] != "READY":
            raise ValueError("fixture manifest identity/status is not exact")
        if self.environment_config.expected_policy_steps != POLICY_STEPS or self.environment_config.environment_config.seed != FIXED_SEED or self.environment_config.environment_config.decimation != DECIMATION or self.environment_config.environment_config.scene.num_envs != NUM_ENVS:
            raise ValueError("fixture environment config is not exact")
        provenance = self.evaluation_config.provenance
        if (self.evaluation_config.expected_policy_steps, self.evaluation_config.num_envs, self.evaluation_config.rays_per_environment) != (POLICY_STEPS, NUM_ENVS, RAYS_PER_ENVIRONMENT):
            raise ValueError("fixture evaluation config coverage is not exact")
        if (provenance.product_commit, provenance.fixture_id, provenance.generator_id, provenance.seed) != (mapping["product_commit"], FIXTURE_ID, GENERATOR_ID, FIXED_SEED):
            raise ValueError("fixture provenance is not exact")
        _exact_schedule(self.command_schedule)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "run_plan", plan)
        object.__setattr__(self, "command_schedule", tuple(tuple(tuple(command) for command in row) for row in self.command_schedule))


def _validate_metric_rows(value: object) -> tuple[OrderedDict[str, object], ...]:
    if type(value) is not list or len(value) != len(_METRIC_SPECS):
        raise ValueError("qualification metrics must contain the exact ten ordered rows")
    rows: list[OrderedDict[str, object]] = []
    for raw, (name, reduction, unit) in zip(value, _METRIC_SPECS, strict=True):
        row = _ordered(f"metrics.{name}", raw, _METRIC_KEYS)
        if (row["name"], row["reduction"], row["unit"]) != (name, reduction, unit):
            raise ValueError(f"metric identity is not exact: {name}")
        if type(row["available"]) is not bool or type(row["denominator"]) is not int or type(row["valid_count"]) is not int or type(row["total_count"]) is not int:
            raise TypeError(f"metric availability/count types are not exact: {name}")
        if row["total_count"] != NUM_ENVS * POLICY_STEPS or not 0 <= row["valid_count"] <= row["total_count"]:
            raise ValueError(f"metric sample counts are not exact: {name}")
        if not row["available"]:
            if (row["value"], row["numerator"], row["denominator"], row["valid_count"]) != (None, None, 0, 0):
                raise ValueError(f"unavailable metric representation is not exact: {name}")
        else:
            if type(row["value"]) is not float or type(row["numerator"]) is not float or not all(
                __import__("math").isfinite(item) for item in (row["value"], row["numerator"])
            ) or row["valid_count"] <= 0:
                raise ValueError(f"available metric scalars are not exact finite floats: {name}")
            if reduction == "minimum":
                consistent = row["denominator"] == 1 and row["value"] == row["numerator"]
            elif reduction == "root_mean_square":
                consistent = row["denominator"] == row["valid_count"] and row["value"] == __import__("math").sqrt(row["numerator"] / row["denominator"])
            else:
                consistent = row["denominator"] == row["valid_count"] and row["value"] == row["numerator"] / row["denominator"]
            if not consistent:
                raise ValueError(f"metric reduction/count semantics are inconsistent: {name}")
        rows.append(row)
    return tuple(rows)


def _validate_b0_record(
    mapping: object,
    *,
    ready_manifest: BaselineRunManifest,
    executed_manifest: BaselineRunManifest,
    receipt: OrderedDict[str, object],
    coverage: OrderedDict[str, object],
    metrics: tuple[OrderedDict[str, object], ...],
) -> B0RuntimeEvaluationRecord:
    root = _ordered("evaluation_record.mapping", mapping, RECORD_KEYS)
    record = B0RuntimeEvaluationRecord(_canonical(root))
    ready = ready_manifest.to_mapping()
    static = {
        "schema_version": "m7.b0-runtime-evaluation.v1",
        "platform": "TASLAB_UAV",
        "sensor": "Livox MID360",
        "baseline_id": BASELINE_ID,
        "claim": _B0_CLAIM,
    }
    if any(root[name] != expected for name, expected in static.items()):
        raise ValueError("B0 record static identity/claim is not exact")
    actor = _ordered("evaluation_record.actor_surface", root["actor_surface"], _B0_ACTOR_SURFACE_KEYS)
    if tuple(actor.values()) != ("command-only", ["commanded_body_velocity_m_s"], False, False):
        raise ValueError("B0 record actor surface is not exact")
    manifest = _ordered("evaluation_record.manifest", root["manifest"], _B0_MANIFEST_KEYS)
    if tuple(manifest.values()) != (ready_manifest.manifest_sha256, executed_manifest.manifest_sha256):
        raise ValueError("B0 record manifest hashes are not bound")
    b0_receipt = _ordered("evaluation_record.receipt", root["receipt"], _B0_RECEIPT_KEYS)
    if tuple(b0_receipt.values()) != (B0_RUN_RECEIPT_SCHEMA_VERSION, PHYSICS_STEPS, receipt["trajectory_sha256"]):
        raise ValueError("B0 record receipt is not bound")
    if _ordered("evaluation_record.coverage", root["coverage"], _COVERAGE_KEYS) != coverage:
        raise ValueError("B0 record coverage is not bound")
    if root["evaluation_input_order"] != list(EVALUATION_INPUT_KEYS):
        raise ValueError("B0 evaluation input order is not exact")
    expected_source = OrderedDict(
        (name, OrderedDict(metadata)) for name, metadata in B0_SOURCE_MANIFEST
    )
    source = root["source_manifest"]
    if not isinstance(source, OrderedDict) or source != expected_source:
        raise ValueError("B0 source manifest is not exact")
    metric_config = _ordered(
        "evaluation_record.metric_config", root["metric_config"],
        ("schema_version", "null_command_threshold_m_s", "full_compliance_beta"),
    )
    if metric_config != OrderedDict(asdict(EvaluationMetricConfig()).items()):
        raise ValueError("B0 metric config is not exact")
    source_counts = _ordered("evaluation_record.observability_source_counts", root["observability_source_counts"], _B0_OBSERVABILITY_COUNT_KEYS)
    if tuple(source_counts.values()) != (0, 0, NUM_ENVS * POLICY_STEPS) or any(type(value) is not int for value in source_counts.values()):
        raise ValueError("B0 observability source counts are not exact")
    b0_metrics = root["metrics"]
    if not isinstance(b0_metrics, OrderedDict) or tuple(b0_metrics) != METRIC_KEYS:
        raise ValueError("B0 metric order is not exact")
    for row in metrics:
        nested = _ordered(f"evaluation_record.metrics.{row['name']}", b0_metrics[row["name"]], _B0_METRIC_KEYS)
        if nested != OrderedDict((key, row[key]) for key in _B0_METRIC_KEYS):
            raise ValueError(f"B0 and qualification metrics are not identical: {row['name']}")
    termination = _ordered("evaluation_record.termination_counts", root["termination_counts"], _B0_TERMINATION_COUNT_KEYS)
    if tuple(termination.values()) != (0, 0, 0, 0, 0, 0, 0, NUM_ENVS * POLICY_STEPS) or any(type(value) is not int for value in termination.values()):
        raise ValueError("B0 termination counts are not exact")
    provenance = _ordered("evaluation_record.provenance", root["provenance"], _B0_PROVENANCE_KEYS)
    if tuple(provenance.values()) != (ready["product_commit"], FIXTURE_ID, GENERATOR_ID, FIXED_SEED):
        raise ValueError("B0 provenance is not exact")
    gates = _ordered("evaluation_record.gates", root["gates"], _B0_GATE_KEYS)
    if tuple(gates.values()) != (True,) * len(_B0_GATE_KEYS) or any(type(value) is not bool for value in gates.values()):
        raise ValueError("B0 record gates are not exact")
    return record


def _validate_qualification_mapping(mapping: OrderedDict[str, object]) -> None:
    root = _ordered("qualification", mapping, _QUALIFICATION_KEYS)
    if root["schema_version"] != QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("qualification schema is not exact")
    catalog = build_scenario_catalog()
    if root["catalog_sha256"] != catalog.catalog_sha256:
        raise ValueError("qualification catalog hash is not current")
    fixture = _ordered("fixture", root["fixture"], _FIXTURE_KEYS)
    if tuple(fixture.values()) != (FIXTURE_ID, GENERATOR_ID, SCENARIO_ID, BASELINE_ID, FIXED_SEED):
        raise ValueError("qualification fixture identity is not exact")
    source_sha = resolve_scenario(SCENARIO_ID).evidence.evidence_sha256
    if root["fixture_source_sha256"] != source_sha or _exact_sha("fixture_source_sha256", root["fixture_source_sha256"]) != source_sha:
        raise ValueError("qualification fixture source hash is not exact")
    manifest_section = _ordered("manifest", root["manifest"], _HASHED_MAPPING_KEYS)
    manifest = BaselineRunManifest.from_json(_canonical(manifest_section["mapping"]))
    if manifest_section["sha256"] != manifest.manifest_sha256:
        raise ValueError("qualification manifest hash is forged")
    manifest_mapping = manifest.to_mapping()
    if (manifest_mapping["baseline_id"], manifest_mapping["scenario_id"], manifest_mapping["seed"], manifest_mapping["actor_audit_status"], manifest_mapping["execution_status"]) != (BASELINE_ID, SCENARIO_ID, FIXED_SEED, "PASS", "READY"):
        raise ValueError("qualification READY manifest identity/status is not exact")
    bindings = tuple(
        CapabilityBinding(item["name"], item["provider"], item["evidence_sha256"])
        for item in manifest_mapping["capability_bindings"]
    )
    executed = build_run_manifest(
        product_commit=manifest_mapping["product_commit"], baseline_id=BASELINE_ID,
        config_switch=manifest_mapping["config_switch"], capability_bindings=bindings,
        scenario_id=SCENARIO_ID, seed=FIXED_SEED,
        actor_audit_status=ActorAuditStatus.PASS,
        execution_status=ExecutionStatus.EXECUTED, skip_reason=None,
    )
    plan_section = _ordered("run_plan", root["run_plan"], _HASHED_MAPPING_KEYS)
    plan = ScenarioRunPlan.from_json(_canonical(plan_section["mapping"]), manifest)
    if plan_section["sha256"] != plan.run_plan_sha256 or not plan.to_mapping()["executable"] or plan.to_mapping()["catalog_sha256"] != catalog.catalog_sha256:
        raise ValueError("qualification executable run plan/hash is not exact")
    geometry_mapping = _ordered("geometry", root["geometry"], _GEOMETRY_KEYS)
    geometry = OpenFieldGeometryEvidence(
        _json_tuple(geometry_mapping["prim_paths_and_types"]),
        _json_tuple(geometry_mapping["forbidden_scenario_geometry_paths"]),
        geometry_mapping["ground_prim_path"], geometry_mapping["ray_scene_prim_path"],
        geometry_mapping["ray_mesh_prim_path"], _json_tuple(geometry_mapping["raycaster_mesh_prim_paths"]),
        geometry_mapping["point_count"], geometry_mapping["triangle_count"], geometry_mapping["face_index_count"],
        _json_tuple(geometry_mapping["minimum_xyz_m"]), _json_tuple(geometry_mapping["maximum_xyz_m"]),
        _json_tuple(geometry_mapping["unique_x_m"]), _json_tuple(geometry_mapping["unique_y_m"]),
        _json_tuple(geometry_mapping["unique_z_m"]),
    )
    _exact_schedule(_json_tuple(root["commands"]))
    receipt = _ordered("environment_receipt", root["environment_receipt"], _RECEIPT_KEYS)
    if receipt["schema_version"] != B0_RUN_RECEIPT_SCHEMA_VERSION or receipt["ready_manifest_sha256"] != manifest.manifest_sha256 or receipt["executed_manifest_sha256"] != executed.manifest_sha256 or type(receipt["step_count"]) is not int or receipt["step_count"] != PHYSICS_STEPS:
        raise ValueError("qualification receipt identity/count is not exact")
    _exact_sha("receipt.executed_manifest_sha256", receipt["executed_manifest_sha256"])
    _exact_sha("receipt.trajectory_sha256", receipt["trajectory_sha256"])
    coverage = _ordered("coverage", root["coverage"], _COVERAGE_KEYS)
    if tuple(coverage.values()) != _EXPECTED_COVERAGE or any(type(value) is not int for value in coverage.values()):
        raise ValueError("qualification coverage is not exact")
    metrics = _validate_metric_rows(root["metrics"])
    record_section = _ordered("evaluation_record", root["evaluation_record"], _HASHED_MAPPING_KEYS)
    record = _validate_b0_record(
        record_section["mapping"], ready_manifest=manifest, executed_manifest=executed,
        receipt=receipt, coverage=coverage, metrics=metrics,
    )
    if record_section["sha256"] != hashlib.sha256(record.canonical_json.encode("ascii")).hexdigest():
        raise ValueError("qualification B0 record hash is forged")
    terminations = _ordered("termination_counts", root["termination_counts"], _TERMINATION_COUNT_KEYS)
    if tuple(terminations.values()) != (0, 0, 0) or any(type(value) is not int for value in terminations.values()):
        raise ValueError("qualification termination/reset counts are not exact")
    gates = _ordered("gates", root["gates"], _QUALIFICATION_GATE_KEYS)
    if tuple(gates.values()) != (True,) * len(_QUALIFICATION_GATE_KEYS) or any(type(value) is not bool for value in gates.values()):
        raise ValueError("qualification gates are not exact")
    if root["claim"] != _CLAIM:
        raise ValueError("qualification no-performance claim is not exact")


@dataclass(frozen=True, slots=True)
class OpenFieldQualificationRecord:
    canonical_json: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("OpenFieldQualificationRecord must not be subclassed")

    def __post_init__(self) -> None:
        decoded = _decode_canonical(self.canonical_json)
        _validate_qualification_mapping(decoded)

    def to_mapping(self) -> dict[str, object]:
        return json.loads(self.canonical_json)

    @property
    def qualification_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("ascii")).hexdigest()


def create_open_field_fixture(*, product_commit: str) -> OpenFieldScenarioFixture:
    """Construct and reparse the sole exact READY open-field B0 fixture."""
    if type(product_commit) is not str or _COMMIT_PATTERN.fullmatch(product_commit) is None:
        raise ValueError("product_commit must be one exact lowercase 40-hex commit")
    baseline = resolve_baseline(BASELINE_ID)
    bindings = tuple(
        CapabilityBinding(item.name, item.accepted_provider, item.accepted_evidence_sha256)
        for item in baseline.capability_requirements
    )
    manifest = build_run_manifest(
        product_commit=product_commit,
        baseline_id=BASELINE_ID,
        config_switch=baseline.config_switch,
        capability_bindings=bindings,
        scenario_id=SCENARIO_ID,
        seed=FIXED_SEED,
        actor_audit_status=ActorAuditStatus.PASS,
        execution_status=ExecutionStatus.READY,
        skip_reason=None,
    )
    selection = select_scenario(scenario_id=ScenarioId.OPEN_FIELD, seed=FIXED_SEED)
    plan = bind_scenario_run_plan(selection=selection, manifest=manifest)
    cfg = TaslabDirectEnvCfg()
    cfg.seed = FIXED_SEED
    environment = B0EnvironmentConfig(POLICY_STEPS, B0DirectConfig(VELOCITY_LIMIT_M_S), cfg)
    evaluation = B0RuntimeEvaluationConfig(
        POLICY_STEPS, NUM_ENVS, RAYS_PER_ENVIRONMENT, EvaluationMetricConfig(),
        B0RuntimeEvaluationProvenance(product_commit, FIXTURE_ID, GENERATOR_ID, FIXED_SEED),
    )
    return OpenFieldScenarioFixture(manifest, selection, plan, environment, evaluation, COMMAND_SCHEDULE)


def finalize_open_field_qualification(
    *,
    fixture: OpenFieldScenarioFixture,
    geometry: OpenFieldGeometryEvidence,
    environment_result: B0EnvironmentResult,
    evaluation_result: B0RuntimeEvaluationResult,
) -> OpenFieldQualificationRecord:
    """Bind exact finalized accepted outputs; no caller supplies metric values."""
    if type(fixture) is not OpenFieldScenarioFixture or type(geometry) is not OpenFieldGeometryEvidence:
        raise TypeError("fixture and geometry require exact open-field types")
    if type(environment_result) is not B0EnvironmentResult or type(evaluation_result) is not B0RuntimeEvaluationResult:
        raise TypeError("qualification requires exact accepted finalized B0 outputs")
    plan = fixture.run_plan.to_mapping()
    catalog = build_scenario_catalog()
    open_field = resolve_scenario(SCENARIO_ID)
    if not plan["executable"] or plan["catalog_sha256"] != catalog.catalog_sha256:
        raise ValueError("qualification requires the current exact executable catalog plan")
    source_sha = open_field.evidence.evidence_sha256
    if type(source_sha) is not str or _SHA_PATTERN.fullmatch(source_sha) is None:
        raise ValueError("open-field committed fixture source evidence is absent")
    receipt = environment_result.receipt
    if evaluation_result.ready_manifest_sha256 != fixture.manifest.manifest_sha256 or receipt.ready_manifest.manifest_sha256 != fixture.manifest.manifest_sha256:
        raise ValueError("READY manifest identity is not bound")
    if evaluation_result.executed_manifest_sha256 != receipt.executed_manifest.manifest_sha256 or evaluation_result.trajectory_sha256 != receipt.trajectory_sha256:
        raise ValueError("evaluation result and environment receipt are not identical")
    coverage = evaluation_result.contract
    if tuple(asdict(coverage).values()) != _EXPECTED_COVERAGE:
        raise ValueError("open-field coverage is not exact")
    if environment_result.coverage.policy_steps != POLICY_STEPS or environment_result.coverage.physics_steps != PHYSICS_STEPS or environment_result.coverage.lifecycle_calls != LIFECYCLE_CALLS or environment_result.coverage.wrench_applications != WRENCH_APPLICATIONS or environment_result.coverage.environment_samples != ENVIRONMENT_SAMPLES:
        raise ValueError("environment coverage is not exact")
    metrics = [OrderedDict((field.name, getattr(metric, field.name)) for field in fields(metric)) for metric in evaluation_result.aggregate_metrics.metrics]
    termination = dict(evaluation_result.evaluation_inputs)
    terminated = int(termination["terminated"].sum().item())
    truncated = int(termination["truncated"].sum().item())
    gates = OrderedDict((
        ("catalog_open_field_only", sum(row["executable"] for row in catalog.to_mapping()["rows"]) == 1),
        ("run_plan_executable", plan["executable"] is True),
        ("geometry_exact", True),
        ("coverage_exact", True),
        ("receipt_bound", True),
        ("evaluation_record_bound", hashlib.sha256(evaluation_result.record.canonical_json.encode("ascii")).hexdigest() == hashlib.sha256(_canonical(evaluation_result.record.to_mapping()).encode("ascii")).hexdigest()),
        ("no_reset_or_done", terminated == 0 and truncated == 0),
        ("actor_command_only", plan["forwarded_actor_fields"] == ["commanded_body_velocity_m_s"]),
    ))
    if not all(gates.values()):
        raise ValueError("open-field qualification gate failed")
    mapping = OrderedDict((
        ("schema_version", QUALIFICATION_SCHEMA_VERSION),
        ("catalog_sha256", catalog.catalog_sha256),
        ("fixture", OrderedDict((("fixture_id", FIXTURE_ID), ("generator_id", GENERATOR_ID), ("scenario_id", SCENARIO_ID), ("baseline_id", BASELINE_ID), ("seed", FIXED_SEED)))),
        ("fixture_source_sha256", source_sha),
        ("run_plan", OrderedDict((("sha256", fixture.run_plan.run_plan_sha256), ("mapping", plan)))),
        ("manifest", OrderedDict((("sha256", fixture.manifest.manifest_sha256), ("mapping", fixture.manifest.to_mapping())))),
        ("geometry", OrderedDict((field.name, getattr(geometry, field.name)) for field in fields(geometry))),
        ("commands", fixture.command_schedule),
        ("evaluation_record", OrderedDict((("sha256", hashlib.sha256(evaluation_result.record.canonical_json.encode("ascii")).hexdigest()), ("mapping", evaluation_result.record.to_mapping())))),
        ("environment_receipt", OrderedDict((("schema_version", receipt.schema_version), ("ready_manifest_sha256", receipt.ready_manifest.manifest_sha256), ("executed_manifest_sha256", receipt.executed_manifest.manifest_sha256), ("trajectory_sha256", receipt.trajectory_sha256), ("step_count", receipt.step_count)))),
        ("coverage", OrderedDict(asdict(coverage).items())),
        ("metrics", metrics),
        ("termination_counts", OrderedDict((("terminated", terminated), ("truncated", truncated), ("reset", 0)))),
        ("gates", gates),
        ("claim", _CLAIM),
    ))
    return OpenFieldQualificationRecord(_canonical(mapping))


__all__: tuple[str, ...] = ()
