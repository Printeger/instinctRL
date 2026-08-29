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
from .baseline_runtime import (
    B0RuntimeEvaluationConfig,
    B0RuntimeEvaluationProvenance,
    B0RuntimeEvaluationResult,
)
from .baselines import (
    ActorAuditStatus,
    BaselineRunManifest,
    CapabilityBinding,
    ExecutionStatus,
    build_run_manifest,
    resolve_baseline,
)
from .metrics import EvaluationMetricConfig
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


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


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
        if len({path for path, _ in self.prim_paths_and_types}) != len(self.prim_paths_and_types):
            raise ValueError("geometry prim paths must not duplicate")
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
        prims = dict(self.prim_paths_and_types)
        if prims.get(GROUND_PRIM_PATH) != "Xform" or prims.get(RAY_SCENE_PRIM_PATH) != "Xform" or prims.get(RAY_MESH_PRIM_PATH) != "Mesh":
            raise ValueError("required live plane-only prim types are absent")


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


@dataclass(frozen=True, slots=True)
class OpenFieldQualificationRecord:
    canonical_json: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("OpenFieldQualificationRecord must not be subclassed")

    def __post_init__(self) -> None:
        if type(self.canonical_json) is not str or not self.canonical_json:
            raise TypeError("canonical_json must be one exact nonempty string")
        decoded = json.loads(self.canonical_json, object_pairs_hook=OrderedDict)
        if tuple(decoded) != _QUALIFICATION_KEYS or decoded["schema_version"] != QUALIFICATION_SCHEMA_VERSION:
            raise ValueError("qualification keys/schema are not exact")
        if _canonical(decoded) != self.canonical_json:
            raise ValueError("qualification must use canonical JSON bytes")

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
    expected_coverage = (NUM_ENVS, POLICY_STEPS, NUM_ENVS * POLICY_STEPS, PHYSICS_STEPS, LIFECYCLE_CALLS, WRENCH_APPLICATIONS, ENVIRONMENT_SAMPLES, RAYS_PER_ENVIRONMENT, RAY_SAMPLES)
    if tuple(asdict(coverage).values()) != expected_coverage:
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
        ("claim", "exact open-field B0 migration wiring and scenario identity only; no performance or comparison claim"),
    ))
    return OpenFieldQualificationRecord(_canonical(mapping))


__all__: tuple[str, ...] = ()
