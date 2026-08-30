"""Pure/static tests for the exact M7-014 open-field B0 fixture."""

from __future__ import annotations

import ast
from collections import OrderedDict
import copy
from dataclasses import dataclass, FrozenInstanceError, fields
import hashlib
import inspect
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from instinctlab.control.baseline_b0 import B0DirectConfig


class FakeTaslabDirectEnvCfg:
    def __init__(self) -> None:
        self.seed = 0
        self.decimation = 20
        self.scene = SimpleNamespace(num_envs=2)
        self.sim = SimpleNamespace(device="cuda:0")
        self.action_noise_model = None


@dataclass(frozen=True, slots=True)
class B0EnvironmentConfig:
    expected_policy_steps: int
    direct_config: B0DirectConfig
    environment_config: FakeTaslabDirectEnvCfg


@dataclass(frozen=True, slots=True)
class B0EnvironmentResult:
    receipt: object
    coverage: object


fake_parent = ModuleType("instinctlab.envs")
fake_parent.__path__ = []
fake_env = ModuleType("instinctlab.envs.taslab_direct_env")
fake_env.DECIMATION = 20
fake_env.TaslabDirectEnvCfg = FakeTaslabDirectEnvCfg
fake_environment = ModuleType("instinctlab.evaluation.baseline_environment")
fake_environment.B0EnvironmentConfig = B0EnvironmentConfig
fake_environment.B0EnvironmentResult = B0EnvironmentResult
previous = {name: sys.modules.get(name) for name in (fake_parent.__name__, fake_env.__name__, fake_environment.__name__)}
sys.modules[fake_parent.__name__] = fake_parent
sys.modules[fake_env.__name__] = fake_env
sys.modules[fake_environment.__name__] = fake_environment

from instinctlab.evaluation.scenario_open_field import (
    BASELINE_ID,
    CANONICAL_SCENARIO_PRIMS,
    COMMAND_SCHEDULE,
    ENVIRONMENT_SAMPLES,
    FACE_INDEX_COUNT,
    FIXED_SEED,
    FIXTURE_ID,
    GENERATOR_ID,
    GROUND_PRIM_PATH,
    LIFECYCLE_CALLS,
    NUM_ENVS,
    OpenFieldGeometryEvidence,
    OpenFieldQualificationRecord,
    OpenFieldScenarioFixture,
    PHYSICS_STEPS,
    POINT_COUNT,
    POLICY_STEPS,
    QUALIFICATION_SCHEMA_VERSION,
    RAYCASTER_MESH_PRIM_PATHS,
    RAY_MESH_PRIM_PATH,
    RAY_SAMPLES,
    RAY_SCENE_PRIM_PATH,
    RAYS_PER_ENVIRONMENT,
    SCENARIO_ID,
    TRIANGLE_COUNT,
    WRENCH_APPLICATIONS,
    create_open_field_fixture,
    finalize_open_field_qualification,
)
from instinctlab.evaluation.scenarios import OPEN_FIELD_EVIDENCE_SHA256
from instinctlab.evaluation.baseline_run import B0_RUN_RECEIPT_SCHEMA_VERSION
from instinctlab.evaluation.baseline_runtime import B0_SOURCE_MANIFEST
from instinctlab.evaluation.baselines import ActorAuditStatus, CapabilityBinding, ExecutionStatus, build_run_manifest
from instinctlab.evaluation.metrics import EVALUATION_INPUT_KEYS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "instinctlab/evaluation/scenario_open_field.py"
PRODUCT_COMMIT = "deeb7ad4221df1f0e74ec0ac24affefbc27b817d"


def _geometry() -> OpenFieldGeometryEvidence:
    axis = tuple(float(value) for value in range(-42, 43, 2))
    return OpenFieldGeometryEvidence(
        CANONICAL_SCENARIO_PRIMS, (),
        GROUND_PRIM_PATH, RAY_SCENE_PRIM_PATH, RAY_MESH_PRIM_PATH,
        RAYCASTER_MESH_PRIM_PATHS, POINT_COUNT, TRIANGLE_COUNT,
        FACE_INDEX_COUNT, (-42.0, -42.0, 0.0), (42.0, 42.0, 0.0),
        axis, axis, (0.0,),
    )


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _metric_rows() -> list[OrderedDict[str, object]]:
    specs = (
        ("tracking_rmse_m_s", "root_mean_square", "m/s", True, 1.0, 16.0, 16, 16),
        ("null_station_drift_mean_m", "mean", "m", True, 1.0, 8.0, 8, 8),
        ("null_station_speed_mean_m_s", "mean", "m/s", True, 0.5, 4.0, 8, 8),
        ("anchor_error_mean_m", "mean", "m", False, None, None, 0, 0),
        ("minimum_clearance_m", "minimum", "m", True, 2.0, 2.0, 1, 16),
        ("collision_rate", "mean", "fraction", True, 0.0, 0.0, 16, 16),
        ("ics_violation_rate", "mean", "fraction", True, 0.0, 0.0, 16, 16),
        ("command_preservation_ratio", "mean", "ratio", True, 1.0, 8.0, 8, 8),
        ("intervention_frequency", "mean", "fraction", True, 0.0, 0.0, 16, 16),
        ("observability_mean", "mean", "source-defined", True, 0.0, 0.0, 16, 16),
    )
    return [OrderedDict((("name", name), ("reduction", reduction), ("unit", unit), ("available", available), ("value", value), ("numerator", numerator), ("denominator", denominator), ("valid_count", valid), ("total_count", 16))) for name, reduction, unit, available, value, numerator, denominator, valid in specs]


def _qualification_payload() -> str:
    fixture = create_open_field_fixture(product_commit=PRODUCT_COMMIT)
    ready = fixture.manifest
    ready_mapping = ready.to_mapping()
    bindings = tuple(CapabilityBinding(item["name"], item["provider"], item["evidence_sha256"]) for item in ready_mapping["capability_bindings"])
    executed = build_run_manifest(
        product_commit=PRODUCT_COMMIT, baseline_id="B0", config_switch="direct_velocity",
        capability_bindings=bindings, scenario_id="open_field", seed=7014,
        actor_audit_status=ActorAuditStatus.PASS, execution_status=ExecutionStatus.EXECUTED,
        skip_reason=None,
    )
    trajectory = "1" * 64
    coverage = OrderedDict((("num_envs", 2), ("policy_steps", 8), ("metric_samples", 16), ("physics_steps", 160), ("lifecycle_calls", 160), ("wrench_applications", 160), ("environment_samples", 320), ("rays_per_environment", 21240), ("ray_samples", 339840)))
    metrics = _metric_rows()
    b0_metrics = OrderedDict((row["name"], OrderedDict((key, row[key]) for key in ("reduction", "unit", "available", "value", "numerator", "denominator", "valid_count", "total_count"))) for row in metrics)
    record = OrderedDict((
        ("schema_version", "m7.b0-runtime-evaluation.v1"), ("platform", "TASLAB_UAV"),
        ("sensor", "Livox MID360"), ("baseline_id", "B0"),
        ("claim", "exact command-only wiring and aggregation evidence; no policy-performance claim"),
        ("actor_surface", OrderedDict((("kind", "command-only"), ("input_fields", ["commanded_body_velocity_m_s"]), ("learned_actor_called", False), ("normalized_action_present", False)))),
        ("manifest", OrderedDict((("ready_sha256", ready.manifest_sha256), ("executed_sha256", executed.manifest_sha256)))),
        ("receipt", OrderedDict((("schema_version", B0_RUN_RECEIPT_SCHEMA_VERSION), ("step_count", 160), ("trajectory_sha256", trajectory)))),
        ("coverage", coverage), ("evaluation_input_order", list(EVALUATION_INPUT_KEYS)),
        ("source_manifest", OrderedDict((name, OrderedDict(metadata)) for name, metadata in B0_SOURCE_MANIFEST)),
        ("metric_config", OrderedDict((("schema_version", "m7.evaluation.v1"), ("null_command_threshold_m_s", 0.001), ("full_compliance_beta", 0.999)))),
        ("observability_source_counts", OrderedDict((("UNAVAILABLE", 0), ("PROXY", 0), ("EXACT", 16)))),
        ("metrics", b0_metrics),
        ("termination_counts", OrderedDict((("collision", 0), ("below_height", 0), ("above_height", 0), ("timeout", 0), ("terminated", 0), ("truncated", 0), ("done", 0), ("reason_none", 16)))),
        ("provenance", OrderedDict((("product_commit", PRODUCT_COMMIT), ("fixture_id", FIXTURE_ID), ("generator_id", GENERATOR_ID), ("seed", 7014)))),
        ("gates", OrderedDict((("coverage_exact", True), ("metrics_unchanged", True), ("command_only_actor_surface", True), ("observability_normal_free", True), ("no_reset", True), ("all_finite", True)))),
    ))
    geometry = _geometry()
    mapping = OrderedDict((
        ("schema_version", QUALIFICATION_SCHEMA_VERSION),
        ("catalog_sha256", fixture.run_plan.to_mapping()["catalog_sha256"]),
        ("fixture", OrderedDict((("fixture_id", FIXTURE_ID), ("generator_id", GENERATOR_ID), ("scenario_id", "open_field"), ("baseline_id", "B0"), ("seed", 7014)))),
        ("fixture_source_sha256", OPEN_FIELD_EVIDENCE_SHA256),
        ("run_plan", OrderedDict((("sha256", fixture.run_plan.run_plan_sha256), ("mapping", fixture.run_plan.to_mapping())))),
        ("manifest", OrderedDict((("sha256", ready.manifest_sha256), ("mapping", ready_mapping)))),
        ("geometry", OrderedDict((field.name, getattr(geometry, field.name)) for field in fields(geometry))),
        ("commands", COMMAND_SCHEDULE),
        ("evaluation_record", OrderedDict((("sha256", hashlib.sha256(_canonical(record).encode("ascii")).hexdigest()), ("mapping", record)))),
        ("environment_receipt", OrderedDict((("schema_version", B0_RUN_RECEIPT_SCHEMA_VERSION), ("ready_manifest_sha256", ready.manifest_sha256), ("executed_manifest_sha256", executed.manifest_sha256), ("trajectory_sha256", trajectory), ("step_count", 160)))),
        ("coverage", coverage), ("metrics", metrics),
        ("termination_counts", OrderedDict((("terminated", 0), ("truncated", 0), ("reset", 0)))),
        ("gates", OrderedDict((("catalog_ready_set_exact", True), ("run_plan_executable", True), ("geometry_exact", True), ("coverage_exact", True), ("receipt_bound", True), ("evaluation_record_bound", True), ("no_reset_or_done", True), ("actor_command_only", True)))),
        ("claim", "exact open-field B0 migration wiring and scenario identity only; no performance or comparison claim"),
    ))
    return _canonical(mapping)


def test_exact_constants_schedule_and_committed_source_identity() -> None:
    assert (SCENARIO_ID, BASELINE_ID, FIXED_SEED) == ("open_field", "B0", 7014)
    assert (NUM_ENVS, POLICY_STEPS, PHYSICS_STEPS, LIFECYCLE_CALLS, WRENCH_APPLICATIONS, ENVIRONMENT_SAMPLES) == (2, 8, 160, 160, 160, 320)
    assert (RAYS_PER_ENVIRONMENT, RAY_SAMPLES) == (21240, 339840)
    assert COMMAND_SCHEDULE == (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.25, 0.0, 0.0), (0.25, 0.0, 0.0)),
        ((0.0, 0.25, 0.0), (0.0, -0.25, 0.0)),
        ((-0.25, 0.0, 0.0), (0.25, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.15, 0.15, 0.0), (0.15, -0.15, 0.0)),
        ((0.0, 0.0, 0.10), (0.0, 0.0, -0.10)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == OPEN_FIELD_EVIDENCE_SHA256


def test_fixture_factory_owns_exact_manifest_plan_configs_and_schedule() -> None:
    fixture = create_open_field_fixture(product_commit=PRODUCT_COMMIT)
    assert type(fixture) is OpenFieldScenarioFixture
    assert fixture.manifest.to_mapping()["product_commit"] == PRODUCT_COMMIT
    assert fixture.manifest.to_mapping()["scenario_id"] == SCENARIO_ID
    assert fixture.manifest.to_mapping()["seed"] == FIXED_SEED
    assert fixture.run_plan.to_mapping()["executable"] is True
    assert fixture.run_plan.to_mapping()["scenario_evidence_sha256"] == OPEN_FIELD_EVIDENCE_SHA256
    assert fixture.environment_config.expected_policy_steps == POLICY_STEPS
    assert fixture.environment_config.environment_config.scene.num_envs == NUM_ENVS
    assert fixture.environment_config.environment_config.decimation == 20
    assert fixture.evaluation_config.provenance.fixture_id == FIXTURE_ID
    assert fixture.evaluation_config.provenance.generator_id == GENERATOR_ID
    assert fixture.command_schedule == COMMAND_SCHEDULE and fixture.command_schedule is not COMMAND_SCHEDULE


@pytest.mark.parametrize("value", ("0" * 39, "G" * 40, "0" * 41, "  " + "0" * 40, 1, None, True))
def test_fixture_factory_rejects_commit_alias_or_coercion(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        create_open_field_fixture(product_commit=value)


def test_geometry_exact_plane_grid_topology_and_frozen_type() -> None:
    geometry = _geometry()
    assert tuple(field.name for field in fields(geometry)) == (
        "prim_paths_and_types", "forbidden_scenario_geometry_paths",
        "ground_prim_path", "ray_scene_prim_path",
        "ray_mesh_prim_path", "raycaster_mesh_prim_paths", "point_count",
        "triangle_count", "face_index_count", "minimum_xyz_m", "maximum_xyz_m",
        "unique_x_m", "unique_y_m", "unique_z_m",
    )
    with pytest.raises(FrozenInstanceError):
        geometry.point_count = 0
    with pytest.raises(TypeError):
        type("Forbidden", (OpenFieldGeometryEvidence,), {})


@pytest.mark.parametrize("field,value", (
    ("prim_paths_and_types", CANONICAL_SCENARIO_PRIMS + (("/World/Wall", "Mesh"),)),
    ("prim_paths_and_types", CANONICAL_SCENARIO_PRIMS[::-1]),
    ("prim_paths_and_types", ((GROUND_PRIM_PATH, "Mesh"),) + CANONICAL_SCENARIO_PRIMS[1:]),
    ("forbidden_scenario_geometry_paths", ("/World/Wall",)),
    ("point_count", POINT_COUNT - 1),
    ("triangle_count", TRIANGLE_COUNT + 1),
    ("face_index_count", FACE_INDEX_COUNT - 3),
    ("raycaster_mesh_prim_paths", (RAY_MESH_PRIM_PATH,)),
    ("unique_z_m", (0.0, 1.0)),
    ("minimum_xyz_m", (-41.0, -42.0, 0.0)),
))
def test_geometry_rejects_count_path_grid_and_bound_mutations(field, value) -> None:
    args = {item.name: getattr(_geometry(), item.name) for item in fields(OpenFieldGeometryEvidence)}
    args[field] = value
    with pytest.raises((TypeError, ValueError)):
        OpenFieldGeometryEvidence(**args)


def test_public_records_are_exact_frozen_non_subclassable_and_signatures_have_no_defaults() -> None:
    fixture = create_open_field_fixture(product_commit=PRODUCT_COMMIT)
    with pytest.raises(FrozenInstanceError):
        fixture.command_schedule = ()
    with pytest.raises(TypeError):
        type("Forbidden", (OpenFieldScenarioFixture,), {})
    with pytest.raises(TypeError):
        type("Forbidden", (OpenFieldQualificationRecord,), {})
    assert tuple(inspect.signature(create_open_field_fixture).parameters) == ("product_commit",)
    assert tuple(inspect.signature(finalize_open_field_qualification).parameters) == ("fixture", "geometry", "environment_result", "evaluation_result")
    assert all(value.default is inspect.Parameter.empty for value in inspect.signature(finalize_open_field_qualification).parameters.values())


def test_qualification_record_rejects_noncanonical_incomplete_or_wrong_schema() -> None:
    for payload in ("{}", '{ "schema_version":"m7.open-field-b0-qualification.v4"}', '{"schema_version":"bad"}'):
        with pytest.raises((TypeError, ValueError)):
            OpenFieldQualificationRecord(payload)
    assert QUALIFICATION_SCHEMA_VERSION == "m7.open-field-b0-qualification.v4"


def test_semantic_qualification_round_trip_is_exact() -> None:
    payload = _qualification_payload()
    record = OpenFieldQualificationRecord(payload)
    assert record.canonical_json == payload
    assert record.to_mapping()["geometry"]["forbidden_scenario_geometry_paths"] == []
    assert record.to_mapping()["claim"].endswith("no performance or comparison claim")
    assert record.qualification_sha256 == hashlib.sha256(payload.encode("ascii")).hexdigest()


def test_all_nested_qualification_mutations_fail_atomically_without_input_change() -> None:
    original = json.loads(_qualification_payload(), object_pairs_hook=OrderedDict)
    snapshot = copy.deepcopy(original)
    accepted: list[OpenFieldQualificationRecord] = []

    def set_path(mapping, path, value):
        target = mapping
        for item in path[:-1]:
            target = target[item]
        target[path[-1]] = value

    cases = (
        ("root_schema", ("schema_version",), "bad"),
        ("catalog", ("catalog_sha256",), "0" * 64),
        ("fixture", ("fixture", "seed"), 7015),
        ("source", ("fixture_source_sha256",), "0" * 64),
        ("plan_hash", ("run_plan", "sha256"), "0" * 64),
        ("plan_mapping", ("run_plan", "mapping", "seed"), 7015),
        ("manifest_hash", ("manifest", "sha256"), "0" * 64),
        ("manifest_mapping", ("manifest", "mapping", "scenario_id"), "corner"),
        ("geometry_count", ("geometry", "point_count"), 0),
        ("geometry_forbidden", ("geometry", "forbidden_scenario_geometry_paths"), ["/World/Wall"]),
        ("geometry_extra", ("geometry", "prim_paths_and_types"), [*original["geometry"]["prim_paths_and_types"], ["/World/Wall", "Mesh"]]),
        ("commands", ("commands", 0, 0, 0), 0.1),
        ("record_hash", ("evaluation_record", "sha256"), "0" * 64),
        ("record_metric", ("evaluation_record", "mapping", "metrics", "tracking_rmse_m_s", "denominator"), 999),
        ("receipt", ("environment_receipt", "step_count"), 159),
        ("coverage", ("coverage", "ray_samples"), 1),
        ("metric_count", ("metrics", 0, "denominator"), 999),
        ("termination", ("termination_counts", "reset"), 1),
        ("gate", ("gates", "geometry_exact"), False),
        ("claim", ("claim",), "performance pass"),
        ("actor_private", ("evaluation_record", "mapping", "actor_surface", "input_fields"), ["commanded_body_velocity_m_s", "pose"]),
        ("b0_source", ("evaluation_record", "mapping", "source_manifest", "commanded_velocity_b_m_s", "frame"), "world"),
        ("b0_observability_count", ("evaluation_record", "mapping", "observability_source_counts", "EXACT"), 15),
        ("b0_termination", ("evaluation_record", "mapping", "termination_counts", "done"), 1),
        ("b0_provenance", ("evaluation_record", "mapping", "provenance", "fixture_id"), "forged"),
    )
    for name, path, value in cases:
        candidate = copy.deepcopy(original)
        set_path(candidate, path, value)
        before = copy.deepcopy(candidate)
        try:
            accepted.append(OpenFieldQualificationRecord(_canonical(candidate)))
        except (TypeError, ValueError):
            pass
        else:
            pytest.fail(f"nested mutation accepted: {name}")
        assert candidate == before, name

    for name, mutate in (
        ("root_missing", lambda value: value.pop("claim")),
        ("root_extra", lambda value: value.update(private_pose=[0.0])),
        ("root_reordered", lambda value: value.move_to_end("schema_version")),
        ("geometry_private", lambda value: value["geometry"].update(private_wall=True)),
        ("geometry_reordered", lambda value: value["geometry"].move_to_end("prim_paths_and_types")),
        ("metric_private", lambda value: value["metrics"][0].update(private_count=1)),
        ("metric_reordered", lambda value: value["metrics"][0].move_to_end("name")),
        ("record_private", lambda value: value["evaluation_record"]["mapping"].update(private_state=True)),
        ("record_reordered", lambda value: value["evaluation_record"]["mapping"].move_to_end("schema_version")),
    ):
        candidate = copy.deepcopy(original); mutate(candidate); before = copy.deepcopy(candidate)
        with pytest.raises((TypeError, ValueError)):
            accepted.append(OpenFieldQualificationRecord(_canonical(candidate)))
        assert candidate == before, name
    assert accepted == [] and original == snapshot


def test_nested_b0_metric_forgery_is_rejected_even_with_recomputed_record_hash() -> None:
    mapping = json.loads(_qualification_payload(), object_pairs_hook=OrderedDict)
    metric = mapping["evaluation_record"]["mapping"]["metrics"]["tracking_rmse_m_s"]
    metric["denominator"] = 999
    mapping["evaluation_record"]["sha256"] = hashlib.sha256(_canonical(mapping["evaluation_record"]["mapping"]).encode("ascii")).hexdigest()
    with pytest.raises((TypeError, ValueError)):
        OpenFieldQualificationRecord(_canonical(mapping))


def _set_path(mapping: OrderedDict[str, object], path: tuple[str, ...], value: object) -> None:
    target = mapping
    for item in path[:-1]:
        target = target[item]
    target[path[-1]] = value


def _repair_record_hash(mapping: OrderedDict[str, object]) -> None:
    record = mapping["evaluation_record"]["mapping"]
    mapping["evaluation_record"]["sha256"] = hashlib.sha256(_canonical(record).encode("ascii")).hexdigest()


@pytest.mark.parametrize("name,path", (
    ("qualification", ("schema_version",)),
    ("run_plan", ("run_plan", "mapping", "schema_version")),
    ("manifest", ("manifest", "mapping", "schema_version")),
    ("b0_record", ("evaluation_record", "mapping", "schema_version")),
    ("outer_receipt", ("environment_receipt", "schema_version")),
    ("embedded_receipt", ("evaluation_record", "mapping", "receipt", "schema_version")),
    ("metric_config", ("evaluation_record", "mapping", "metric_config", "schema_version")),
))
def test_every_static_schema_field_rejects_atomic_mutation_without_input_change(name, path) -> None:
    mapping = json.loads(_qualification_payload(), object_pairs_hook=OrderedDict)
    _set_path(mapping, path, f"forged.{name}.v9")
    before = copy.deepcopy(mapping)
    accepted: list[OpenFieldQualificationRecord] = []
    with pytest.raises((TypeError, ValueError)):
        accepted.append(OpenFieldQualificationRecord(_canonical(mapping)))
    assert accepted == [] and mapping == before


@pytest.mark.parametrize("name", (
    "qualification", "run_plan", "manifest", "b0_record",
    "receipt_pair", "metric_config",
))
def test_every_static_schema_identity_rejects_consistent_mutation_after_digest_repairs(name) -> None:
    mapping = json.loads(_qualification_payload(), object_pairs_hook=OrderedDict)
    forged = f"forged.{name}.v9"
    if name == "qualification":
        mapping["schema_version"] = forged
    elif name == "run_plan":
        mapping["run_plan"]["mapping"]["schema_version"] = forged
        mapping["run_plan"]["sha256"] = hashlib.sha256(_canonical(mapping["run_plan"]["mapping"]).encode("ascii")).hexdigest()
    elif name == "manifest":
        manifest = mapping["manifest"]["mapping"]
        manifest["schema_version"] = forged
        manifest_sha = hashlib.sha256(_canonical(manifest).encode("ascii")).hexdigest()
        mapping["manifest"]["sha256"] = manifest_sha
        mapping["run_plan"]["mapping"]["manifest_sha256"] = manifest_sha
        mapping["run_plan"]["sha256"] = hashlib.sha256(_canonical(mapping["run_plan"]["mapping"]).encode("ascii")).hexdigest()
        mapping["environment_receipt"]["ready_manifest_sha256"] = manifest_sha
        mapping["evaluation_record"]["mapping"]["manifest"]["ready_sha256"] = manifest_sha
        _repair_record_hash(mapping)
    elif name == "b0_record":
        mapping["evaluation_record"]["mapping"]["schema_version"] = forged
        _repair_record_hash(mapping)
    elif name == "receipt_pair":
        mapping["environment_receipt"]["schema_version"] = forged
        mapping["evaluation_record"]["mapping"]["receipt"]["schema_version"] = forged
        _repair_record_hash(mapping)
    else:
        mapping["evaluation_record"]["mapping"]["metric_config"]["schema_version"] = forged
        _repair_record_hash(mapping)
    before = copy.deepcopy(mapping)
    accepted: list[OpenFieldQualificationRecord] = []
    with pytest.raises((TypeError, ValueError)):
        accepted.append(OpenFieldQualificationRecord(_canonical(mapping)))
    assert accepted == [] and mapping == before


def test_commander_f3_consistent_two_copy_receipt_schema_forgery_is_rejected_atomically() -> None:
    mapping = json.loads(_qualification_payload(), object_pairs_hook=OrderedDict)
    mapping["environment_receipt"]["schema_version"] = "forged.receipt.v9"
    mapping["evaluation_record"]["mapping"]["receipt"]["schema_version"] = "forged.receipt.v9"
    _repair_record_hash(mapping)
    before = copy.deepcopy(mapping)
    accepted: list[OpenFieldQualificationRecord] = []
    with pytest.raises((TypeError, ValueError)):
        accepted.append(OpenFieldQualificationRecord(_canonical(mapping)))
    assert accepted == [] and mapping == before
    assert B0_RUN_RECEIPT_SCHEMA_VERSION == "m7.b0-run-receipt.v1"


def test_duplicate_nested_key_and_nonfinite_json_are_rejected() -> None:
    payload = _qualification_payload()
    duplicate = payload.replace('"point_count":1849', '"point_count":1849,"point_count":1849', 1)
    nonfinite = payload.replace('"point_count":1849', '"point_count":NaN', 1)
    for candidate in (duplicate, nonfinite):
        with pytest.raises((TypeError, ValueError)):
            OpenFieldQualificationRecord(candidate)


def test_direct_import_isolation_and_no_aggregate_export_or_old_smoke_reference() -> None:
    source = SOURCE.read_text()
    imports = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(any(token in name for token in ("AppLauncher", "actor", "training", "legacy")) for name in imports)
    assert "run_m7_b0_runtime_evaluation_smoke" not in source
    assert "M7-012" not in source and "7012" not in source
    import instinctlab.evaluation as aggregate
    assert not hasattr(aggregate, "OpenFieldScenarioFixture")
    assert "OpenFieldScenarioFixture" not in aggregate.__all__
