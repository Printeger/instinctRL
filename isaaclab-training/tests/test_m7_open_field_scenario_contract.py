"""Pure/static tests for the exact M7-014 open-field B0 fixture."""

from __future__ import annotations

import ast
from dataclasses import dataclass, FrozenInstanceError, fields
import hashlib
import importlib
import inspect
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


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "instinctlab/evaluation/scenario_open_field.py"
PRODUCT_COMMIT = "deeb7ad4221df1f0e74ec0ac24affefbc27b817d"


def _geometry() -> OpenFieldGeometryEvidence:
    axis = tuple(float(value) for value in range(-42, 43, 2))
    return OpenFieldGeometryEvidence(
        ((GROUND_PRIM_PATH, "Xform"), (RAY_SCENE_PRIM_PATH, "Xform"), (RAY_MESH_PRIM_PATH, "Mesh")),
        GROUND_PRIM_PATH, RAY_SCENE_PRIM_PATH, RAY_MESH_PRIM_PATH,
        RAYCASTER_MESH_PRIM_PATHS, POINT_COUNT, TRIANGLE_COUNT,
        FACE_INDEX_COUNT, (-42.0, -42.0, 0.0), (42.0, 42.0, 0.0),
        axis, axis, (0.0,),
    )


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
        "prim_paths_and_types", "ground_prim_path", "ray_scene_prim_path",
        "ray_mesh_prim_path", "raycaster_mesh_prim_paths", "point_count",
        "triangle_count", "face_index_count", "minimum_xyz_m", "maximum_xyz_m",
        "unique_x_m", "unique_y_m", "unique_z_m",
    )
    with pytest.raises(FrozenInstanceError):
        geometry.point_count = 0
    with pytest.raises(TypeError):
        type("Forbidden", (OpenFieldGeometryEvidence,), {})


@pytest.mark.parametrize("field,value", (
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
    for payload in ("{}", '{ "schema_version":"m7.open-field-b0-qualification.v1"}', '{"schema_version":"bad"}'):
        with pytest.raises((TypeError, ValueError)):
            OpenFieldQualificationRecord(payload)
    assert QUALIFICATION_SCHEMA_VERSION == "m7.open-field-b0-qualification.v1"


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
