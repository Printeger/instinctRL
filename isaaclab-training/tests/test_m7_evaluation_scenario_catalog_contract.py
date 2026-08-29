"""Pure/static contract tests for the M7-013 canonical scenario catalog."""

from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import FrozenInstanceError, fields
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import pytest

from instinctlab.evaluation.baselines import (
    ActorAuditStatus,
    BaselineRunManifest,
    CapabilityBinding,
    ExecutionStatus,
    build_run_manifest,
    resolve_baseline,
)
from instinctlab.evaluation.metrics import METRIC_KEYS
from instinctlab.evaluation.scenarios import (
    ACCEPTED_METRIC_NAMES,
    CATALOG_FIELD_ORDER,
    CATALOG_SCHEMA_VERSION,
    ExecutionDomain,
    FORWARDED_ACTOR_FIELDS,
    GENERATOR_PRIVILEGED_FIELDS,
    OPEN_FIELD_EVIDENCE_SHA256,
    OPEN_FIELD_PROVIDER,
    OPEN_FIELD_SEED,
    REAL_ROBOT_REPEATABILITY_METRIC,
    REQUIRED_RUN_LOG_FIELDS,
    RUN_PLAN_FIELD_ORDER,
    RUN_PLAN_SCHEMA_VERSION,
    RequirementClass,
    SCENARIO_EVIDENCE_FIELD_ORDER,
    SCENARIO_IDS,
    SCENARIO_ROW_FIELD_ORDER,
    SCENARIO_SELECTION_FIELD_ORDER,
    SCENARIO_SPEC_FIELD_ORDER,
    ScenarioCatalog,
    ScenarioEvidence,
    ScenarioId,
    ScenarioReadiness,
    ScenarioRunPlan,
    ScenarioSelection,
    ScenarioSpec,
    bind_scenario_run_plan,
    build_scenario_catalog,
    resolve_scenario,
    scenario_registry,
    select_scenario,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "instinctlab/evaluation/scenarios.py"
PRODUCT_COMMIT = "49c11d8f91305b847252d5e089165f2da6a3bda0"
EXPECTED_IDS = (
    "corner", "flat_wall", "corridor_tunnel", "open_field",
    "cluttered_indoor", "unsafe_command_toward_wall", "gap_doorway",
    "dynamic_obstacle", "real_robot_observable_holding",
    "real_robot_unsafe_command_approach",
)


def _bindings() -> tuple[CapabilityBinding, ...]:
    row = resolve_baseline("B0")
    return tuple(
        CapabilityBinding(item.name, item.accepted_provider, item.accepted_evidence_sha256)
        for item in row.capability_requirements
    )


def _manifest(
    scenario_id: str = "corner",
    seed: int = 13,
    status: ExecutionStatus = ExecutionStatus.READY,
) -> BaselineRunManifest:
    row = resolve_baseline("B0")
    return build_run_manifest(
        product_commit=PRODUCT_COMMIT,
        baseline_id="B0",
        config_switch=row.config_switch,
        capability_bindings=_bindings() if status in (ExecutionStatus.READY, ExecutionStatus.EXECUTED) else (),
        scenario_id=scenario_id,
        seed=seed,
        actor_audit_status=ActorAuditStatus.PASS if status in (ExecutionStatus.READY, ExecutionStatus.EXECUTED) else ActorAuditStatus.NOT_RUN,
        execution_status=status,
        skip_reason=None,
    )


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def test_exact_enum_values_and_public_record_field_order() -> None:
    assert tuple(item.value for item in ScenarioId) == EXPECTED_IDS == SCENARIO_IDS
    assert tuple(item.value for item in ExecutionDomain) == ("simulator", "real_robot")
    assert tuple(item.value for item in RequirementClass) == ("required", "optional_when_available")
    assert tuple(item.value for item in ScenarioReadiness) == ("NOT_READY", "READY")
    assert SCENARIO_EVIDENCE_FIELD_ORDER == tuple(item.name for item in fields(ScenarioEvidence))
    assert SCENARIO_SPEC_FIELD_ORDER == tuple(item.name for item in fields(ScenarioSpec))
    assert SCENARIO_SELECTION_FIELD_ORDER == ("scenario_id", "seed")
    assert tuple(item.name for item in fields(ScenarioCatalog)) == ("canonical_json",)
    assert tuple(item.name for item in fields(ScenarioRunPlan)) == ("canonical_json", "manifest")


@pytest.mark.parametrize("args", (
    ("NOT_READY", None, None, "reason"),
    (ScenarioReadiness.NOT_READY, "provider", "0" * 64, "reason"),
    (ScenarioReadiness.NOT_READY, None, None, None),
    (ScenarioReadiness.NOT_READY, None, None, " reason"),
    (ScenarioReadiness.READY, None, None, None),
    (ScenarioReadiness.READY, "provider", None, None),
    (ScenarioReadiness.READY, "provider", "not-a-hash", None),
    (ScenarioReadiness.READY, "provider", "0" * 64, "reason"),
))
def test_evidence_exact_type_and_paired_readiness_invariants(args) -> None:
    with pytest.raises((TypeError, ValueError)):
        ScenarioEvidence(*args)


def test_ad_hoc_ready_evidence_record_cannot_change_registry_truth() -> None:
    future = ScenarioEvidence(ScenarioReadiness.READY, "future.provider", "0" * 64, None)
    assert future.readiness is ScenarioReadiness.READY
    assert tuple(item.scenario_id for item in scenario_registry() if item.evidence.readiness is ScenarioReadiness.READY) == (ScenarioId.OPEN_FIELD,)


@pytest.mark.parametrize("record,args", (
    (ScenarioEvidence, (ScenarioReadiness.NOT_READY, None, None, "reason")),
    (ScenarioSelection, (ScenarioId.CORNER, 0)),
    (ScenarioCatalog, (build_scenario_catalog().canonical_json,)),
    (ScenarioRunPlan, (bind_scenario_run_plan(selection=select_scenario(scenario_id="corner", seed=13), manifest=_manifest()).canonical_json, _manifest())),
))
def test_public_records_are_frozen_and_non_subclassable(record, args) -> None:
    value = record(*args)
    with pytest.raises(FrozenInstanceError):
        setattr(value, fields(value)[0].name, None)
    with pytest.raises(TypeError):
        type("Forbidden", (record,), {})


def test_exact_registry_order_domains_optional_and_sole_ready_truth() -> None:
    rows = scenario_registry()
    assert type(rows) is tuple and len(rows) == 10
    assert tuple(item.scenario_id.value for item in rows) == EXPECTED_IDS
    assert tuple(item.execution_domain for item in rows[:8]) == (ExecutionDomain.SIMULATOR,) * 8
    assert tuple(item.execution_domain for item in rows[8:]) == (ExecutionDomain.REAL_ROBOT,) * 2
    assert tuple(item.scenario_id for item in rows if item.requirement_class is RequirementClass.OPTIONAL_WHEN_AVAILABLE) == (ScenarioId.DYNAMIC_OBSTACLE,)
    ready = rows[3]
    assert ready.scenario_id is ScenarioId.OPEN_FIELD and ready.evidence.readiness is ScenarioReadiness.READY
    assert (ready.evidence.provider, ready.evidence.evidence_sha256, ready.evidence.unavailability_reason) == (OPEN_FIELD_PROVIDER, OPEN_FIELD_EVIDENCE_SHA256, None)
    fixture_source = ROOT / "instinctlab/evaluation/scenario_open_field.py"
    assert hashlib.sha256(fixture_source.read_bytes()).hexdigest() == OPEN_FIELD_EVIDENCE_SHA256
    not_ready = rows[:3] + rows[4:]
    assert len(not_ready) == 9 and all(item.evidence.readiness is ScenarioReadiness.NOT_READY for item in not_ready)
    assert all(item.evidence.provider is None and item.evidence.evidence_sha256 is None for item in not_ready)
    assert all(type(item.evidence.unavailability_reason) is str and item.evidence.unavailability_reason for item in not_ready)
    assert all(item.requirement_class is RequirementClass.REQUIRED for item in rows if item.scenario_id is not ScenarioId.DYNAMIC_OBSTACLE)


def test_registry_and_lookup_outputs_are_independently_immutable() -> None:
    first = scenario_registry()
    second = scenario_registry()
    assert first == second and first is not second
    assert all(left is not right and left.evidence is not right.evidence for left, right in zip(first, second, strict=True))
    one = resolve_scenario("corner")
    two = resolve_scenario(ScenarioId.CORNER)
    assert one == two and one is not two and one.evidence is not two.evidence


@pytest.mark.parametrize("value", (
    "Corner", "CORNER", " corner", "corner ", "corner/", "flat-wall",
    "corridor/tunnel", "dynamic", "unknown", "", 1, None, object(),
))
def test_lookup_rejects_alias_normalization_unknown_and_wrong_types(value) -> None:
    with pytest.raises((TypeError, KeyError)):
        resolve_scenario(value)


def test_lookup_rejects_string_subclass() -> None:
    class StringAlias(str):
        pass
    with pytest.raises(TypeError):
        resolve_scenario(StringAlias("corner"))


def test_privacy_boundary_and_required_log_identity_are_exact() -> None:
    assert FORWARDED_ACTOR_FIELDS == ("commanded_body_velocity_m_s",)
    assert GENERATOR_PRIVILEGED_FIELDS == (
        "pose", "velocity", "obstacle_state", "mesh", "ray_hits",
        "surface_normals", "odometry", "map", "ground_truth",
    )
    assert REQUIRED_RUN_LOG_FIELDS == (
        "product_commit", "platform", "sensor", "baseline_id", "scenario_id",
        "seed", "actor_audit_status", "scenario_readiness", "scenario_provider",
        "scenario_evidence_sha256", "scenario_unavailability_reason",
        "execution_domain", "requirement_class",
    )
    for row in scenario_registry():
        assert row.forwarded_actor_fields == FORWARDED_ACTOR_FIELDS
        assert row.generator_privileged_fields == GENERATOR_PRIVILEGED_FIELDS
        assert not set(row.forwarded_actor_fields).intersection(row.generator_privileged_fields)
        assert row.required_run_log_fields == REQUIRED_RUN_LOG_FIELDS


def test_metric_obligations_reference_exact_accepted_names_without_values() -> None:
    assert ACCEPTED_METRIC_NAMES == METRIC_KEYS
    for row in scenario_registry()[:8]:
        assert row.metric_obligations == METRIC_KEYS
    for row in scenario_registry()[8:]:
        assert row.metric_obligations == METRIC_KEYS + (REAL_ROBOT_REPEATABILITY_METRIC,)
    source = SOURCE.read_text()
    assert "from .metrics" not in source and "compute_step_metrics" not in source
    assert all(token not in source for token in ("threshold", "ranking", "denominator", "metric_value"))


@pytest.mark.parametrize("args", (
    {"scenario_id": "corner", "seed": True},
    {"scenario_id": "corner", "seed": -1},
    {"scenario_id": "corner", "seed": 1.0},
    {"scenario_id": "Corner", "seed": 0},
    {"scenario_id": "corner ", "seed": 0},
    {"scenario_id": object(), "seed": 0},
))
def test_selection_rejects_seed_and_id_coercion(args) -> None:
    with pytest.raises((TypeError, ValueError, KeyError)):
        select_scenario(**args)


def test_selection_signatures_have_no_defaults_or_execution_surface() -> None:
    assert tuple(inspect.signature(select_scenario).parameters) == ("scenario_id", "seed")
    assert all(item.default is inspect.Parameter.empty for item in inspect.signature(select_scenario).parameters.values())
    assert tuple(inspect.signature(bind_scenario_run_plan).parameters) == ("selection", "manifest")
    selection = select_scenario(scenario_id="corner", seed=13)
    assert selection == ScenarioSelection(ScenarioId.CORNER, 13)
    assert not hasattr(selection, "execute") and not hasattr(selection, "callback")


def test_manifest_binding_reparses_owns_and_preserves_exact_identity() -> None:
    manifest = _manifest()
    original = manifest.canonical_json
    selection = select_scenario(scenario_id="corner", seed=13)
    plan = bind_scenario_run_plan(selection=selection, manifest=manifest)
    mapping = plan.to_mapping()
    assert plan.manifest is not manifest
    assert plan.manifest.canonical_json == manifest.canonical_json == original
    assert mapping["schema_version"] == RUN_PLAN_SCHEMA_VERSION
    assert tuple(mapping) == RUN_PLAN_FIELD_ORDER
    assert mapping["manifest_sha256"] == manifest.manifest_sha256
    assert mapping["product_commit"] == PRODUCT_COMMIT
    assert (mapping["platform"], mapping["sensor"], mapping["baseline_id"]) == ("TASLAB_UAV", "Livox MID360", "B0")
    assert (mapping["scenario_id"], mapping["seed"]) == ("corner", 13)
    assert mapping["actor_audit_status"] == "PASS" and mapping["baseline_execution_status"] == "READY"
    assert mapping["scenario_readiness"] == "NOT_READY" and mapping["executable"] is False
    assert mapping["scenario_provider"] is None and mapping["scenario_evidence_sha256"] is None
    assert not hasattr(plan, "execute") and not hasattr(plan, "callback")


@pytest.mark.parametrize("selection,manifest", (
    (select_scenario(scenario_id="corner", seed=13), _manifest("flat_wall", 13)),
    (select_scenario(scenario_id="corner", seed=13), _manifest("corner", 14)),
    (object(), _manifest()),
    (select_scenario(scenario_id="corner", seed=13), object()),
))
def test_manifest_binding_rejects_mismatch_and_wrong_exact_types(selection, manifest) -> None:
    with pytest.raises((TypeError, ValueError)):
        bind_scenario_run_plan(selection=selection, manifest=manifest)


def test_manifest_and_selection_subclasses_are_rejected() -> None:
    class ManifestSubclass(BaselineRunManifest):
        pass
    manifest = ManifestSubclass(_manifest().canonical_json)
    with pytest.raises(TypeError):
        bind_scenario_run_plan(selection=select_scenario(scenario_id="corner", seed=13), manifest=manifest)


@pytest.mark.parametrize("payload", ("{}", '{ "bad":1}', '{"bad":NaN}', '{"bad":1,"bad":1}'))
def test_manifest_binding_reparses_and_rejects_malformed_noncanonical_bytes(payload: str) -> None:
    forged = object.__new__(BaselineRunManifest)
    object.__setattr__(forged, "canonical_json", payload)
    with pytest.raises((TypeError, ValueError, KeyError)):
        bind_scenario_run_plan(
            selection=select_scenario(scenario_id="corner", seed=13), manifest=forged,
        )


def test_catalog_canonical_bytes_hash_round_trip_and_exact_rows() -> None:
    first = build_scenario_catalog()
    second = build_scenario_catalog()
    assert first is not second and first.canonical_json == second.canonical_json
    assert first.catalog_sha256 == second.catalog_sha256 == hashlib.sha256(first.to_bytes()).hexdigest()
    assert ScenarioCatalog.from_json(first.canonical_json) == first
    mapping = first.to_mapping()
    assert tuple(mapping) == CATALOG_FIELD_ORDER
    assert mapping["schema_version"] == CATALOG_SCHEMA_VERSION
    assert mapping["scenario_order"] == list(EXPECTED_IDS)
    assert len(mapping["rows"]) == 10
    assert all(tuple(row) == SCENARIO_ROW_FIELD_ORDER for row in mapping["rows"])
    assert tuple(row["scenario_id"] for row in mapping["rows"] if row["executable"]) == ("open_field",)
    assert _canonical(mapping) == first.canonical_json


@pytest.mark.parametrize("mutation", ("missing", "extra", "reordered", "changed", "private", "row_reordered"))
def test_catalog_canonical_mutations_reject(mutation: str) -> None:
    mapping = json.loads(build_scenario_catalog().canonical_json, object_pairs_hook=OrderedDict)
    if mutation == "missing":
        mapping.pop("sensor")
    elif mutation == "extra":
        mapping["extra"] = None
    elif mutation == "reordered":
        mapping.move_to_end("schema_version")
    elif mutation == "changed":
        mapping["rows"][0]["readiness"] = "READY"
    elif mutation == "private":
        mapping["rows"][0]["private_pose"] = [0.0]
    else:
        mapping["rows"][0].move_to_end("scenario_id")
    with pytest.raises(ValueError):
        ScenarioCatalog(_canonical(mapping))


@pytest.mark.parametrize("payload", (
    '{"schema_version":"m7.scenario-catalog.v1","schema_version":"m7.scenario-catalog.v1"}',
    '{ "schema_version":"m7.scenario-catalog.v1"}',
    '{"schema_version":"m7.scenario-catalog.v1","value":NaN}',
    '[]',
))
def test_catalog_json_rejects_duplicate_noncanonical_nonfinite_and_nonobject(payload: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        ScenarioCatalog(payload)


@pytest.mark.parametrize("mutation", ("missing", "extra", "reordered", "changed", "private"))
def test_run_plan_canonical_mutations_reject(mutation: str) -> None:
    manifest = _manifest()
    plan = bind_scenario_run_plan(selection=select_scenario(scenario_id="corner", seed=13), manifest=manifest)
    mapping = json.loads(plan.canonical_json, object_pairs_hook=OrderedDict)
    if mutation == "missing":
        mapping.pop("seed")
    elif mutation == "extra":
        mapping["extra"] = None
    elif mutation == "reordered":
        mapping.move_to_end("schema_version")
    elif mutation == "changed":
        mapping["scenario_readiness"] = "READY"
    else:
        mapping["private_obstacle_state"] = [0.0]
    with pytest.raises(ValueError):
        ScenarioRunPlan(_canonical(mapping), manifest)


def test_run_plan_hash_round_trip_is_exact_and_manifest_bound() -> None:
    manifest = _manifest("real_robot_observable_holding", 0, ExecutionStatus.EXECUTED)
    selection = select_scenario(scenario_id="real_robot_observable_holding", seed=0)
    plan = bind_scenario_run_plan(selection=selection, manifest=manifest)
    assert plan.run_plan_sha256 == hashlib.sha256(plan.to_bytes()).hexdigest()
    assert ScenarioRunPlan.from_json(plan.canonical_json, manifest) == plan
    assert plan.to_mapping()["metric_obligations"] == list(METRIC_KEYS + (REAL_ROBOT_REPEATABILITY_METRIC,))
    assert plan.to_mapping()["baseline_execution_status"] == "EXECUTED"
    assert plan.to_mapping()["scenario_readiness"] == "NOT_READY"


def test_exact_open_field_ready_plan_is_sole_executable_plan() -> None:
    manifest = _manifest("open_field", OPEN_FIELD_SEED, ExecutionStatus.READY)
    plan = bind_scenario_run_plan(
        selection=select_scenario(scenario_id="open_field", seed=OPEN_FIELD_SEED),
        manifest=manifest,
    )
    mapping = plan.to_mapping()
    assert mapping["executable"] is True
    assert mapping["scenario_provider"] == OPEN_FIELD_PROVIDER
    assert mapping["scenario_evidence_sha256"] == OPEN_FIELD_EVIDENCE_SHA256
    assert not hasattr(plan, "execute") and not hasattr(plan, "callback")


@pytest.mark.parametrize("scenario_id,seed,status", (
    ("open_field", OPEN_FIELD_SEED, ExecutionStatus.EXECUTED),
    ("open_field", OPEN_FIELD_SEED, ExecutionStatus.NOT_READY),
    ("open_field", OPEN_FIELD_SEED + 1, ExecutionStatus.READY),
    ("corner", OPEN_FIELD_SEED, ExecutionStatus.READY),
))
def test_nonexact_status_identity_or_seed_is_nonexecutable(scenario_id, seed, status) -> None:
    manifest = _manifest(scenario_id, seed, status)
    plan = bind_scenario_run_plan(
        selection=select_scenario(scenario_id=scenario_id, seed=seed), manifest=manifest,
    )
    assert plan.to_mapping()["executable"] is False


def test_direct_import_isolation_import_time_purity_and_no_forbidden_paths() -> None:
    source = SOURCE.read_text()
    tree = ast.parse(source)
    imported = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module)
        elif isinstance(node, ast.Call):
            calls.append(ast.unparse(node.func))
    assert all(name is None or not any(token in name for token in (
        "metrics", "torch", "isaac", "env", "actor", "training", "legacy",
    )) for name in imported)
    assert all(token not in source for token in (
        "RuntimeEvaluationCollector", "B0RuntimeEvaluationCollector", "AppLauncher",
        "cuda", "scenario_generator", "optimizer", "checkpoint",
        "run_m7_b0_runtime_evaluation_smoke", "universal_generator",
    ))
    assert not any(call.endswith("step") or call.endswith("execute") for call in calls)
    import instinctlab.evaluation as aggregate
    assert "ScenarioCatalog" not in aggregate.__all__
    assert not hasattr(aggregate, "ScenarioCatalog")


def test_b0_b8_registry_and_m7_012_collector_source_are_unchanged() -> None:
    baseline_source = ROOT / "instinctlab/evaluation/baselines.py"
    collector_source = ROOT / "instinctlab/evaluation/baseline_runtime.py"
    assert hashlib.sha256(baseline_source.read_bytes()).hexdigest() == "725d6d1bf57a5c168d96deaa7cb7179382caa64b101ad0fd79172626dc8272b2"
    assert hashlib.sha256(collector_source.read_bytes()).hexdigest() == "c3f1e6f73f48bdcbf995c6f8178d5b93be2d6107af00d81b0ceac69c4a358e66"
    rows = tuple(resolve_baseline(f"B{index}") for index in range(9))
    assert tuple(row.baseline_id.value for row in rows) == tuple(f"B{index}" for index in range(9))
    assert rows[0].readiness.value == "READY" and len(rows[0].capability_requirements) == 8
    assert all(row.readiness.value == "NOT_READY" for row in rows[1:])


def test_no_existing_smoke_is_relabelled_as_handbook_scenario() -> None:
    source = SOURCE.read_text()
    assert PRODUCT_COMMIT not in source
    assert "M7-012.runtime-evaluation-smoke" not in source
    assert all(item.evidence.unavailability_reason is None or "smoke" not in item.evidence.unavailability_reason for item in scenario_registry())


def test_d0001_non_open_field_registry_rows_remain_byte_equivalent() -> None:
    reviewed = subprocess.run(
        ["git", "show", "72c4f4b60fece213d368e0def3921bcb29e38a81:isaaclab-training/instinctlab/evaluation/scenarios.py"],
        cwd=ROOT.parent, check=True, capture_output=True, text=True,
    ).stdout
    current = SOURCE.read_text()
    def non_open_lines(source: str) -> tuple[str, ...]:
        registry = source.split("_SCENARIO_REGISTRY = (", 1)[1].split("\n)\n", 1)[0]
        return tuple(line for line in registry.splitlines() if "_spec(ScenarioId." in line and "ScenarioId.OPEN_FIELD" not in line)
    assert len(non_open_lines(current)) == 9
    assert non_open_lines(current) == non_open_lines(reviewed)
