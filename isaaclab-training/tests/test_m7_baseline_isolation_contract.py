"""Focused registry, manifest, negative, ownership, and import-isolation tests."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import FrozenInstanceError, replace
import ast
import hashlib
import json
from pathlib import Path

import pytest

from instinctlab.evaluation import (
    ActorAuditStatus,
    BASELINE_CONFIG_SWITCHES,
    BASELINE_IDS,
    BASELINE_REGISTRY,
    BASELINE_SPEC_FIELD_ORDER,
    BaselineId,
    BaselineRunManifest,
    CAPABILITY_BINDING_FIELD_ORDER,
    COMPONENT_TOGGLE_FIELD_ORDER,
    CapabilityBinding,
    CapabilityRequirement,
    ComponentToggles,
    EVALUATION_ONLY_FIELDS,
    ExecutionFamily,
    ExecutionStatus,
    FULL_ACTOR_INPUT_FIELDS,
    MANIFEST_FIELD_ORDER,
    MANIFEST_PAYLOAD_FIELD_ORDER,
    MANIFEST_SCHEMA_VERSION,
    PrivilegeClass,
    ReadinessState,
    build_run_manifest,
    resolve_baseline,
    resolve_config_switch,
    validate_baseline_registry,
    validate_capability_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "dbc05a58ca98958c9100fa5e002fe6b9083e40ec"
B0_GOVERNOR_SHA256 = "cd5cf6c73578bc69dc5f696ac6d294d9d4c8db0baaf12265e6715646554b7197"
B0_ROTATION_SHA256 = "a79ea8d88fb0391762a9efcebbfa783d5287ee1393d3b6e2757812032c32991d"
B0_ADAPTER_SHA256 = "5fa2b8d746eb231a1bb12f331fe1a76e8f988005e9a315e1dbf622e0bf1ed3d1"
B0_EXECUTION_SHA256 = "db7b73ee7e9c4b39689814530363bafb6a1ef48c2513a0208c0ff29c7073c108"
B0_CONTROLLER_SHA256 = "72d104850f8c8fede59998f14661b5b7ea7cd2c113ec11218fe232337732ab52"
B0_RUN_SHA256 = "5210566e987b737657d30ba44075a4dc835e379eae438726449dbc76002ab847"
B0_ENVIRONMENT_SHA256 = "714080d04dee16e73c408f6c0e8c7c8d4a5d0b5a603e698714c5a330c8509e5d"


def _b0_bindings() -> tuple[CapabilityBinding, ...]:
    requirements = resolve_baseline(BaselineId.B0).capability_requirements
    assert all(item.accepted_provider is not None for item in requirements)
    assert all(item.accepted_evidence_sha256 is not None for item in requirements)
    return tuple(
        CapabilityBinding(item.name, item.accepted_provider, item.accepted_evidence_sha256)
        for item in requirements
    )


def _manifest(
    baseline_id: str = "B0",
    status: ExecutionStatus = ExecutionStatus.READY,
) -> BaselineRunManifest:
    row = resolve_baseline(baseline_id)
    bindings = _b0_bindings() if status in (ExecutionStatus.READY, ExecutionStatus.EXECUTED) else ()
    audit = (
        ActorAuditStatus.PASS
        if status in (ExecutionStatus.READY, ExecutionStatus.EXECUTED)
        else ActorAuditStatus.NOT_RUN
    )
    reason = "capability_unavailable" if status is ExecutionStatus.SKIPPED else None
    return build_run_manifest(
        product_commit=BASE_COMMIT,
        baseline_id=baseline_id,
        config_switch=row.config_switch,
        capability_bindings=bindings,
        scenario_id="S1.corner",
        seed=7,
        actor_audit_status=audit,
        execution_status=status,
        skip_reason=reason,
    )


def _resign(mapping: OrderedDict[str, object]) -> str:
    payload = OrderedDict((name, mapping[name]) for name in MANIFEST_PAYLOAD_FIELD_ORDER)
    payload_bytes = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("ascii")
    mapping["manifest_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    return json.dumps(mapping, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def test_exact_canonical_rows_and_every_field() -> None:
    assert BASELINE_IDS == tuple(f"B{index}" for index in range(9))
    assert BASELINE_CONFIG_SWITCHES == (
        "direct_velocity",
        "state_velocity_upper_bound",
        "no_history",
        "no_anchor",
        "no_imu",
        "no_ics",
        "distance_only_filter",
        "braking_heuristic",
        "simplified_cbf_vo_optional",
    )
    assert BASELINE_SPEC_FIELD_ORDER == (
        "baseline_id", "config_switch", "execution_family", "description",
        "contribution_tested", "deployable", "privilege_class", "actor_input_fields",
        "evaluation_only_fields", "component_toggles", "capability_requirements",
        "optional", "readiness",
    )
    assert COMPONENT_TOGGLE_FIELD_ORDER == ("history", "imu", "anchor", "ics")
    assert CAPABILITY_BINDING_FIELD_ORDER == ("name", "provider", "evidence_sha256")
    assert tuple(row.baseline_id.value for row in BASELINE_REGISTRY) == BASELINE_IDS
    assert tuple(row.config_switch for row in BASELINE_REGISTRY) == BASELINE_CONFIG_SWITCHES
    assert tuple(row.execution_family for row in BASELINE_REGISTRY) == (
        ExecutionFamily.DIRECT_CONTROL,
        ExecutionFamily.LEARNED_GOVERNOR,
        ExecutionFamily.LEARNED_GOVERNOR,
        ExecutionFamily.LEARNED_GOVERNOR,
        ExecutionFamily.LEARNED_GOVERNOR,
        ExecutionFamily.LEARNED_GOVERNOR,
        ExecutionFamily.DETERMINISTIC_FILTER,
        ExecutionFamily.HEURISTIC_FILTER,
        ExecutionFamily.MODEL_BASED_FILTER,
    )
    assert tuple(row.description for row in BASELINE_REGISTRY) == (
        "Direct velocity execution with no learned governor, anchor, or intervention.",
        "State/velocity-aided governor used only as a privileged diagnostic upper bound.",
        "Governor using one current MID360, IMU, and command frame without history.",
        "Full actor inputs with measurement-space anchor loss and features disabled.",
        "Governor using MID360 history, commands, and previous output without IMU cues.",
        "Full learned governor with ICS-inspired command attenuation disabled.",
        "Deterministic command attenuation based only on minimum range.",
        "Non-learned braking-feasibility attenuation using MID360 range-rate history.",
        "Optional local model-based CBF/VO-style command filter.",
    )
    assert tuple(row.contribution_tested for row in BASELINE_REGISTRY) == (
        "TASLAB velocity controller alone.",
        "Gap to the forbidden state-aided policy.",
        "Value of temporal history for motion inference.",
        "Measurement-space anchor contribution to station keeping.",
        "IMU contribution to short-horizon inference and robustness.",
        "ICS attenuation contribution to unsafe-command outcomes.",
        "Distance-only reactivity compared with braking feasibility.",
        "Gain from learning beyond the hand-designed braking rule.",
        "Comparison with a conventional local safety-filter style baseline.",
    )
    assert tuple(row.component_toggles for row in BASELINE_REGISTRY) == (
        ComponentToggles(False, False, False, False),
        ComponentToggles(True, True, True, True),
        ComponentToggles(False, True, True, True),
        ComponentToggles(True, True, False, True),
        ComponentToggles(True, False, True, True),
        ComponentToggles(True, True, True, False),
        ComponentToggles(False, False, False, False),
        ComponentToggles(True, False, False, True),
        ComponentToggles(False, False, False, False),
    )
    assert all(row.description and row.contribution_tested for row in BASELINE_REGISTRY)
    assert all(row.evaluation_only_fields == EVALUATION_ONLY_FIELDS for row in BASELINE_REGISTRY)
    assert tuple(row.optional for row in BASELINE_REGISTRY) == (False,) * 8 + (True,)
    assert tuple(row.readiness for row in BASELINE_REGISTRY) == (
        ReadinessState.READY,
    ) + (ReadinessState.NOT_READY,) * 8
    assert tuple(item.name for row in BASELINE_REGISTRY for item in row.capability_requirements) == (
        "m5.b0_direct_velocity",
        "m5.b0_body_to_world",
        "m7.b0_direct_adapter",
        "m7.b0_manifest_execution",
        "m7.b0_taslab_controller_handoff",
        "m7.b0_run_lifecycle",
        "m7.b0_taslab_environment",
        "m7.b1_state_velocity_upper_bound_adapter",
        "m7.b2_no_history_adapter",
        "m7.b3_no_anchor_adapter",
        "m7.b4_no_imu_adapter",
        "m7.b5_no_ics_adapter",
        "m7.b6_distance_only_filter",
        "m7.b7_braking_heuristic",
        "m7.b8_simplified_cbf_vo_filter",
    )


def test_exact_policy_schemas_and_b0_evidence_binding() -> None:
    assert resolve_baseline("B0").actor_input_fields == ("commanded_body_velocity_m_s",)
    assert resolve_baseline("B1").actor_input_fields == FULL_ACTOR_INPUT_FIELDS + (
        "explicit_body_velocity_m_s", "local_odometry",
    )
    assert resolve_baseline("B2").actor_input_fields == FULL_ACTOR_INPUT_FIELDS[:-1]
    assert resolve_baseline("B3").actor_input_fields == FULL_ACTOR_INPUT_FIELDS
    assert resolve_baseline("B4").actor_input_fields == (
        "ranges_m", "validity", "sensor_frame_time_s", "policy_time_s",
        "commanded_body_velocity_m_s", "previous_issued_body_velocity_m_s",
    )
    assert resolve_baseline("B5").actor_input_fields == FULL_ACTOR_INPUT_FIELDS
    assert resolve_baseline("B6").actor_input_fields == (
        "minimum_range_m", "commanded_body_velocity_m_s",
    )
    assert resolve_baseline("B7").actor_input_fields == (
        "ranges_m", "validity", "sensor_frame_time_s", "policy_time_s",
        "commanded_body_velocity_m_s",
    )
    assert resolve_baseline("B8").actor_input_fields == resolve_baseline("B7").actor_input_fields
    expected_sources = {
        "instinctlab/control/governor.py": B0_GOVERNOR_SHA256,
        "instinctlab/control/command_adapter.py": B0_ROTATION_SHA256,
        "instinctlab/control/baseline_b0.py": B0_ADAPTER_SHA256,
        "instinctlab/evaluation/baseline_execution.py": B0_EXECUTION_SHA256,
        "instinctlab/evaluation/baseline_controller_execution.py": B0_CONTROLLER_SHA256,
        "instinctlab/evaluation/baseline_run.py": B0_RUN_SHA256,
        "instinctlab/evaluation/baseline_environment.py": B0_ENVIRONMENT_SHA256,
    }
    assert {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in expected_sources
    } == expected_sources
    requirements = resolve_baseline("B0").capability_requirements
    assert tuple(item.name for item in requirements) == (
        "m5.b0_direct_velocity", "m5.b0_body_to_world", "m7.b0_direct_adapter",
        "m7.b0_manifest_execution",
        "m7.b0_taslab_controller_handoff",
        "m7.b0_run_lifecycle",
        "m7.b0_taslab_environment",
    )
    assert tuple(item.accepted_provider for item in requirements) == (
        "instinctlab.control.governor.b0_governor",
        "instinctlab.control.command_adapter.rotate_body_to_world",
        "instinctlab.control.baseline_b0.execute_b0_direct_velocity",
        "instinctlab.evaluation.baseline_execution.execute_ready_b0",
        "instinctlab.evaluation.baseline_controller_execution.execute_ready_b0_controller",
        "instinctlab.evaluation.baseline_run.begin_ready_b0_run",
        "instinctlab.evaluation.baseline_environment.TaslabB0EvaluationEnv",
    )
    assert tuple(item.accepted_evidence_sha256 for item in requirements) == (
        B0_GOVERNOR_SHA256, B0_ROTATION_SHA256, B0_ADAPTER_SHA256,
        B0_EXECUTION_SHA256,
        B0_CONTROLLER_SHA256,
        B0_RUN_SHA256,
        B0_ENVIRONMENT_SHA256,
    )
    assert all(
        row.capability_requirements[0].accepted_provider is None
        and row.capability_requirements[0].accepted_evidence_sha256 is None
        for row in BASELINE_REGISTRY[1:]
    )


def test_b1_privilege_and_all_deployable_rows_are_actor_clean() -> None:
    b1 = resolve_baseline("B1")
    assert not b1.deployable
    assert b1.privilege_class is PrivilegeClass.PRIVILEGED_UPPER_BOUND
    assert {"explicit_body_velocity_m_s", "local_odometry"}.issubset(b1.actor_input_fields)
    forbidden = (
        "pose", "position", "odometry", "explicit_body_velocity", "actual_velocity",
        "privileged", "velocity_estimate", "map", "slam", "root_state",
        "obstacle_truth", "surface_normal", "future",
    )
    for row in BASELINE_REGISTRY:
        if not row.deployable:
            continue
        assert row.privilege_class is PrivilegeClass.ACTOR_CLEAN
        assert not any(token in field for field in row.actor_input_fields for token in forbidden)


@pytest.mark.parametrize(
    "value",
    ("", "b0", " B0", "B0 ", "B00", "B9", "direct_velocity", "full", "ours"),
)
def test_unknown_alias_case_whitespace_and_switch_as_id_reject(value: str) -> None:
    with pytest.raises(KeyError):
        resolve_baseline(value)


@pytest.mark.parametrize(
    "value",
    ("", "DIRECT_VELOCITY", " direct_velocity", "direct_velocity ", "B0", "ours", "full"),
)
def test_unknown_alias_case_whitespace_and_id_as_switch_reject(value: str) -> None:
    with pytest.raises(KeyError):
        resolve_config_switch(value)


def test_registry_frozen_and_incomplete_reordered_duplicate_or_false_ready_rejects() -> None:
    with pytest.raises(FrozenInstanceError):
        BASELINE_REGISTRY[0].optional = True
    with pytest.raises(TypeError):
        validate_baseline_registry(list(BASELINE_REGISTRY))
    with pytest.raises(ValueError):
        validate_baseline_registry(BASELINE_REGISTRY[:-1])
    with pytest.raises(ValueError):
        validate_baseline_registry((BASELINE_REGISTRY[1], BASELINE_REGISTRY[0]) + BASELINE_REGISTRY[2:])
    duplicate_switch = replace(BASELINE_REGISTRY[1], config_switch="direct_velocity")
    with pytest.raises(ValueError):
        validate_baseline_registry((BASELINE_REGISTRY[0], duplicate_switch) + BASELINE_REGISTRY[2:])
    false_ready = replace(
        BASELINE_REGISTRY[1],
        capability_requirements=(
            CapabilityRequirement(
                "m7.b1_state_velocity_upper_bound_adapter",
                "instinctlab.control.governor.b0_governor",
                B0_GOVERNOR_SHA256,
            ),
        ),
        readiness=ReadinessState.READY,
    )
    with pytest.raises(ValueError):
        validate_baseline_registry((BASELINE_REGISTRY[0], false_ready) + BASELINE_REGISTRY[2:])


def test_independently_reconstructed_value_identical_registry_passes() -> None:
    reconstructed = tuple(replace(row) for row in BASELINE_REGISTRY)
    assert reconstructed is not BASELINE_REGISTRY
    assert all(actual is not canonical for actual, canonical in zip(
        reconstructed, BASELINE_REGISTRY, strict=True
    ))
    validate_baseline_registry(reconstructed)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("baseline_id", BaselineId.B3),
        ("config_switch", "wrong_switch"),
        ("execution_family", ExecutionFamily.DIRECT_CONTROL),
        ("description", "wrong"),
        ("contribution_tested", "wrong"),
        ("deployable", False),
        ("privilege_class", PrivilegeClass.PRIVILEGED_UPPER_BOUND),
        ("actor_input_fields", ("commanded_body_velocity_m_s",)),
        ("evaluation_only_fields", ("wrong",)),
        ("component_toggles", ComponentToggles(True, True, True, True)),
        (
            "capability_requirements",
            (CapabilityRequirement("wrong.capability", None, None),),
        ),
        ("optional", True),
        ("readiness", ReadinessState.READY),
    ),
)
def test_every_baseline_spec_field_drift_rejects(field_name: str, value: object) -> None:
    mutated = replace(BASELINE_REGISTRY[2], **{field_name: value})
    candidate = BASELINE_REGISTRY[:2] + (mutated,) + BASELINE_REGISTRY[3:]
    with pytest.raises((TypeError, ValueError)):
        validate_baseline_registry(candidate)


def test_capabilities_reject_mutability_alias_duplicates_order_hash_and_fallback() -> None:
    b0, b1 = resolve_baseline("B0"), resolve_baseline("B1")
    bindings = _b0_bindings()
    validate_capability_bindings(b0, bindings, ExecutionStatus.READY)
    with pytest.raises(TypeError):
        validate_capability_bindings(b0, list(bindings), ExecutionStatus.READY)
    with pytest.raises(ValueError):
        validate_capability_bindings(b0, (), ExecutionStatus.READY)
    with pytest.raises(ValueError):
        validate_capability_bindings(b0, bindings[:1], ExecutionStatus.READY)
    with pytest.raises(ValueError):
        validate_capability_bindings(b0, bindings[:-1], ExecutionStatus.READY)
    with pytest.raises(ValueError):
        validate_capability_bindings(
            b0, (bindings[1], bindings[0], *bindings[2:]), ExecutionStatus.READY
        )
    with pytest.raises(ValueError):
        validate_capability_bindings(
            b0, (bindings[0], bindings[0], *bindings[2:]), ExecutionStatus.READY
        )
    duplicate_name = replace(bindings[1], name=bindings[0].name)
    duplicate_provider = replace(bindings[1], provider=bindings[0].provider)
    with pytest.raises(ValueError):
        validate_capability_bindings(
            b0, (bindings[0], duplicate_name, *bindings[2:]), ExecutionStatus.READY
        )
    with pytest.raises(ValueError):
        validate_capability_bindings(
            b0, (bindings[0], duplicate_provider, *bindings[2:]), ExecutionStatus.READY
        )
    extra = CapabilityBinding("m7.extra", "instinctlab.control.extra", "1" * 64)
    with pytest.raises(ValueError):
        validate_capability_bindings(b0, bindings + (extra,), ExecutionStatus.READY)
    wrong_name = replace(bindings[0], name="m5.unknown")
    wrong_provider = replace(bindings[1], provider="instinctlab.control.command_adapter.alias")
    wrong_hash = replace(bindings[2], evidence_sha256="0" * 64)
    for candidate in (wrong_name, wrong_provider, wrong_hash):
        index = (wrong_name, wrong_provider, wrong_hash).index(candidate)
        mutated = bindings[:index] + (candidate,) + bindings[index + 1:]
        with pytest.raises(ValueError):
            validate_capability_bindings(b0, mutated, ExecutionStatus.READY)
    with pytest.raises(ValueError):
        validate_capability_bindings(b1, bindings, ExecutionStatus.READY)
    with pytest.raises(ValueError):
        validate_capability_bindings(b1, bindings, ExecutionStatus.NOT_READY)


def test_ready_executed_not_ready_and_b8_skipped_hand_fixtures() -> None:
    ready = _manifest("B0", ExecutionStatus.READY)
    executed = _manifest("B0", ExecutionStatus.EXECUTED)
    not_ready = _manifest("B1", ExecutionStatus.NOT_READY)
    skipped = _manifest("B8", ExecutionStatus.SKIPPED)
    expected_hashes = (
        "489287a2ac180dfb8ff525e859498fdfa9801a49cdc9809499cc0170f93d6178",
        "9cbb1b442f615b685540e42692900e8192c9256c500ca11d9b10f752f832b69a",
        "7b685595dc93cfd585e8a1c8ba09f8e2378c354c0d9a5007f98a7960c69ab0dd",
        "ef82d1209605dabc9c5fc289e368bb32d7c9a4107672d3cb5867ef812cdbb8b9",
    )
    assert tuple(item.manifest_sha256 for item in (ready, executed, not_ready, skipped)) == (
        expected_hashes
    )
    for record, baseline_id, status in zip(
        (ready, executed, not_ready, skipped),
        ("B0", "B0", "B1", "B8"),
        ("READY", "EXECUTED", "NOT_READY", "SKIPPED"),
        strict=True,
    ):
        mapping = record.to_mapping()
        assert tuple(mapping) == MANIFEST_FIELD_ORDER
        assert mapping["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert mapping["product_commit"] == BASE_COMMIT
        assert mapping["platform"] == "TASLAB_UAV" and mapping["sensor"] == "Livox MID360"
        assert mapping["baseline_id"] == baseline_id and mapping["execution_status"] == status
        assert mapping["scenario_id"] == "S1.corner" and mapping["seed"] == 7
        assert hashlib.sha256(
            json.dumps(
                OrderedDict((key, mapping[key]) for key in MANIFEST_PAYLOAD_FIELD_ORDER),
                ensure_ascii=True, allow_nan=False, separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest() == mapping["manifest_sha256"]
    assert ready.to_mapping()["actor_audit_status"] == "PASS"
    assert not_ready.to_mapping()["capability_bindings"] == []
    assert skipped.to_mapping()["skip_reason"] == "capability_unavailable"


def test_manifest_bytes_round_trip_and_returned_mapping_mutation_isolated() -> None:
    record = _manifest()
    original = record.canonical_json
    decoded = record.to_mapping()
    decoded["baseline_id"] = "B8"
    decoded["actor_input_fields"].append("root_state")
    decoded["component_toggles"]["history"] = True
    decoded["capability_bindings"][0]["provider"] = "mutated"
    assert record.canonical_json == original
    assert BaselineRunManifest.from_json(original).canonical_json == original
    assert BaselineRunManifest.from_json(original).to_bytes() == original.encode("ascii")
    with pytest.raises(FrozenInstanceError):
        record.canonical_json = "changed"


@pytest.mark.parametrize(
    ("root_key", "nested_key", "value"),
    (
        ("deployable", None, 1),
        ("component_toggles", "history", 0),
        ("component_toggles", "imu", 0),
        ("component_toggles", "anchor", 0),
        ("component_toggles", "ics", 0),
        ("deployable", None, 1.0),
        ("component_toggles", "history", "false"),
        ("component_toggles", "imu", None),
        ("component_toggles", "anchor", [False]),
        ("component_toggles", "ics", {"value": False}),
    ),
)
def test_manifest_boolean_positions_require_exact_json_booleans(
    root_key: str,
    nested_key: str | None,
    value: object,
) -> None:
    mapping = OrderedDict(_manifest().to_mapping())
    if nested_key is None:
        mapping[root_key] = value
    else:
        toggles = mapping[root_key]
        assert isinstance(toggles, dict)
        toggles[nested_key] = value
    with pytest.raises(TypeError, match="exact JSON boolean"):
        BaselineRunManifest.from_json(_resign(mapping))


def test_manifest_exact_json_booleans_still_pass() -> None:
    for record in (
        _manifest("B0", ExecutionStatus.READY),
        _manifest("B0", ExecutionStatus.EXECUTED),
        _manifest("B1", ExecutionStatus.NOT_READY),
        _manifest("B8", ExecutionStatus.SKIPPED),
    ):
        mapping = record.to_mapping()
        assert type(mapping["deployable"]) is bool
        assert all(type(value) is bool for value in mapping["component_toggles"].values())
        assert BaselineRunManifest.from_json(record.canonical_json).canonical_json == (
            record.canonical_json
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing", "extra", "reordered", "id_switch_mismatch", "privileged_deployable",
        "actor_schema", "evaluation_schema", "component_missing", "component_extra",
        "component_reordered", "capability_missing_key", "capability_extra_key",
        "capability_reordered", "wrong_hash", "audit_not_pass", "negative_seed",
        "float_seed", "control_scenario", "unknown_status", "skip_reason_on_ready",
    ),
)
def test_manifest_semantic_and_order_mutations_fail_closed(mutation: str) -> None:
    mapping = OrderedDict(_manifest().to_mapping())
    if mutation == "missing":
        mapping.pop("sensor")
    elif mutation == "extra":
        mapping["extra"] = False
    elif mutation == "reordered":
        mapping.move_to_end("schema_version")
    elif mutation == "id_switch_mismatch":
        mapping["config_switch"] = "no_history"
    elif mutation == "privileged_deployable":
        mapping["privilege_class"] = PrivilegeClass.PRIVILEGED_UPPER_BOUND.value
    elif mutation == "actor_schema":
        mapping["actor_input_fields"] = ["commanded_body_velocity_m_s", "root_state"]
    elif mutation == "evaluation_schema":
        mapping["evaluation_only_fields"] = mapping["evaluation_only_fields"][:-1]
    elif mutation == "component_missing":
        mapping["component_toggles"].pop("ics")
    elif mutation == "component_extra":
        mapping["component_toggles"]["extra"] = False
    elif mutation == "component_reordered":
        mapping["component_toggles"] = OrderedDict(
            reversed(tuple(mapping["component_toggles"].items()))
        )
    elif mutation == "capability_missing_key":
        mapping["capability_bindings"][0].pop("provider")
    elif mutation == "capability_extra_key":
        mapping["capability_bindings"][0]["extra"] = "x"
    elif mutation == "capability_reordered":
        item = mapping["capability_bindings"][0]
        mapping["capability_bindings"][0] = OrderedDict(reversed(tuple(item.items())))
    elif mutation == "wrong_hash":
        mapping["capability_bindings"][0]["evidence_sha256"] = "0" * 64
    elif mutation == "audit_not_pass":
        mapping["actor_audit_status"] = "NOT_RUN"
    elif mutation == "negative_seed":
        mapping["seed"] = -1
    elif mutation == "float_seed":
        mapping["seed"] = float("inf")
    elif mutation == "control_scenario":
        mapping["scenario_id"] = "bad\nscenario"
    elif mutation == "unknown_status":
        mapping["execution_status"] = "SUCCESS"
    elif mutation == "skip_reason_on_ready":
        mapping["skip_reason"] = "fallback"
    with pytest.raises((KeyError, TypeError, ValueError)):
        BaselineRunManifest.from_json(_resign(mapping))


def test_manifest_builder_rejects_incomplete_unsupported_and_hidden_fallback_claims() -> None:
    base = dict(
        product_commit=BASE_COMMIT,
        baseline_id="B0",
        config_switch="direct_velocity",
        capability_bindings=_b0_bindings(),
        scenario_id="S1.corner",
        seed=0,
        actor_audit_status=ActorAuditStatus.PASS,
        execution_status=ExecutionStatus.READY,
        skip_reason=None,
    )
    with pytest.raises(TypeError):
        build_run_manifest(**{key: value for key, value in base.items() if key != "seed"})
    with pytest.raises(TypeError):
        build_run_manifest(**base, extra=True)
    for baseline_id in ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"):
        row = resolve_baseline(baseline_id)
        candidate = dict(base, baseline_id=baseline_id, config_switch=row.config_switch)
        with pytest.raises(ValueError):
            build_run_manifest(**candidate)
    mismatch = dict(base, baseline_id="B1", config_switch="direct_velocity")
    with pytest.raises(ValueError):
        build_run_manifest(**mismatch)
    unavailable_b8 = dict(
        base,
        baseline_id="B8",
        config_switch="simplified_cbf_vo_optional",
        capability_bindings=(),
        actor_audit_status=ActorAuditStatus.NOT_RUN,
        execution_status=ExecutionStatus.NOT_READY,
    )
    with pytest.raises(ValueError):
        build_run_manifest(**unavailable_b8)


def test_noncanonical_duplicate_nonfinite_and_hash_tampering_reject() -> None:
    canonical = _manifest().canonical_json
    with pytest.raises(ValueError):
        BaselineRunManifest.from_json(canonical + "\n")
    with pytest.raises(ValueError):
        BaselineRunManifest.from_json(canonical.replace('"seed":7', '"seed":NaN'))
    with pytest.raises(ValueError):
        BaselineRunManifest.from_json(canonical.replace('"schema_version":', '"seed":7,"schema_version":'))
    mapping = OrderedDict(_manifest().to_mapping())
    mapping["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        BaselineRunManifest.from_json(
            json.dumps(mapping, ensure_ascii=True, separators=(",", ":"))
        )


def test_module_is_standard_library_only_and_reverse_import_isolation_holds() -> None:
    baseline_path = ROOT / "instinctlab/evaluation/baselines.py"
    tree = ast.parse(baseline_path.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(item.name.split(".")[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imports.add((node.module or "").split(".")[0])
    assert imports <= {
        "__future__", "collections", "dataclasses", "enum", "hashlib", "json", "re", "typing",
    }
    reverse_roots = (
        ROOT / "instinctlab/observations",
        ROOT / "instinctlab/control",
        ROOT / "instinctlab/envs",
        ROOT / "instinctlab/task",
        ROOT / "instinctlab/training",
    )
    reverse_sources = [path for root in reverse_roots for path in root.rglob("*.py")]
    reverse_sources.extend(
        ROOT / f"instinctlab/evaluation/{name}.py"
        for name in ("audit", "metrics", "observability", "runtime", "raycaster_observability")
    )
    for path in reverse_sources:
        source = path.read_text()
        assert "evaluation.baselines" not in source and "from .baselines" not in source


def test_accepted_actor_and_source_manifest_files_remain_byte_identical() -> None:
    expected = {
        "instinctlab/observations/actor.py": "dfc339233e146e585493bfcc1893b5ee87c46c827963e36d6f9ec79e4b81ce44",
        "instinctlab/control/governor.py": B0_GOVERNOR_SHA256,
        "instinctlab/control/command_adapter.py": B0_ROTATION_SHA256,
        "instinctlab/control/baseline_b0.py": B0_ADAPTER_SHA256,
        "instinctlab/evaluation/baseline_execution.py": B0_EXECUTION_SHA256,
        "instinctlab/evaluation/baseline_controller_execution.py": B0_CONTROLLER_SHA256,
        "instinctlab/evaluation/baseline_run.py": B0_RUN_SHA256,
        "instinctlab/evaluation/baseline_environment.py": B0_ENVIRONMENT_SHA256,
        "instinctlab/evaluation/metrics.py": "a37cf88a04d55ac4e32fee93f5504a6ff446f8804694a310013fb704438a3960",
        "instinctlab/evaluation/runtime.py": "7abaa594a63b0a90d9a0bb5b1ec66a9f71d3c91aa464f2890001a2040384cde3",
    }
    assert {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in expected
    } == expected
