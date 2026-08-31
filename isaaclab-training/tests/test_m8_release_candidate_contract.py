"""Exhaustive pure tests for the immutable M8 release-candidate contract."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import FrozenInstanceError, fields, replace
import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import instinctlab.qualification as qualification
from instinctlab.qualification import (
    ACCEPTED_MILESTONES,
    ACTOR_BOUNDARY,
    ACTOR_BOUNDARY_FIELD_ORDER,
    AVAILABILITY_FIELD_ORDER,
    AVAILABILITY_SUMMARY,
    FORMAL_SOURCE_COMMIT,
    FROZEN_BASELINE,
    GATE_FIELD_ORDER,
    MANIFEST_FIELD_ORDER,
    MANIFEST_PAYLOAD_FIELD_ORDER,
    MILESTONE_FIELD_ORDER,
    MILESTONE_ORDER,
    M7_RELEASE_INPUT_COMMIT,
    PLATFORM,
    QUALIFICATION_GATES,
    RUNTIME_FIELD_ORDER,
    RUNTIME_IDENTITY,
    SCHEMA_VERSION,
    SENSOR,
    ActorBoundary,
    AvailabilitySummary,
    EvidenceClass,
    GateClassification,
    MilestoneAcceptance,
    QualificationGate,
    ReleaseCandidateManifest,
    ReleaseStatus,
    RuntimeIdentity,
    build_release_candidate,
    decode_release_candidate,
    encode_release_candidate,
    validate_release_candidate,
)


CANDIDATE = "f" * 40
EXPECTED_API = (
    "ACCEPTED_MILESTONES", "ACTOR_BOUNDARY", "ACTOR_BOUNDARY_FIELD_ORDER",
    "AVAILABILITY_FIELD_ORDER", "AVAILABILITY_SUMMARY", "ActorBoundary",
    "AvailabilitySummary", "EvidenceClass", "FORMAL_SOURCE_COMMIT", "FROZEN_BASELINE",
    "GATE_FIELD_ORDER", "GateClassification", "MANIFEST_FIELD_ORDER",
    "MANIFEST_PAYLOAD_FIELD_ORDER", "MILESTONE_FIELD_ORDER", "MILESTONE_ORDER",
    "M7_RELEASE_INPUT_COMMIT", "MilestoneAcceptance", "PLATFORM", "QUALIFICATION_GATES",
    "QualificationGate", "RUNTIME_FIELD_ORDER", "RUNTIME_IDENTITY", "ReleaseCandidateManifest",
    "ReleaseStatus", "RuntimeIdentity", "SCHEMA_VERSION", "SENSOR", "build_release_candidate",
    "decode_release_candidate", "encode_release_candidate", "validate_release_candidate",
)
EXPECTED_CHAIN = (
    ("M1A", "M1A-Q-001", 0, "M1A-001-Q0005", "6eb521f2174a64030e30ea5e8ce8db1f89006619", "commander/reviews/M1A-Q-001_REVIEW.md", "773ea350eaa6925fed0fad0851a10fbda410491cf217b3ce0e40513c05af29cf", ("STATIC",)),
    ("M1B", "M1B-001", 3, "M1B-001-D0004", "0f4345e979707f62d9e5ae8ca399bd0029f005b3", "commander/reviews/M1B-001_D0004_APPROVAL.md", "ac8d111f860ecb157083e335bf37a6830719d9b3b3ab7f9a6e2e97418830505a", ("STATIC", "PURE")),
    ("M2", "M2-001", 0, "M2-001-D0001", "ecdb482bb05c08cfad141fb290a3d6c2e96298de", "commander/reviews/M2-001_D0001_APPROVAL.md", "48f0da97c532a37ced1ea80a7edac933e00a44f5081345bec86702fbea8eb333", ("STATIC", "PURE", "BOUNDED_CUDA", "BOUNDED_SIMULATOR")),
    ("M3", "M3-001", 1, "M3-001-D0002", "99ec3b0099b887a31962b2a5c862e3e73efc6872", "commander/reviews/M3-001_D0002_APPROVAL.md", "b8749d36c7696c0ca54e8348afa8ef3fcc6278300685334b79f944ede37dac82", ("STATIC", "PURE", "BOUNDED_CUDA", "BOUNDED_SIMULATOR")),
    ("M4", "M4-001", 1, "M4-001-D0002", "bff6f853eb6d95b12d4de7b95cbfa5a7f2283bbc", "commander/reviews/M4-001_D0002_APPROVAL.md", "8449d053d06ccc9e4b98e7a55734e6fb7b3217bf2c7c8a73e688f253c6c1617c", ("STATIC", "PURE", "BOUNDED_CUDA")),
    ("M5", "M5-002", 1, "M5-002-D0002", "acda9b1b30ca8aceaf6cf6eaca9d2c1d4bdaaa73", "commander/reviews/M5-002_D0002_APPROVAL.md", "93a1a45b524eb702521251d7fba6e385ae93948d27bd839f2a2c2e182c762aba", ("STATIC", "PURE", "BOUNDED_CUDA", "BOUNDED_SIMULATOR")),
    ("M6", "M6-004", 0, "M6-004-D0001", "033514d8fa7d8d2bb2ece5faa1521d528e32b0a4", "commander/reviews/M6-004_D0001_APPROVAL.md", "7fbb7c919081a8614995d1a95cf4976cd1c777b826139bc8c037f4ee47e0db6a", ("STATIC", "PURE", "BOUNDED_CUDA", "BOUNDED_SIMULATOR")),
    ("M7", "M7-021", 1, "M7-021-D0002", M7_RELEASE_INPUT_COMMIT, "commander/reviews/M7-021_D0002_APPROVAL.md", "9bac59948573fe24fdd073ccabf010b7914a7173f45cd7a250f5346ffd8a3a7f", ("STATIC", "PURE", "BOUNDED_CUDA")),
)
EXPECTED_GATE_CLASSES = (
    ("accepted_milestone_chain_integrity", "EVIDENCE_BOUND"),
    ("frozen_runtime_identity", "EVIDENCE_BOUND"),
    ("release_candidate_tree", "REQUIRES_BOUNDED_EXECUTION"),
    ("platform_sensor_contract", "EVIDENCE_BOUND"),
    ("actor_input_audit", "EVIDENCE_BOUND"),
    ("privilege_isolation", "EVIDENCE_BOUND"),
    ("bounded_pipeline", "REQUIRES_BOUNDED_EXECUTION"),
    ("b0_scenario_consistency", "REQUIRES_BOUNDED_EXECUTION"),
    ("b1_b7_required_unavailable", "REQUIRES_BOUNDED_EXECUTION"),
    ("b8_dynamic_optional_unavailable", "UNAVAILABLE_OPTIONAL"),
    ("real_robot_required_unavailable", "REQUIRES_BOUNDED_EXECUTION"),
    ("release_evidence_completeness", "REQUIRES_BOUNDED_EXECUTION"),
    ("user_authorization", "USER_AUTHORIZATION_REQUIRED"),
)


def _mapping() -> OrderedDict[str, object]:
    return json.loads(encode_release_candidate(build_release_candidate(CANDIDATE)), object_pairs_hook=OrderedDict)


def _encoded(mapping: OrderedDict[str, object]) -> bytes:
    return json.dumps(mapping, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


def _resign(mapping: OrderedDict[str, object]) -> bytes:
    payload = OrderedDict((key, value) for key, value in mapping.items() if key != "manifest_sha256")
    mapping["manifest_sha256"] = hashlib.sha256(_encoded(payload)).hexdigest()
    return _encoded(mapping)


def test_exact_public_api_signatures_and_field_order():
    assert tuple(qualification.__all__) == EXPECTED_API
    assert tuple(inspect.signature(build_release_candidate).parameters) == ("candidate_commit",)
    assert tuple(inspect.signature(validate_release_candidate).parameters) == ("manifest",)
    assert tuple(inspect.signature(encode_release_candidate).parameters) == ("manifest",)
    assert tuple(inspect.signature(decode_release_candidate).parameters) == ("encoded",)
    assert MILESTONE_FIELD_ORDER == tuple(field.name for field in fields(MilestoneAcceptance))
    assert RUNTIME_FIELD_ORDER == tuple(field.name for field in fields(RuntimeIdentity))
    assert ACTOR_BOUNDARY_FIELD_ORDER == tuple(field.name for field in fields(ActorBoundary))
    assert AVAILABILITY_FIELD_ORDER == tuple(field.name for field in fields(AvailabilitySummary))
    assert GATE_FIELD_ORDER == tuple(field.name for field in fields(QualificationGate))
    assert MANIFEST_FIELD_ORDER == tuple(field.name for field in fields(ReleaseCandidateManifest))
    assert MANIFEST_PAYLOAD_FIELD_ORDER == MANIFEST_FIELD_ORDER[:-1]


def test_exact_enum_order_and_no_aliases():
    assert tuple(item.value for item in EvidenceClass) == ("STATIC", "PURE", "BOUNDED_CUDA", "BOUNDED_SIMULATOR")
    assert tuple(item.value for item in GateClassification) == (
        "EVIDENCE_BOUND", "REQUIRES_BOUNDED_EXECUTION", "UNAVAILABLE_OPTIONAL", "USER_AUTHORIZATION_REQUIRED",
    )
    assert tuple(item.value for item in ReleaseStatus) == ("MANIFESTED_NOT_QUALIFIED",)
    for enum_type in (EvidenceClass, GateClassification, ReleaseStatus):
        with pytest.raises(ValueError):
            enum_type(enum_type.__name__)
        with pytest.raises(ValueError):
            enum_type(next(iter(enum_type)).value.lower())


def test_records_are_frozen_slotted_and_non_subclassable():
    manifest = build_release_candidate(CANDIDATE)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        manifest.platform = "other"  # type: ignore[misc]
    assert not hasattr(manifest, "__dict__")
    for record in (MilestoneAcceptance, RuntimeIdentity, ActorBoundary, AvailabilitySummary, QualificationGate, ReleaseCandidateManifest):
        with pytest.raises(TypeError):
            type("Forbidden", (record,), {})


def test_exact_accepted_chain_and_approval_hashes():
    actual = tuple((row.milestone_id, row.order_id, row.order_revision, row.accepted_attempt_id, row.product_commit, row.approval_record, row.approval_sha256, tuple(item.value for item in row.evidence_classes)) for row in ACCEPTED_MILESTONES)
    assert MILESTONE_ORDER == ("M1A", "M1B", "M2", "M3", "M4", "M5", "M6", "M7")
    assert actual == EXPECTED_CHAIN


def test_exact_frozen_runtime_platform_sensor_and_source_identity():
    assert (SCHEMA_VERSION, PLATFORM, SENSOR, FROZEN_BASELINE) == (
        "m8.blackwell-release-candidate.v1", "TASLAB_UAV", "Livox MID360", "BLACKWELL_BASELINE_V2_R580",
    )
    assert FORMAL_SOURCE_COMMIT == "6f6dee3da72d0b0202846b330adcaeeec8ee7bc3"
    assert M7_RELEASE_INPUT_COMMIT == "e2d46fc3ad909052182200b7cc958012000a454a"
    assert tuple(getattr(RUNTIME_IDENTITY, name) for name in RUNTIME_FIELD_ORDER) == (
        FROZEN_BASELINE, "580.173.02", "3.11.15", "2.7.0+cu128", "5.1.0", "2.3.2", "0.54.2", "0.10.0", "0.10.0",
        "5045c71cc73d224144cc08bc38d65cabeedaec8c", "3725bcc2e7c1be4ecf1455d922299ae85042603a", FORMAL_SOURCE_COMMIT,
    )


def test_actor_boundary_is_exact_and_fail_closed():
    assert ACTOR_BOUNDARY.low_level_stabilization_state_actor_visible is False
    assert "commanded velocity" in ACTOR_BOUNDARY.approved_input_classes
    assert "pose" in ACTOR_BOUNDARY.forbidden_input_classes
    assert "explicit or reconstructed velocity estimates" in ACTOR_BOUNDARY.forbidden_input_classes
    assert "labelled upper-bound baselines" in ACTOR_BOUNDARY.privileged_use_classes
    with pytest.raises(ValueError):
        replace(ACTOR_BOUNDARY, low_level_stabilization_state_actor_visible=True)


def test_exact_m7_availability_and_deferred_classes():
    assert (AVAILABILITY_SUMMARY.cell_count, AVAILABILITY_SUMMARY.ready_count, AVAILABILITY_SUMMARY.not_ready_count, AVAILABILITY_SUMMARY.skipped_count) == (90, 7, 74, 9)
    assert AVAILABILITY_SUMMARY.matrix_sha256 == "a0d1c1cdb8cedf18705a7ab4144a682b75ba259ef5e83809d85947546f540b65"
    assert AVAILABILITY_SUMMARY.execution_performed is False
    assert AVAILABILITY_SUMMARY.required_unavailable_baselines == ("B1", "B2", "B3", "B4", "B5", "B6", "B7")
    assert AVAILABILITY_SUMMARY.optional_unavailable_baselines == ("B8",)
    assert AVAILABILITY_SUMMARY.optional_unavailable_scenarios == ("dynamic_obstacle",)
    assert AVAILABILITY_SUMMARY.required_unavailable_scenarios == ("real_robot_observable_holding", "real_robot_unsafe_command_approach")


def test_exact_qualification_gate_order_and_classification():
    assert tuple((gate.gate_id, gate.classification.value) for gate in QUALIFICATION_GATES) == EXPECTED_GATE_CLASSES


def test_builder_requires_exact_lowercase_commit_and_keeps_m7_input():
    manifest = build_release_candidate(CANDIDATE)
    assert manifest.candidate_commit == CANDIDATE
    assert manifest.m7_release_input_commit == M7_RELEASE_INPUT_COMMIT
    for value in ("f" * 39, "f" * 41, "G" * 40, " " + CANDIDATE, CANDIDATE + " ", b"f" * 40, None):
        with pytest.raises((TypeError, ValueError)):
            build_release_candidate(value)  # type: ignore[arg-type]


def test_release_claims_are_always_false():
    manifest = build_release_candidate(CANDIDATE)
    assert manifest.release_status is ReleaseStatus.MANIFESTED_NOT_QUALIFIED
    assert (manifest.qualification_executed, manifest.release_approved, manifest.scale_training_authorized, manifest.merge_authorized, manifest.push_authorized) == (False,) * 5
    for name in ("qualification_executed", "release_approved", "scale_training_authorized", "merge_authorized", "push_authorized"):
        with pytest.raises(ValueError):
            replace(manifest, **{name: True})


def test_canonical_ascii_round_trip_and_hash_binding():
    manifest = build_release_candidate(CANDIDATE)
    encoded = encode_release_candidate(manifest)
    assert encoded.decode("ascii").encode("ascii") == encoded
    assert b'": ' not in encoded and b'", ' not in encoded and b"\n" not in encoded
    payload = OrderedDict((key, value) for key, value in _mapping().items() if key != "manifest_sha256")
    assert manifest.manifest_sha256 == hashlib.sha256(_encoded(payload)).hexdigest()
    assert decode_release_candidate(encoded) == manifest
    assert encode_release_candidate(decode_release_candidate(encoded)) == encoded


def test_caller_mutation_isolation():
    source = list(MILESTONE_ORDER)
    manifest = build_release_candidate(CANDIDATE)
    source.reverse()
    assert manifest.milestone_order == MILESTONE_ORDER
    mapping = _mapping()
    cast_order = mapping["milestone_order"]
    assert isinstance(cast_order, list)
    cast_order.reverse()
    assert encode_release_candidate(manifest) != _encoded(mapping)
    assert manifest.milestone_order == MILESTONE_ORDER


@pytest.mark.parametrize("candidate", ["missing", "extra", "reordered"])
def test_decoder_rejects_top_level_field_shape(candidate: str):
    mapping = _mapping()
    if candidate == "missing":
        mapping.pop("platform")
    elif candidate == "extra":
        mapping["unexpected"] = False
    else:
        mapping.move_to_end("schema_version")
    with pytest.raises(ValueError):
        decode_release_candidate(_encoded(mapping))


def test_decoder_rejects_duplicate_fields():
    encoded = encode_release_candidate(build_release_candidate(CANDIDATE))
    duplicate = encoded.replace(b'{"schema_version":', b'{"schema_version":"duplicate","schema_version":', 1)
    with pytest.raises(ValueError, match="canonical"):
        decode_release_candidate(duplicate)


@pytest.mark.parametrize("path,value", [
    (("release_status",), "QUALIFIED"),
    (("qualification_executed",), True),
    (("release_approved",), True),
    (("scale_training_authorized",), True),
    (("merge_authorized",), True),
    (("push_authorized",), True),
    (("m7_release_input_commit",), "0" * 40),
])
def test_decoder_rejects_resigned_false_semantic_claims(path: tuple[str, ...], value: object):
    mapping = _mapping()
    mapping[path[0]] = value
    with pytest.raises(ValueError):
        decode_release_candidate(_resign(mapping))


def test_decoder_rejects_resigned_reordered_milestones_and_gates():
    for key in ("accepted_milestones", "qualification_gates"):
        mapping = _mapping()
        rows = mapping[key]
        assert isinstance(rows, list)
        rows.reverse()
        with pytest.raises(ValueError):
            decode_release_candidate(_resign(mapping))


def test_decoder_rejects_nested_missing_extra_reordered_and_alias_fields():
    mutations = []
    missing = _mapping(); missing["accepted_milestones"][0].pop("order_id"); mutations.append(missing)  # type: ignore[index,union-attr]
    extra = _mapping(); extra["runtime_identity"]["extra"] = "x"; mutations.append(extra)  # type: ignore[index]
    reordered = _mapping(); reordered["qualification_gates"][0].move_to_end("gate_id"); mutations.append(reordered)  # type: ignore[index,union-attr]
    alias = _mapping(); alias["qualification_gates"][0]["classification"] = "evidence_bound"; mutations.append(alias)  # type: ignore[index]
    for mapping in mutations:
        with pytest.raises(ValueError):
            decode_release_candidate(_resign(mapping))


def test_decoder_rejects_noncanonical_and_nonfinite_json():
    encoded = encode_release_candidate(build_release_candidate(CANDIDATE))
    with pytest.raises(ValueError):
        decode_release_candidate(encoded + b"\n")
    mapping = _mapping(); mapping["candidate_commit"] = float("nan")
    nonfinite = json.dumps(mapping, ensure_ascii=True, allow_nan=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(ValueError):
        decode_release_candidate(nonfinite)
    with pytest.raises(ValueError):
        decode_release_candidate(b"\xff")
    with pytest.raises(TypeError):
        decode_release_candidate(encoded.decode("ascii"))  # type: ignore[arg-type]


def test_record_constructors_reject_malformed_commits_hashes_duplicates_and_counts():
    with pytest.raises(ValueError):
        replace(ACCEPTED_MILESTONES[0], product_commit="A" * 40)
    with pytest.raises(ValueError):
        replace(ACCEPTED_MILESTONES[0], approval_sha256="0" * 63)
    with pytest.raises(ValueError):
        replace(ACCEPTED_MILESTONES[0], evidence_classes=(EvidenceClass.STATIC, EvidenceClass.STATIC))
    with pytest.raises(ValueError):
        replace(AVAILABILITY_SUMMARY, ready_count=8)
    with pytest.raises(ValueError):
        replace(AVAILABILITY_SUMMARY, execution_performed=True)
    with pytest.raises(TypeError):
        replace(QUALIFICATION_GATES[0], classification="EVIDENCE_BOUND")


def test_validate_rederives_complete_contract():
    manifest = build_release_candidate(CANDIDATE)
    validate_release_candidate(manifest)
    for mutation in (
        replace(manifest, platform="OTHER"),
        replace(manifest, accepted_milestones=tuple(reversed(manifest.accepted_milestones))),
        replace(manifest, qualification_gates=tuple(reversed(manifest.qualification_gates))),
        replace(manifest, manifest_sha256="0" * 64),
    ):
        with pytest.raises(ValueError):
            validate_release_candidate(mutation)
    with pytest.raises(TypeError):
        validate_release_candidate(object())  # type: ignore[arg-type]


def test_import_graph_is_standard_library_only_and_has_no_runtime_reverse_import():
    root = Path(__file__).parents[1]
    module_path = root / "instinctlab" / "qualification" / "release_candidate.py"
    package_path = root / "instinctlab" / "qualification" / "__init__.py"
    allowed = {"__future__", "collections", "dataclasses", "enum", "hashlib", "json", "re", "typing"}
    for path in (module_path, package_path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] in allowed for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 1 or (node.module or "").split(".")[0] in allowed
    forbidden = {"actor", "control", "envs", "environment", "sensor", "task", "training", "evaluation", "torch", "omni", "isaac"}
    for path in (root / "instinctlab").rglob("*.py"):
        if "qualification" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "qualification" not in (node.module or "").split(".")
            elif isinstance(node, ast.Import):
                assert all("qualification" not in alias.name.split(".") for alias in node.names)
    module_source = module_path.read_text(encoding="utf-8").lower()
    assert not any(f"import {name}" in module_source or f"from {name}" in module_source for name in forbidden)


def test_manifest_contains_no_forbidden_moving_or_performance_fields():
    mapping = _mapping()
    text = encode_release_candidate(build_release_candidate(CANDIDATE)).decode("ascii").lower()
    assert tuple(mapping) == MANIFEST_FIELD_ORDER
    keys: set[str] = set()
    def collect(value: object) -> None:
        if isinstance(value, dict):
            keys.update(value)
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)
    collect(mapping)
    assert not {"timestamp", "moving_ref", "local_path", "checkpoint", "metric_value", "threshold", "performance"} & keys
    assert "refs/heads" not in text and "/home/" not in text
