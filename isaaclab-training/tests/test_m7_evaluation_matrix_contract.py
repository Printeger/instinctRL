"""Pure exhaustive contract tests for the M7 availability matrix."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import FrozenInstanceError, fields, replace
import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from instinctlab.evaluation import (
    ACTOR_AUDIT_REQUIREMENT,
    BASELINE_ORDER,
    BASELINE_SOURCE_SHA256,
    CELL_FIELD_ORDER,
    MATRIX_FIELD_ORDER,
    MATRIX_SCHEMA_VERSION,
    SCENARIO_ORDER,
    SCENARIO_SOURCE_SHA256,
    STATUS_ORDER,
    ActorAuditRequirement,
    EvaluationAvailabilityCell,
    EvaluationAvailabilityMatrix,
    MatrixStatus,
    build_evaluation_availability_matrix,
    decode_evaluation_availability_matrix,
    encode_evaluation_availability_matrix,
    validate_evaluation_availability_matrix,
)
from instinctlab.evaluation.baselines import BASELINE_REGISTRY, PrivilegeClass
from instinctlab.evaluation.scenarios import RequirementClass, scenario_registry


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "instinctlab/evaluation/matrix.py"
PRODUCT_COMMIT = "39206507367e0fdd65ed99b6f3113dd5901d43bc"


def _matrix() -> EvaluationAvailabilityMatrix:
    return build_evaluation_availability_matrix(product_commit=PRODUCT_COMMIT)


def _mapping() -> OrderedDict[str, object]:
    return json.loads(
        encode_evaluation_availability_matrix(_matrix()),
        object_pairs_hook=OrderedDict,
    )


def _canonical(mapping: object) -> str:
    return json.dumps(mapping, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _resign(mapping: OrderedDict[str, object]) -> str:
    payload = OrderedDict((key, value) for key, value in mapping.items() if key != "matrix_sha256")
    mapping["matrix_sha256"] = hashlib.sha256(_canonical(payload).encode("ascii")).hexdigest()
    return _canonical(mapping)


def _cell(matrix: EvaluationAvailabilityMatrix, baseline: str, scenario: str) -> EvaluationAvailabilityCell:
    return next(cell for cell in matrix.cells if cell.baseline_id == baseline and cell.scenario_id == scenario)


def test_exact_public_api_signatures_types_and_field_order() -> None:
    assert MATRIX_SCHEMA_VERSION == "m7.evaluation-availability-matrix.v1"
    assert ACTOR_AUDIT_REQUIREMENT == "PASS_BEFORE_EXECUTION"
    assert tuple(item.value for item in MatrixStatus) == ("READY", "NOT_READY", "SKIPPED")
    assert STATUS_ORDER == ("READY", "NOT_READY", "SKIPPED")
    assert tuple(item.value for item in ActorAuditRequirement) == ("PASS_BEFORE_EXECUTION",)
    assert BASELINE_ORDER == tuple(f"B{index}" for index in range(9))
    assert SCENARIO_ORDER == (
        "corner", "flat_wall", "corridor_tunnel", "open_field", "cluttered_indoor",
        "unsafe_command_toward_wall", "gap_doorway", "dynamic_obstacle",
        "real_robot_observable_holding", "real_robot_unsafe_command_approach",
    )
    assert CELL_FIELD_ORDER == tuple(item.name for item in fields(EvaluationAvailabilityCell)) == (
        "baseline_id", "config_switch", "execution_family", "privilege_class", "deployable",
        "baseline_optional", "baseline_readiness", "scenario_id", "execution_domain",
        "requirement_class", "scenario_readiness", "scenario_seed_or_null",
        "scenario_provider_or_null", "scenario_evidence_sha256_or_null", "status",
        "required_blockers", "optional_blockers", "actor_audit_requirement",
        "execution_performed",
    )
    assert MATRIX_FIELD_ORDER == tuple(item.name for item in fields(EvaluationAvailabilityMatrix))
    signature = inspect.signature(build_evaluation_availability_matrix)
    assert tuple(signature.parameters) == ("product_commit",)
    assert signature.parameters["product_commit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["product_commit"].default is inspect.Parameter.empty
    for function in (
        encode_evaluation_availability_matrix,
        decode_evaluation_availability_matrix,
        validate_evaluation_availability_matrix,
    ):
        assert len(inspect.signature(function).parameters) == 1
    assert EvaluationAvailabilityCell.__slots__
    assert EvaluationAvailabilityMatrix.__slots__
    with pytest.raises(TypeError):
        type("CellChild", (EvaluationAvailabilityCell,), {})
    with pytest.raises(TypeError):
        type("MatrixChild", (EvaluationAvailabilityMatrix,), {})


def test_exact_ninety_cell_order_counts_and_source_identities() -> None:
    matrix = _matrix()
    assert matrix.schema_version == MATRIX_SCHEMA_VERSION
    assert matrix.product_commit == PRODUCT_COMMIT
    assert matrix.baseline_source_schema_version == "m7.baseline-run-manifest.v1"
    assert matrix.scenario_source_schema_version == "m7.scenario-catalog.v1"
    assert matrix.baseline_source_sha256 == BASELINE_SOURCE_SHA256
    assert matrix.scenario_source_sha256 == SCENARIO_SOURCE_SHA256
    assert len(BASELINE_SOURCE_SHA256) == len(SCENARIO_SOURCE_SHA256) == 64
    assert matrix.baseline_order == BASELINE_ORDER
    assert matrix.scenario_order == SCENARIO_ORDER
    assert matrix.status_order == STATUS_ORDER
    assert (matrix.cell_count, matrix.ready_count, matrix.not_ready_count, matrix.skipped_count) == (90, 7, 74, 9)
    assert tuple((cell.baseline_id, cell.scenario_id) for cell in matrix.cells) == tuple(
        (baseline, scenario) for baseline in BASELINE_ORDER for scenario in SCENARIO_ORDER
    )
    assert len({(cell.baseline_id, cell.scenario_id) for cell in matrix.cells}) == 90
    assert all(cell.execution_performed is False for cell in matrix.cells)
    assert all(cell.actor_audit_requirement is ActorAuditRequirement.PASS_BEFORE_EXECUTION for cell in matrix.cells)


def test_every_cell_is_exactly_derived_from_both_accepted_registries() -> None:
    matrix = _matrix()
    for cell, baseline, scenario in zip(
        matrix.cells,
        (baseline for baseline in BASELINE_REGISTRY for _ in scenario_registry()),
        (scenario for _ in BASELINE_REGISTRY for scenario in scenario_registry()),
        strict=True,
    ):
        assert (
            cell.baseline_id, cell.config_switch, cell.execution_family,
            cell.privilege_class, cell.deployable, cell.baseline_optional,
            cell.baseline_readiness,
        ) == (
            baseline.baseline_id.value, baseline.config_switch, baseline.execution_family.value,
            baseline.privilege_class.value, baseline.deployable, baseline.optional,
            baseline.readiness.value,
        )
        assert (
            cell.scenario_id, cell.execution_domain, cell.requirement_class,
            cell.scenario_readiness, cell.scenario_provider_or_null,
            cell.scenario_evidence_sha256_or_null,
        ) == (
            scenario.scenario_id.value, scenario.execution_domain.value,
            scenario.requirement_class.value, scenario.evidence.readiness.value,
            scenario.evidence.provider, scenario.evidence.evidence_sha256,
        )
        unavailable = tuple(
            requirement.name for requirement in baseline.capability_requirements
            if requirement.accepted_provider is None
        )
        required = () if baseline.optional else unavailable
        optional = unavailable if baseline.optional else ()
        if scenario.evidence.unavailability_reason is not None:
            if scenario.requirement_class is RequirementClass.REQUIRED:
                required += (scenario.evidence.unavailability_reason,)
            else:
                optional += (scenario.evidence.unavailability_reason,)
        assert cell.required_blockers == required
        assert cell.optional_blockers == optional
        expected = MatrixStatus.NOT_READY if required else MatrixStatus.SKIPPED if optional else MatrixStatus.READY
        assert cell.status is expected


def test_b0_b1_b8_dynamic_and_real_robot_truth() -> None:
    matrix = _matrix()
    assert tuple(cell.scenario_id for cell in matrix.cells if cell.status is MatrixStatus.READY) == SCENARIO_ORDER[:7]
    assert all(cell.baseline_id == "B0" for cell in matrix.cells if cell.status is MatrixStatus.READY)
    assert _cell(matrix, "B0", "dynamic_obstacle").status is MatrixStatus.SKIPPED
    assert _cell(matrix, "B0", "dynamic_obstacle").required_blockers == ()
    b1 = [cell for cell in matrix.cells if cell.baseline_id == "B1"]
    assert all(not cell.deployable for cell in b1)
    assert all(cell.privilege_class == PrivilegeClass.PRIVILEGED_UPPER_BOUND.value for cell in b1)
    assert all(cell.status is MatrixStatus.NOT_READY for cell in b1)
    assert all(
        cell.deployable and cell.privilege_class == PrivilegeClass.ACTOR_CLEAN.value
        for cell in matrix.cells if cell.baseline_id != "B1"
    )
    assert all(_cell(matrix, baseline, scenario).status is MatrixStatus.NOT_READY for baseline in BASELINE_ORDER[1:8] for scenario in SCENARIO_ORDER)
    assert all(_cell(matrix, "B8", scenario).status is MatrixStatus.SKIPPED for scenario in SCENARIO_ORDER[:8])
    for scenario in SCENARIO_ORDER[8:]:
        b8 = _cell(matrix, "B8", scenario)
        assert b8.status is MatrixStatus.NOT_READY
        assert b8.required_blockers and b8.optional_blockers == ("m7.b8_simplified_cbf_vo_filter",)


def test_canonical_ascii_hash_round_trip_stability_and_mutation_isolation() -> None:
    first = _matrix()
    second = _matrix()
    encoded = encode_evaluation_availability_matrix(first)
    decoded = decode_evaluation_availability_matrix(encoded)
    assert first == second == decoded
    assert encoded == encode_evaluation_availability_matrix(decoded)
    assert encoded.encode("ascii").decode("ascii") == encoded
    mapping = json.loads(encoded)
    payload = OrderedDict((key, mapping[key]) for key in MATRIX_FIELD_ORDER[:-1])
    assert hashlib.sha256(_canonical(payload).encode("ascii")).hexdigest() == first.matrix_sha256
    mapping["cells"][0]["required_blockers"].append("caller_mutation")
    assert decoded.cells[0].required_blockers == ()
    with pytest.raises(FrozenInstanceError):
        decoded.cell_count = 0
    with pytest.raises(FrozenInstanceError):
        decoded.cells[0].status = MatrixStatus.NOT_READY


@pytest.mark.parametrize("commit", ["", "A" * 40, "0" * 39, "0" * 41, "  " + "0" * 40, None, 7])
def test_builder_rejects_nonexact_product_commit(commit: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_evaluation_availability_matrix(product_commit=commit)


@pytest.mark.parametrize(
    ("mutation", "resign"),
    (
        (lambda root: root.pop("cell_count"), False),
        (lambda root: root.update((('extra', 1),)), False),
        (lambda root: root.__setitem__("ready_count", 8), True),
        (lambda root: root.__setitem__("matrix_sha256", "0" * 64), False),
        (lambda root: root["cells"].pop(), True),
        (lambda root: root["cells"].append(root["cells"][0]), True),
        (lambda root: root["cells"].__setitem__(slice(0, 2), reversed(root["cells"][:2])), True),
        (lambda root: root["cells"][0].__setitem__("baseline_id", "b0"), True),
        (lambda root: root["cells"][0].__setitem__("scenario_id", " corner"), True),
        (lambda root: root["cells"][0].__setitem__("status", "EXECUTED"), True),
        (lambda root: root["cells"][10].__setitem__("status", "READY"), True),
        (lambda root: root["cells"][0].__setitem__("execution_performed", True), True),
        (lambda root: root["cells"][0].__setitem__("actor_audit_requirement", "PASS"), True),
        (lambda root: root["cells"][10].__setitem__("deployable", True), True),
        (lambda root: root["cells"][10].__setitem__("privilege_class", "actor_clean"), True),
        (lambda root: root["cells"][10].__setitem__("required_blockers", []), True),
        (lambda root: root["cells"][0].__setitem__("scenario_seed_or_null", 1), True),
        (lambda root: root["cells"][0].__setitem__("scenario_provider_or_null", "wrong.provider"), True),
        (lambda root: root["cells"][0].__setitem__("scenario_evidence_sha256_or_null", "0" * 64), True),
    ),
)
def test_decoder_rejects_missing_extra_reordered_duplicate_or_false_claims(mutation, resign: bool) -> None:
    mapping = _mapping()
    mutation(mapping)
    candidate = _resign(mapping) if resign else _canonical(mapping)
    with pytest.raises((TypeError, ValueError)):
        decode_evaluation_availability_matrix(candidate)


def test_decoder_rejects_noncanonical_duplicate_nonfinite_and_mutable_inputs() -> None:
    encoded = encode_evaluation_availability_matrix(_matrix())
    with pytest.raises(ValueError):
        decode_evaluation_availability_matrix(encoded + "\n")
    with pytest.raises(ValueError):
        decode_evaluation_availability_matrix(encoded.replace('{"schema_version"', '{"schema_version":"x","schema_version"', 1))
    with pytest.raises(ValueError):
        decode_evaluation_availability_matrix(encoded.replace('"cell_count":90', '"cell_count":NaN'))
    with pytest.raises(TypeError):
        decode_evaluation_availability_matrix(_mapping())
    with pytest.raises(TypeError):
        validate_evaluation_availability_matrix({})
    with pytest.raises(ValueError):
        validate_evaluation_availability_matrix(replace(_matrix(), ready_count=8, matrix_sha256="0" * 64))
    root_reordered = _mapping()
    root_reordered.move_to_end("schema_version")
    with pytest.raises(ValueError):
        decode_evaluation_availability_matrix(_canonical(root_reordered))
    cell_reordered = _mapping()
    cell_reordered["cells"][0].move_to_end("baseline_id")
    with pytest.raises(ValueError):
        decode_evaluation_availability_matrix(_resign(cell_reordered))


def test_matrix_imports_only_accepted_pure_contracts_and_has_no_reverse_runtime_imports() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    relative = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level]
    assert {(node.module, tuple(alias.name for alias in node.names)) for node in relative} == {
        (None, ("baselines",)), (None, ("scenarios",)),
    }
    forbidden = ("actor", "control", "environment", "task", "training", "runtime")
    imported = tuple(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    )
    assert not any(any(word in name for word in forbidden) for name in imported)
    reverse_importers = []
    for path in (ROOT / "instinctlab").rglob("*.py"):
        if path == SOURCE or path.name == "__init__.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "evaluation.matrix" in source or " import matrix" in source:
            reverse_importers.append(path.relative_to(ROOT).as_posix())
    assert reverse_importers == []
