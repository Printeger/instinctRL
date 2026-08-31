"""Pure immutable availability matrix derived from accepted M7 contracts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import re
from typing import Final

from . import baselines as _baselines
from . import scenarios as _scenarios


MATRIX_SCHEMA_VERSION: Final = "m7.evaluation-availability-matrix.v1"
ACTOR_AUDIT_REQUIREMENT: Final = "PASS_BEFORE_EXECUTION"
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class MatrixStatus(str, Enum):
    READY = "READY"
    NOT_READY = "NOT_READY"
    SKIPPED = "SKIPPED"


class ActorAuditRequirement(str, Enum):
    PASS_BEFORE_EXECUTION = ACTOR_AUDIT_REQUIREMENT


STATUS_ORDER: Final = tuple(item.value for item in MatrixStatus)
BASELINE_ORDER: Final = _baselines.BASELINE_IDS
SCENARIO_ORDER: Final = _scenarios.SCENARIO_IDS


@dataclass(frozen=True, slots=True)
class EvaluationAvailabilityCell:
    baseline_id: str
    config_switch: str
    execution_family: str
    privilege_class: str
    deployable: bool
    baseline_optional: bool
    baseline_readiness: str
    scenario_id: str
    execution_domain: str
    requirement_class: str
    scenario_readiness: str
    scenario_seed_or_null: int | None
    scenario_provider_or_null: str | None
    scenario_evidence_sha256_or_null: str | None
    status: MatrixStatus
    required_blockers: tuple[str, ...]
    optional_blockers: tuple[str, ...]
    actor_audit_requirement: ActorAuditRequirement
    execution_performed: bool

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("EvaluationAvailabilityCell must not be subclassed")

    def __post_init__(self) -> None:
        string_fields = (
            "baseline_id", "config_switch", "execution_family", "privilege_class",
            "baseline_readiness", "scenario_id", "execution_domain",
            "requirement_class", "scenario_readiness",
        )
        if any(type(getattr(self, name)) is not str for name in string_fields):
            raise TypeError("cell identity fields must be exact strings")
        if type(self.deployable) is not bool or type(self.baseline_optional) is not bool:
            raise TypeError("cell baseline booleans must be exact booleans")
        if self.scenario_seed_or_null is not None and (
            type(self.scenario_seed_or_null) is not int or self.scenario_seed_or_null < 0
        ):
            raise ValueError("scenario seed must be null or an exact nonnegative integer")
        if self.scenario_provider_or_null is not None and type(self.scenario_provider_or_null) is not str:
            raise TypeError("scenario provider must be null or an exact string")
        evidence = self.scenario_evidence_sha256_or_null
        if evidence is not None and (type(evidence) is not str or _SHA256_PATTERN.fullmatch(evidence) is None):
            raise ValueError("scenario evidence must be null or an exact SHA-256")
        if type(self.status) is not MatrixStatus:
            raise TypeError("status must be an exact MatrixStatus")
        for name in ("required_blockers", "optional_blockers"):
            value = getattr(self, name)
            if type(value) is not tuple or any(type(item) is not str or not item for item in value):
                raise TypeError(f"{name} must be an immutable tuple of nonempty strings")
            if len(value) != len(set(value)):
                raise ValueError(f"{name} must not contain duplicates")
        if type(self.actor_audit_requirement) is not ActorAuditRequirement:
            raise TypeError("actor audit requirement must be exact")
        if self.execution_performed is not False:
            raise ValueError("availability cells must never claim execution")


CELL_FIELD_ORDER: Final = tuple(item.name for item in fields(EvaluationAvailabilityCell))


@dataclass(frozen=True, slots=True)
class EvaluationAvailabilityMatrix:
    schema_version: str
    product_commit: str
    baseline_source_schema_version: str
    baseline_source_sha256: str
    scenario_source_schema_version: str
    scenario_source_sha256: str
    baseline_order: tuple[str, ...]
    scenario_order: tuple[str, ...]
    status_order: tuple[str, ...]
    cell_count: int
    ready_count: int
    not_ready_count: int
    skipped_count: int
    cells: tuple[EvaluationAvailabilityCell, ...]
    matrix_sha256: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("EvaluationAvailabilityMatrix must not be subclassed")

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str:
            raise TypeError("schema_version must be an exact string")
        if type(self.product_commit) is not str or _COMMIT_PATTERN.fullmatch(self.product_commit) is None:
            raise ValueError("product_commit must be an exact lowercase 40-hex commit")
        for name in (
            "baseline_source_schema_version", "baseline_source_sha256",
            "scenario_source_schema_version", "scenario_source_sha256", "matrix_sha256",
        ):
            if type(getattr(self, name)) is not str:
                raise TypeError(f"{name} must be an exact string")
        if any(
            _SHA256_PATTERN.fullmatch(getattr(self, name)) is None
            for name in ("baseline_source_sha256", "scenario_source_sha256", "matrix_sha256")
        ):
            raise ValueError("source and matrix identities must be exact SHA-256 values")
        for name in ("baseline_order", "scenario_order", "status_order"):
            value = getattr(self, name)
            if type(value) is not tuple or any(type(item) is not str for item in value):
                raise TypeError(f"{name} must be an immutable ordered string tuple")
        for name in ("cell_count", "ready_count", "not_ready_count", "skipped_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be an exact nonnegative integer")
        if type(self.cells) is not tuple or any(type(item) is not EvaluationAvailabilityCell for item in self.cells):
            raise TypeError("cells must be an immutable tuple of exact cells")
        expected_hash = hashlib.sha256(_canonical_bytes(_matrix_payload_mapping(self))).hexdigest()
        if self.matrix_sha256 != expected_hash:
            raise ValueError("matrix_sha256 does not bind the exact canonical payload")


MATRIX_FIELD_ORDER: Final = tuple(item.name for item in fields(EvaluationAvailabilityMatrix))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


def _baseline_source_mapping() -> OrderedDict[str, object]:
    rows: list[OrderedDict[str, object]] = []
    for row in _baselines.BASELINE_REGISTRY:
        requirements = [
            OrderedDict((
                ("name", requirement.name),
                ("accepted_provider", requirement.accepted_provider),
                ("accepted_evidence_sha256", requirement.accepted_evidence_sha256),
            ))
            for requirement in row.capability_requirements
        ]
        rows.append(OrderedDict((
            ("baseline_id", row.baseline_id.value),
            ("config_switch", row.config_switch),
            ("execution_family", row.execution_family.value),
            ("privilege_class", row.privilege_class.value),
            ("deployable", row.deployable),
            ("optional", row.optional),
            ("readiness", row.readiness.value),
            ("capability_requirements", requirements),
        )))
    return OrderedDict((
        ("schema_version", _baselines.MANIFEST_SCHEMA_VERSION),
        ("platform", _baselines.PLATFORM_ID),
        ("sensor", _baselines.SENSOR_ID),
        ("baseline_order", list(BASELINE_ORDER)),
        ("rows", rows),
    ))


_SCENARIO_SEEDS: Final = {
    _scenarios.ScenarioId.CORNER: _scenarios.CORNER_SEED,
    _scenarios.ScenarioId.FLAT_WALL: _scenarios.FLAT_WALL_SEED,
    _scenarios.ScenarioId.CORRIDOR_TUNNEL: _scenarios.CORRIDOR_TUNNEL_SEED,
    _scenarios.ScenarioId.OPEN_FIELD: _scenarios.OPEN_FIELD_SEED,
    _scenarios.ScenarioId.CLUTTERED_INDOOR: _scenarios.CLUTTERED_INDOOR_SEED,
    _scenarios.ScenarioId.UNSAFE_COMMAND_TOWARD_WALL: _scenarios.UNSAFE_COMMAND_TOWARD_WALL_SEED,
    _scenarios.ScenarioId.GAP_DOORWAY: _scenarios.GAP_DOORWAY_SEED,
}


def _scenario_source_mapping() -> OrderedDict[str, object]:
    rows: list[OrderedDict[str, object]] = []
    for row in _scenarios.scenario_registry():
        evidence = row.evidence
        rows.append(OrderedDict((
            ("scenario_id", row.scenario_id.value),
            ("execution_domain", row.execution_domain.value),
            ("requirement_class", row.requirement_class.value),
            ("readiness", evidence.readiness.value),
            ("seed_or_null", _SCENARIO_SEEDS.get(row.scenario_id)),
            ("provider_or_null", evidence.provider),
            ("evidence_sha256_or_null", evidence.evidence_sha256),
            ("unavailability_reason_or_null", evidence.unavailability_reason),
        )))
    return OrderedDict((
        ("schema_version", _scenarios.CATALOG_SCHEMA_VERSION),
        ("platform", _baselines.PLATFORM_ID),
        ("sensor", _baselines.SENSOR_ID),
        ("scenario_order", list(SCENARIO_ORDER)),
        ("rows", rows),
    ))


BASELINE_SOURCE_SHA256: Final = hashlib.sha256(_canonical_bytes(_baseline_source_mapping())).hexdigest()
SCENARIO_SOURCE_SHA256: Final = hashlib.sha256(_canonical_bytes(_scenario_source_mapping())).hexdigest()


def _derive_cell(
    baseline: _baselines.BaselineSpec,
    scenario: _scenarios.ScenarioSpec,
) -> EvaluationAvailabilityCell:
    unavailable_capabilities = tuple(
        requirement.name
        for requirement in baseline.capability_requirements
        if requirement.accepted_provider is None or requirement.accepted_evidence_sha256 is None
    )
    required_blockers = () if baseline.optional else unavailable_capabilities
    optional_blockers = unavailable_capabilities if baseline.optional else ()
    evidence = scenario.evidence
    if evidence.readiness is _scenarios.ScenarioReadiness.NOT_READY:
        reason = evidence.unavailability_reason
        if reason is None:
            raise ValueError("NOT_READY scenario lost its accepted blocker reason")
        if scenario.requirement_class is _scenarios.RequirementClass.REQUIRED:
            required_blockers += (reason,)
        else:
            optional_blockers += (reason,)
    if required_blockers:
        status = MatrixStatus.NOT_READY
    elif optional_blockers:
        status = MatrixStatus.SKIPPED
    else:
        status = MatrixStatus.READY
    return EvaluationAvailabilityCell(
        baseline.baseline_id.value,
        baseline.config_switch,
        baseline.execution_family.value,
        baseline.privilege_class.value,
        baseline.deployable,
        baseline.optional,
        baseline.readiness.value,
        scenario.scenario_id.value,
        scenario.execution_domain.value,
        scenario.requirement_class.value,
        evidence.readiness.value,
        _SCENARIO_SEEDS.get(scenario.scenario_id),
        evidence.provider,
        evidence.evidence_sha256,
        status,
        required_blockers,
        optional_blockers,
        ActorAuditRequirement.PASS_BEFORE_EXECUTION,
        False,
    )


def _cell_mapping(cell: EvaluationAvailabilityCell) -> OrderedDict[str, object]:
    return OrderedDict((
        ("baseline_id", cell.baseline_id),
        ("config_switch", cell.config_switch),
        ("execution_family", cell.execution_family),
        ("privilege_class", cell.privilege_class),
        ("deployable", cell.deployable),
        ("baseline_optional", cell.baseline_optional),
        ("baseline_readiness", cell.baseline_readiness),
        ("scenario_id", cell.scenario_id),
        ("execution_domain", cell.execution_domain),
        ("requirement_class", cell.requirement_class),
        ("scenario_readiness", cell.scenario_readiness),
        ("scenario_seed_or_null", cell.scenario_seed_or_null),
        ("scenario_provider_or_null", cell.scenario_provider_or_null),
        ("scenario_evidence_sha256_or_null", cell.scenario_evidence_sha256_or_null),
        ("status", cell.status.value),
        ("required_blockers", list(cell.required_blockers)),
        ("optional_blockers", list(cell.optional_blockers)),
        ("actor_audit_requirement", cell.actor_audit_requirement.value),
        ("execution_performed", cell.execution_performed),
    ))


def _matrix_payload_mapping(matrix: EvaluationAvailabilityMatrix) -> OrderedDict[str, object]:
    return OrderedDict((
        ("schema_version", matrix.schema_version),
        ("product_commit", matrix.product_commit),
        ("baseline_source_schema_version", matrix.baseline_source_schema_version),
        ("baseline_source_sha256", matrix.baseline_source_sha256),
        ("scenario_source_schema_version", matrix.scenario_source_schema_version),
        ("scenario_source_sha256", matrix.scenario_source_sha256),
        ("baseline_order", list(matrix.baseline_order)),
        ("scenario_order", list(matrix.scenario_order)),
        ("status_order", list(matrix.status_order)),
        ("cell_count", matrix.cell_count),
        ("ready_count", matrix.ready_count),
        ("not_ready_count", matrix.not_ready_count),
        ("skipped_count", matrix.skipped_count),
        ("cells", [_cell_mapping(cell) for cell in matrix.cells]),
    ))


def _matrix_mapping(matrix: EvaluationAvailabilityMatrix) -> OrderedDict[str, object]:
    output = _matrix_payload_mapping(matrix)
    output["matrix_sha256"] = matrix.matrix_sha256
    return output


def _derive_matrix(product_commit: str) -> EvaluationAvailabilityMatrix:
    if type(product_commit) is not str or _COMMIT_PATTERN.fullmatch(product_commit) is None:
        raise ValueError("product_commit must be an exact lowercase 40-hex commit")
    cells = tuple(
        _derive_cell(baseline, scenario)
        for baseline in _baselines.BASELINE_REGISTRY
        for scenario in _scenarios.scenario_registry()
    )
    counts = {status: sum(cell.status is status for cell in cells) for status in MatrixStatus}
    partial = EvaluationAvailabilityMatrix.__new__(EvaluationAvailabilityMatrix)
    values = (
        MATRIX_SCHEMA_VERSION, product_commit, _baselines.MANIFEST_SCHEMA_VERSION,
        BASELINE_SOURCE_SHA256, _scenarios.CATALOG_SCHEMA_VERSION, SCENARIO_SOURCE_SHA256,
        BASELINE_ORDER, SCENARIO_ORDER, STATUS_ORDER, len(cells),
        counts[MatrixStatus.READY], counts[MatrixStatus.NOT_READY],
        counts[MatrixStatus.SKIPPED], cells,
    )
    for field, value in zip(fields(EvaluationAvailabilityMatrix)[:-1], values, strict=True):
        object.__setattr__(partial, field.name, value)
    digest = hashlib.sha256(_canonical_bytes(_matrix_payload_mapping(partial))).hexdigest()
    return EvaluationAvailabilityMatrix(*values, digest)


def build_evaluation_availability_matrix(*, product_commit: str) -> EvaluationAvailabilityMatrix:
    """Build the exact accepted 9-by-10 matrix without executing any cell."""
    matrix = _derive_matrix(product_commit)
    validate_evaluation_availability_matrix(matrix)
    return matrix


def validate_evaluation_availability_matrix(matrix: EvaluationAvailabilityMatrix) -> None:
    """Reject any matrix that differs from exact current registry derivation."""
    if type(matrix) is not EvaluationAvailabilityMatrix:
        raise TypeError("matrix must be an exact EvaluationAvailabilityMatrix")
    expected = _derive_matrix(matrix.product_commit)
    if matrix != expected:
        raise ValueError("matrix differs from the exact accepted registry derivation")
    if (matrix.cell_count, matrix.ready_count, matrix.not_ready_count, matrix.skipped_count) != (90, 7, 74, 9):
        raise ValueError("matrix totals do not match the ordered 9-by-10 contract")


def encode_evaluation_availability_matrix(matrix: EvaluationAvailabilityMatrix) -> str:
    """Encode one valid matrix as exact finite canonical ASCII JSON."""
    validate_evaluation_availability_matrix(matrix)
    return _canonical_bytes(_matrix_mapping(matrix)).decode("ascii")


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> OrderedDict[str, object]:
    output: OrderedDict[str, object] = OrderedDict()
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        output[key] = value
    return output


def decode_evaluation_availability_matrix(canonical_json: str) -> EvaluationAvailabilityMatrix:
    """Decode only exact canonical bytes and rederive every accepted cell."""
    if type(canonical_json) is not str or not canonical_json:
        raise TypeError("canonical_json must be an exact nonempty string")
    if any(ord(char) > 127 for char in canonical_json):
        raise ValueError("canonical_json must contain ASCII only")
    mapping = json.loads(
        canonical_json, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
    )
    if not isinstance(mapping, OrderedDict) or tuple(mapping) != MATRIX_FIELD_ORDER:
        raise ValueError("matrix root keys are missing, extra, or reordered")
    if _canonical_bytes(mapping).decode("ascii") != canonical_json:
        raise ValueError("matrix JSON is not exact canonical encoding")
    cell_mappings = mapping["cells"]
    if type(cell_mappings) is not list:
        raise TypeError("cells JSON member must be an exact list")
    cells: list[EvaluationAvailabilityCell] = []
    for cell in cell_mappings:
        if not isinstance(cell, OrderedDict) or tuple(cell) != CELL_FIELD_ORDER:
            raise ValueError("cell keys are missing, extra, or reordered")
        cells.append(EvaluationAvailabilityCell(
            cell["baseline_id"], cell["config_switch"], cell["execution_family"],
            cell["privilege_class"], cell["deployable"], cell["baseline_optional"],
            cell["baseline_readiness"], cell["scenario_id"], cell["execution_domain"],
            cell["requirement_class"], cell["scenario_readiness"],
            cell["scenario_seed_or_null"], cell["scenario_provider_or_null"],
            cell["scenario_evidence_sha256_or_null"], MatrixStatus(cell["status"]),
            tuple(cell["required_blockers"]), tuple(cell["optional_blockers"]),
            ActorAuditRequirement(cell["actor_audit_requirement"]),
            cell["execution_performed"],
        ))
    matrix = EvaluationAvailabilityMatrix(
        mapping["schema_version"], mapping["product_commit"],
        mapping["baseline_source_schema_version"], mapping["baseline_source_sha256"],
        mapping["scenario_source_schema_version"], mapping["scenario_source_sha256"],
        tuple(mapping["baseline_order"]), tuple(mapping["scenario_order"]),
        tuple(mapping["status_order"]), mapping["cell_count"], mapping["ready_count"],
        mapping["not_ready_count"], mapping["skipped_count"], tuple(cells),
        mapping["matrix_sha256"],
    )
    validate_evaluation_availability_matrix(matrix)
    return matrix


__all__ = [
    "ACTOR_AUDIT_REQUIREMENT", "ActorAuditRequirement", "BASELINE_ORDER",
    "BASELINE_SOURCE_SHA256", "CELL_FIELD_ORDER", "EvaluationAvailabilityCell",
    "EvaluationAvailabilityMatrix", "MATRIX_FIELD_ORDER", "MATRIX_SCHEMA_VERSION",
    "MatrixStatus", "SCENARIO_ORDER", "SCENARIO_SOURCE_SHA256", "STATUS_ORDER",
    "build_evaluation_availability_matrix", "decode_evaluation_availability_matrix",
    "encode_evaluation_availability_matrix", "validate_evaluation_availability_matrix",
]
