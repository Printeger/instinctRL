"""Pure immutable Blackwell release-candidate manifest contract."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import re
from typing import Final


SCHEMA_VERSION: Final = "m8.blackwell-release-candidate.v1"
PLATFORM: Final = "TASLAB_UAV"
SENSOR: Final = "Livox MID360"
FROZEN_BASELINE: Final = "BLACKWELL_BASELINE_V2_R580"
FORMAL_SOURCE_COMMIT: Final = "6f6dee3da72d0b0202846b330adcaeeec8ee7bc3"
M7_RELEASE_INPUT_COMMIT: Final = "e2d46fc3ad909052182200b7cc958012000a454a"
MILESTONE_ORDER: Final = ("M1A", "M1B", "M2", "M3", "M4", "M5", "M6", "M7")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class EvidenceClass(str, Enum):
    STATIC = "STATIC"
    PURE = "PURE"
    BOUNDED_CUDA = "BOUNDED_CUDA"
    BOUNDED_SIMULATOR = "BOUNDED_SIMULATOR"


class GateClassification(str, Enum):
    EVIDENCE_BOUND = "EVIDENCE_BOUND"
    REQUIRES_BOUNDED_EXECUTION = "REQUIRES_BOUNDED_EXECUTION"
    UNAVAILABLE_OPTIONAL = "UNAVAILABLE_OPTIONAL"
    USER_AUTHORIZATION_REQUIRED = "USER_AUTHORIZATION_REQUIRED"


class ReleaseStatus(str, Enum):
    MANIFESTED_NOT_QUALIFIED = "MANIFESTED_NOT_QUALIFIED"


def _require_string(value: object, name: str) -> None:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be an exact nonempty string")


def _require_string_tuple(value: object, name: str) -> None:
    if type(value) is not tuple or any(type(item) is not str or not item for item in value):
        raise TypeError(f"{name} must be an immutable tuple of exact nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True, slots=True)
class MilestoneAcceptance:
    milestone_id: str
    order_id: str
    order_revision: int
    accepted_attempt_id: str
    product_commit: str
    approval_record: str
    approval_sha256: str
    evidence_classes: tuple[EvidenceClass, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MilestoneAcceptance must not be subclassed")

    def __post_init__(self) -> None:
        for name in ("milestone_id", "order_id", "accepted_attempt_id", "approval_record"):
            _require_string(getattr(self, name), name)
        if type(self.order_revision) is not int or self.order_revision < 0:
            raise ValueError("order_revision must be an exact nonnegative integer")
        if type(self.product_commit) is not str or _COMMIT_RE.fullmatch(self.product_commit) is None:
            raise ValueError("product_commit must be an exact lowercase 40-hex commit")
        if type(self.approval_sha256) is not str or _SHA256_RE.fullmatch(self.approval_sha256) is None:
            raise ValueError("approval_sha256 must be an exact lowercase SHA-256")
        if type(self.evidence_classes) is not tuple or not self.evidence_classes:
            raise TypeError("evidence_classes must be a nonempty immutable tuple")
        if any(type(item) is not EvidenceClass for item in self.evidence_classes):
            raise TypeError("evidence_classes must contain exact EvidenceClass values")
        if len(self.evidence_classes) != len(set(self.evidence_classes)):
            raise ValueError("evidence_classes must not contain duplicates")


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    baseline: str
    driver: str
    python: str
    pytorch: str
    isaac_sim: str
    isaac_lab_release: str
    isaac_lab_distribution: str
    torchrl: str
    tensordict: str
    omnidrones_commit: str
    navrl_commit: str
    formal_source_commit: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("RuntimeIdentity must not be subclassed")

    def __post_init__(self) -> None:
        for field in fields(self):
            _require_string(getattr(self, field.name), field.name)
        for name in ("omnidrones_commit", "navrl_commit", "formal_source_commit"):
            if _COMMIT_RE.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be an exact lowercase 40-hex commit")


@dataclass(frozen=True, slots=True)
class ActorBoundary:
    approved_input_classes: tuple[str, ...]
    forbidden_input_classes: tuple[str, ...]
    privileged_use_classes: tuple[str, ...]
    low_level_stabilization_state_actor_visible: bool

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ActorBoundary must not be subclassed")

    def __post_init__(self) -> None:
        for name in ("approved_input_classes", "forbidden_input_classes", "privileged_use_classes"):
            _require_string_tuple(getattr(self, name), name)
        if self.low_level_stabilization_state_actor_visible is not False:
            raise ValueError("low-level stabilization state must remain actor-invisible")


@dataclass(frozen=True, slots=True)
class AvailabilitySummary:
    matrix_sha256: str
    cell_count: int
    ready_count: int
    not_ready_count: int
    skipped_count: int
    execution_performed: bool
    required_unavailable_baselines: tuple[str, ...]
    optional_unavailable_baselines: tuple[str, ...]
    optional_unavailable_scenarios: tuple[str, ...]
    required_unavailable_scenarios: tuple[str, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("AvailabilitySummary must not be subclassed")

    def __post_init__(self) -> None:
        if type(self.matrix_sha256) is not str or _SHA256_RE.fullmatch(self.matrix_sha256) is None:
            raise ValueError("matrix_sha256 must be an exact lowercase SHA-256")
        for name in ("cell_count", "ready_count", "not_ready_count", "skipped_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be an exact nonnegative integer")
        if self.ready_count + self.not_ready_count + self.skipped_count != self.cell_count:
            raise ValueError("availability counts must sum to cell_count")
        if self.execution_performed is not False:
            raise ValueError("availability summary must never claim execution")
        for name in (
            "required_unavailable_baselines", "optional_unavailable_baselines",
            "optional_unavailable_scenarios", "required_unavailable_scenarios",
        ):
            _require_string_tuple(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class QualificationGate:
    gate_id: str
    description: str
    classification: GateClassification

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("QualificationGate must not be subclassed")

    def __post_init__(self) -> None:
        _require_string(self.gate_id, "gate_id")
        _require_string(self.description, "description")
        if type(self.classification) is not GateClassification:
            raise TypeError("classification must be an exact GateClassification")


@dataclass(frozen=True, slots=True)
class ReleaseCandidateManifest:
    schema_version: str
    candidate_commit: str
    m7_release_input_commit: str
    platform: str
    sensor: str
    frozen_baseline: str
    formal_source_commit: str
    milestone_order: tuple[str, ...]
    accepted_milestones: tuple[MilestoneAcceptance, ...]
    runtime_identity: RuntimeIdentity
    actor_boundary: ActorBoundary
    availability_summary: AvailabilitySummary
    qualification_gates: tuple[QualificationGate, ...]
    release_status: ReleaseStatus
    qualification_executed: bool
    release_approved: bool
    scale_training_authorized: bool
    merge_authorized: bool
    push_authorized: bool
    manifest_sha256: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ReleaseCandidateManifest must not be subclassed")

    def __post_init__(self) -> None:
        for name in (
            "schema_version", "m7_release_input_commit", "platform", "sensor",
            "frozen_baseline", "formal_source_commit",
        ):
            _require_string(getattr(self, name), name)
        if type(self.candidate_commit) is not str or _COMMIT_RE.fullmatch(self.candidate_commit) is None:
            raise ValueError("candidate_commit must be an exact lowercase 40-hex commit")
        if _COMMIT_RE.fullmatch(self.m7_release_input_commit) is None or _COMMIT_RE.fullmatch(self.formal_source_commit) is None:
            raise ValueError("source commits must be exact lowercase 40-hex commits")
        _require_string_tuple(self.milestone_order, "milestone_order")
        if type(self.accepted_milestones) is not tuple or any(type(item) is not MilestoneAcceptance for item in self.accepted_milestones):
            raise TypeError("accepted_milestones must be an immutable exact-record tuple")
        if type(self.runtime_identity) is not RuntimeIdentity or type(self.actor_boundary) is not ActorBoundary:
            raise TypeError("runtime and actor records must have exact types")
        if type(self.availability_summary) is not AvailabilitySummary:
            raise TypeError("availability_summary must have the exact type")
        if type(self.qualification_gates) is not tuple or any(type(item) is not QualificationGate for item in self.qualification_gates):
            raise TypeError("qualification_gates must be an immutable exact-record tuple")
        if type(self.release_status) is not ReleaseStatus:
            raise TypeError("release_status must be an exact ReleaseStatus")
        for name in (
            "qualification_executed", "release_approved", "scale_training_authorized",
            "merge_authorized", "push_authorized",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"{name} must remain false")
        if type(self.manifest_sha256) is not str or _SHA256_RE.fullmatch(self.manifest_sha256) is None:
            raise ValueError("manifest_sha256 must be an exact lowercase SHA-256")


MILESTONE_FIELD_ORDER: Final = tuple(field.name for field in fields(MilestoneAcceptance))
RUNTIME_FIELD_ORDER: Final = tuple(field.name for field in fields(RuntimeIdentity))
ACTOR_BOUNDARY_FIELD_ORDER: Final = tuple(field.name for field in fields(ActorBoundary))
AVAILABILITY_FIELD_ORDER: Final = tuple(field.name for field in fields(AvailabilitySummary))
GATE_FIELD_ORDER: Final = tuple(field.name for field in fields(QualificationGate))
MANIFEST_FIELD_ORDER: Final = tuple(field.name for field in fields(ReleaseCandidateManifest))
MANIFEST_PAYLOAD_FIELD_ORDER: Final = MANIFEST_FIELD_ORDER[:-1]


ACCEPTED_MILESTONES: Final = (
    MilestoneAcceptance("M1A", "M1A-Q-001", 0, "M1A-001-Q0005", "6eb521f2174a64030e30ea5e8ce8db1f89006619", "commander/reviews/M1A-Q-001_REVIEW.md", "773ea350eaa6925fed0fad0851a10fbda410491cf217b3ce0e40513c05af29cf", (EvidenceClass.STATIC,)),
    MilestoneAcceptance("M1B", "M1B-001", 3, "M1B-001-D0004", "0f4345e979707f62d9e5ae8ca399bd0029f005b3", "commander/reviews/M1B-001_D0004_APPROVAL.md", "ac8d111f860ecb157083e335bf37a6830719d9b3b3ab7f9a6e2e97418830505a", (EvidenceClass.STATIC, EvidenceClass.PURE)),
    MilestoneAcceptance("M2", "M2-001", 0, "M2-001-D0001", "ecdb482bb05c08cfad141fb290a3d6c2e96298de", "commander/reviews/M2-001_D0001_APPROVAL.md", "48f0da97c532a37ced1ea80a7edac933e00a44f5081345bec86702fbea8eb333", (EvidenceClass.STATIC, EvidenceClass.PURE, EvidenceClass.BOUNDED_CUDA, EvidenceClass.BOUNDED_SIMULATOR)),
    MilestoneAcceptance("M3", "M3-001", 1, "M3-001-D0002", "99ec3b0099b887a31962b2a5c862e3e73efc6872", "commander/reviews/M3-001_D0002_APPROVAL.md", "b8749d36c7696c0ca54e8348afa8ef3fcc6278300685334b79f944ede37dac82", (EvidenceClass.STATIC, EvidenceClass.PURE, EvidenceClass.BOUNDED_CUDA, EvidenceClass.BOUNDED_SIMULATOR)),
    MilestoneAcceptance("M4", "M4-001", 1, "M4-001-D0002", "bff6f853eb6d95b12d4de7b95cbfa5a7f2283bbc", "commander/reviews/M4-001_D0002_APPROVAL.md", "8449d053d06ccc9e4b98e7a55734e6fb7b3217bf2c7c8a73e688f253c6c1617c", (EvidenceClass.STATIC, EvidenceClass.PURE, EvidenceClass.BOUNDED_CUDA)),
    MilestoneAcceptance("M5", "M5-002", 1, "M5-002-D0002", "acda9b1b30ca8aceaf6cf6eaca9d2c1d4bdaaa73", "commander/reviews/M5-002_D0002_APPROVAL.md", "93a1a45b524eb702521251d7fba6e385ae93948d27bd839f2a2c2e182c762aba", (EvidenceClass.STATIC, EvidenceClass.PURE, EvidenceClass.BOUNDED_CUDA, EvidenceClass.BOUNDED_SIMULATOR)),
    MilestoneAcceptance("M6", "M6-004", 0, "M6-004-D0001", "033514d8fa7d8d2bb2ece5faa1521d528e32b0a4", "commander/reviews/M6-004_D0001_APPROVAL.md", "7fbb7c919081a8614995d1a95cf4976cd1c777b826139bc8c037f4ee47e0db6a", (EvidenceClass.STATIC, EvidenceClass.PURE, EvidenceClass.BOUNDED_CUDA, EvidenceClass.BOUNDED_SIMULATOR)),
    MilestoneAcceptance("M7", "M7-021", 1, "M7-021-D0002", M7_RELEASE_INPUT_COMMIT, "commander/reviews/M7-021_D0002_APPROVAL.md", "9bac59948573fe24fdd073ccabf010b7914a7173f45cd7a250f5346ffd8a3a7f", (EvidenceClass.STATIC, EvidenceClass.PURE, EvidenceClass.BOUNDED_CUDA)),
)

RUNTIME_IDENTITY: Final = RuntimeIdentity(
    FROZEN_BASELINE, "580.173.02", "3.11.15", "2.7.0+cu128", "5.1.0",
    "2.3.2", "0.54.2", "0.10.0", "0.10.0",
    "5045c71cc73d224144cc08bc38d65cabeedaec8c",
    "3725bcc2e7c1be4ecf1455d922299ae85042603a", FORMAL_SOURCE_COMMIT,
)

ACTOR_BOUNDARY: Final = ActorBoundary(
    ("MID360 measurement history", "validity and reliability", "timestamps and age", "approved IMU-derived fields", "commanded velocity"),
    ("pose", "odometry", "explicit or reconstructed velocity estimates", "maps", "SLAM state", "goal-relative privileged state", "obstacle ground truth", "privileged simulator state", "future information"),
    ("reward", "critic", "evaluation", "logging", "labelled upper-bound baselines"),
    False,
)

AVAILABILITY_SUMMARY: Final = AvailabilitySummary(
    "a0d1c1cdb8cedf18705a7ab4144a682b75ba259ef5e83809d85947546f540b65",
    90, 7, 74, 9, False,
    ("B1", "B2", "B3", "B4", "B5", "B6", "B7"), ("B8",),
    ("dynamic_obstacle",),
    ("real_robot_observable_holding", "real_robot_unsafe_command_approach"),
)

QUALIFICATION_GATES: Final = (
    QualificationGate("accepted_milestone_chain_integrity", "accepted milestone-chain integrity", GateClassification.EVIDENCE_BOUND),
    QualificationGate("frozen_runtime_identity", "frozen runtime/package/driver/source identity", GateClassification.EVIDENCE_BOUND),
    QualificationGate("release_candidate_tree", "clean exact release-candidate Git tree and reproducible install/import surface", GateClassification.REQUIRES_BOUNDED_EXECUTION),
    QualificationGate("platform_sensor_contract", "TASLAB platform and MID360 topology/frame contract", GateClassification.EVIDENCE_BOUND),
    QualificationGate("actor_input_audit", "deployed actor positive/negative input audit", GateClassification.EVIDENCE_BOUND),
    QualificationGate("privilege_isolation", "reward/critic/evaluation privilege isolation", GateClassification.EVIDENCE_BOUND),
    QualificationGate("bounded_pipeline", "deterministic bounded environment/controller/rollout/checkpoint reload", GateClassification.REQUIRES_BOUNDED_EXECUTION),
    QualificationGate("b0_scenario_consistency", "M7 B0 seven-scenario evidence consistency", GateClassification.REQUIRES_BOUNDED_EXECUTION),
    QualificationGate("b1_b7_required_unavailable", "B1-B7 unavailable required capability disclosure", GateClassification.REQUIRES_BOUNDED_EXECUTION),
    QualificationGate("b8_dynamic_optional_unavailable", "B8 and dynamic-obstacle optional-unavailable disclosure", GateClassification.UNAVAILABLE_OPTIONAL),
    QualificationGate("real_robot_required_unavailable", "real-robot required-unavailable disclosure", GateClassification.REQUIRES_BOUNDED_EXECUTION),
    QualificationGate("release_evidence_completeness", "release evidence/index completeness", GateClassification.REQUIRES_BOUNDED_EXECUTION),
    QualificationGate("user_authorization", "user authorization for scale training, merge and push", GateClassification.USER_AUTHORIZATION_REQUIRED),
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


def _milestone_mapping(row: MilestoneAcceptance) -> OrderedDict[str, object]:
    return OrderedDict((
        ("milestone_id", row.milestone_id), ("order_id", row.order_id),
        ("order_revision", row.order_revision), ("accepted_attempt_id", row.accepted_attempt_id),
        ("product_commit", row.product_commit), ("approval_record", row.approval_record),
        ("approval_sha256", row.approval_sha256),
        ("evidence_classes", [item.value for item in row.evidence_classes]),
    ))


def _runtime_mapping(row: RuntimeIdentity) -> OrderedDict[str, object]:
    return OrderedDict((field.name, getattr(row, field.name)) for field in fields(row))


def _actor_mapping(row: ActorBoundary) -> OrderedDict[str, object]:
    return OrderedDict((
        ("approved_input_classes", list(row.approved_input_classes)),
        ("forbidden_input_classes", list(row.forbidden_input_classes)),
        ("privileged_use_classes", list(row.privileged_use_classes)),
        ("low_level_stabilization_state_actor_visible", row.low_level_stabilization_state_actor_visible),
    ))


def _availability_mapping(row: AvailabilitySummary) -> OrderedDict[str, object]:
    return OrderedDict((
        ("matrix_sha256", row.matrix_sha256), ("cell_count", row.cell_count),
        ("ready_count", row.ready_count), ("not_ready_count", row.not_ready_count),
        ("skipped_count", row.skipped_count), ("execution_performed", row.execution_performed),
        ("required_unavailable_baselines", list(row.required_unavailable_baselines)),
        ("optional_unavailable_baselines", list(row.optional_unavailable_baselines)),
        ("optional_unavailable_scenarios", list(row.optional_unavailable_scenarios)),
        ("required_unavailable_scenarios", list(row.required_unavailable_scenarios)),
    ))


def _gate_mapping(row: QualificationGate) -> OrderedDict[str, object]:
    return OrderedDict((
        ("gate_id", row.gate_id), ("description", row.description),
        ("classification", row.classification.value),
    ))


def _payload_mapping(manifest: ReleaseCandidateManifest) -> OrderedDict[str, object]:
    return OrderedDict((
        ("schema_version", manifest.schema_version), ("candidate_commit", manifest.candidate_commit),
        ("m7_release_input_commit", manifest.m7_release_input_commit), ("platform", manifest.platform),
        ("sensor", manifest.sensor), ("frozen_baseline", manifest.frozen_baseline),
        ("formal_source_commit", manifest.formal_source_commit),
        ("milestone_order", list(manifest.milestone_order)),
        ("accepted_milestones", [_milestone_mapping(row) for row in manifest.accepted_milestones]),
        ("runtime_identity", _runtime_mapping(manifest.runtime_identity)),
        ("actor_boundary", _actor_mapping(manifest.actor_boundary)),
        ("availability_summary", _availability_mapping(manifest.availability_summary)),
        ("qualification_gates", [_gate_mapping(row) for row in manifest.qualification_gates]),
        ("release_status", manifest.release_status.value),
        ("qualification_executed", manifest.qualification_executed),
        ("release_approved", manifest.release_approved),
        ("scale_training_authorized", manifest.scale_training_authorized),
        ("merge_authorized", manifest.merge_authorized), ("push_authorized", manifest.push_authorized),
    ))


def _manifest_mapping(manifest: ReleaseCandidateManifest) -> OrderedDict[str, object]:
    result = _payload_mapping(manifest)
    result["manifest_sha256"] = manifest.manifest_sha256
    return result


def _validate_static_records() -> None:
    if tuple(row.milestone_id for row in ACCEPTED_MILESTONES) != MILESTONE_ORDER:
        raise RuntimeError("accepted milestone order drift")
    if len({row.milestone_id for row in ACCEPTED_MILESTONES}) != len(ACCEPTED_MILESTONES):
        raise RuntimeError("duplicate accepted milestone")
    if len({gate.gate_id for gate in QUALIFICATION_GATES}) != len(QUALIFICATION_GATES):
        raise RuntimeError("duplicate qualification gate")
    for row in ACCEPTED_MILESTONES:
        if type(row) is not MilestoneAcceptance or type(row.order_revision) is not int or row.order_revision < 0:
            raise RuntimeError("invalid accepted milestone record")
        if _COMMIT_RE.fullmatch(row.product_commit) is None or _SHA256_RE.fullmatch(row.approval_sha256) is None:
            raise RuntimeError("invalid accepted milestone identity")
        if type(row.evidence_classes) is not tuple or not row.evidence_classes:
            raise RuntimeError("missing accepted evidence classification")
        if any(type(item) is not EvidenceClass for item in row.evidence_classes):
            raise RuntimeError("invalid accepted evidence classification")
    if RUNTIME_IDENTITY.formal_source_commit != FORMAL_SOURCE_COMMIT:
        raise RuntimeError("formal source identity drift")
    if AVAILABILITY_SUMMARY.execution_performed is not False:
        raise RuntimeError("availability summary must not claim execution")
    if (AVAILABILITY_SUMMARY.cell_count, AVAILABILITY_SUMMARY.ready_count, AVAILABILITY_SUMMARY.not_ready_count, AVAILABILITY_SUMMARY.skipped_count) != (90, 7, 74, 9):
        raise RuntimeError("M7 availability summary drift")


def build_release_candidate(candidate_commit: str) -> ReleaseCandidateManifest:
    """Build the exact immutable manifest for a caller-supplied candidate."""
    if type(candidate_commit) is not str or _COMMIT_RE.fullmatch(candidate_commit) is None:
        raise ValueError("candidate_commit must be an exact lowercase 40-hex commit")
    _validate_static_records()
    provisional = ReleaseCandidateManifest(
        SCHEMA_VERSION, candidate_commit, M7_RELEASE_INPUT_COMMIT, PLATFORM, SENSOR,
        FROZEN_BASELINE, FORMAL_SOURCE_COMMIT, MILESTONE_ORDER, ACCEPTED_MILESTONES,
        RUNTIME_IDENTITY, ACTOR_BOUNDARY, AVAILABILITY_SUMMARY, QUALIFICATION_GATES,
        ReleaseStatus.MANIFESTED_NOT_QUALIFIED, False, False, False, False, False, "0" * 64,
    )
    digest = hashlib.sha256(_canonical_bytes(_payload_mapping(provisional))).hexdigest()
    return ReleaseCandidateManifest(*tuple(getattr(provisional, name) for name in MANIFEST_PAYLOAD_FIELD_ORDER), digest)


def validate_release_candidate(manifest: ReleaseCandidateManifest) -> None:
    """Fail closed unless *manifest* equals the complete accepted contract."""
    if type(manifest) is not ReleaseCandidateManifest:
        raise TypeError("manifest must be an exact ReleaseCandidateManifest")
    if manifest != build_release_candidate(manifest.candidate_commit):
        raise ValueError("manifest does not equal the accepted release-candidate contract")


def encode_release_candidate(manifest: ReleaseCandidateManifest) -> bytes:
    """Return finite deterministic canonical ASCII JSON."""
    validate_release_candidate(manifest)
    return _canonical_bytes(_manifest_mapping(manifest))


def _ordered_object(pairs: list[tuple[str, object]]) -> OrderedDict[str, object]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON field")
    return OrderedDict(pairs)


def _reject_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant: {value}")


def decode_release_candidate(encoded: bytes) -> ReleaseCandidateManifest:
    """Decode canonical JSON and rederive every semantic field."""
    if type(encoded) is not bytes:
        raise TypeError("encoded manifest must be exact bytes")
    try:
        text = encoded.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("manifest must be finite ASCII JSON") from exc
    try:
        value = json.loads(text, object_pairs_hook=_ordered_object, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid canonical manifest JSON") from exc
    if type(value) is not OrderedDict or tuple(value) != MANIFEST_FIELD_ORDER:
        raise ValueError("manifest fields are missing, extra or reordered")
    candidate = value.get("candidate_commit")
    if type(candidate) is not str or _COMMIT_RE.fullmatch(candidate) is None:
        raise ValueError("invalid candidate commit")
    expected = build_release_candidate(candidate)
    if value != _manifest_mapping(expected) or encoded != encode_release_candidate(expected):
        raise ValueError("manifest semantic or canonical identity mismatch")
    return expected


_validate_static_records()
