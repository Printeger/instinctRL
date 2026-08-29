"""Immutable Handbook evaluation-scenario identities and fail-closed run plans."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import re
from typing import Final

from .baselines import BaselineRunManifest, PLATFORM_ID, SENSOR_ID


CATALOG_SCHEMA_VERSION: Final = "m7.scenario-catalog.v1"
RUN_PLAN_SCHEMA_VERSION: Final = "m7.scenario-run-plan.v1"
REAL_ROBOT_REPEATABILITY_METRIC: Final = "real_robot_repeatability"


class ScenarioId(str, Enum):
    CORNER = "corner"
    FLAT_WALL = "flat_wall"
    CORRIDOR_TUNNEL = "corridor_tunnel"
    OPEN_FIELD = "open_field"
    CLUTTERED_INDOOR = "cluttered_indoor"
    UNSAFE_COMMAND_TOWARD_WALL = "unsafe_command_toward_wall"
    GAP_DOORWAY = "gap_doorway"
    DYNAMIC_OBSTACLE = "dynamic_obstacle"
    REAL_ROBOT_OBSERVABLE_HOLDING = "real_robot_observable_holding"
    REAL_ROBOT_UNSAFE_COMMAND_APPROACH = "real_robot_unsafe_command_approach"


class ExecutionDomain(str, Enum):
    SIMULATOR = "simulator"
    REAL_ROBOT = "real_robot"


class RequirementClass(str, Enum):
    REQUIRED = "required"
    OPTIONAL_WHEN_AVAILABLE = "optional_when_available"


class ScenarioReadiness(str, Enum):
    NOT_READY = "NOT_READY"
    READY = "READY"


SCENARIO_IDS: Final = tuple(item.value for item in ScenarioId)
ACCEPTED_METRIC_NAMES: Final = (
    "tracking_rmse_m_s",
    "null_station_drift_mean_m",
    "null_station_speed_mean_m_s",
    "anchor_error_mean_m",
    "minimum_clearance_m",
    "collision_rate",
    "ics_violation_rate",
    "command_preservation_ratio",
    "intervention_frequency",
    "observability_mean",
)
FORWARDED_ACTOR_FIELDS: Final = ("commanded_body_velocity_m_s",)
GENERATOR_PRIVILEGED_FIELDS: Final = (
    "pose", "velocity", "obstacle_state", "mesh", "ray_hits",
    "surface_normals", "odometry", "map", "ground_truth",
)
REQUIRED_RUN_LOG_FIELDS: Final = (
    "product_commit", "platform", "sensor", "baseline_id", "scenario_id",
    "seed", "actor_audit_status", "scenario_readiness", "scenario_provider",
    "scenario_evidence_sha256", "scenario_unavailability_reason",
    "execution_domain", "requirement_class",
)
CATALOG_FIELD_ORDER: Final = (
    "schema_version", "platform", "sensor", "scenario_order",
    "forwarded_actor_fields", "generator_privileged_fields",
    "required_run_log_fields", "accepted_metric_names", "rows",
)
SCENARIO_ROW_FIELD_ORDER: Final = (
    "scenario_id", "execution_domain", "requirement_class", "readiness",
    "provider", "evidence_sha256", "unavailability_reason",
    "forwarded_actor_fields", "generator_privileged_fields",
    "required_run_log_fields", "metric_obligations", "executable",
)
RUN_PLAN_FIELD_ORDER: Final = (
    "schema_version", "catalog_sha256", "manifest_sha256", "product_commit",
    "platform", "sensor", "baseline_id", "scenario_id", "seed",
    "actor_audit_status", "baseline_execution_status", "execution_domain",
    "requirement_class", "scenario_readiness", "scenario_provider",
    "scenario_evidence_sha256", "scenario_unavailability_reason",
    "forwarded_actor_fields", "required_run_log_fields", "metric_obligations",
    "executable",
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")


def _exact_string_tuple(name: str, value: object, *, allow_empty: bool = False) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact immutable tuple")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    if any(type(item) is not str or not item or any(ord(char) < 32 for char in item) for item in value):
        raise TypeError(f"{name} entries must be exact nonempty control-free strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} entries must not duplicate")
    return value


def _nonempty_reason(value: object) -> str:
    if type(value) is not str or not value.strip() or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError("unavailability_reason must be one canonical nonempty control-free string")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> OrderedDict[str, object]:
    output: OrderedDict[str, object] = OrderedDict()
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        output[key] = value
    return output


def _decode_canonical_json(canonical_json: object) -> OrderedDict[str, object]:
    if type(canonical_json) is not str or not canonical_json:
        raise TypeError("canonical_json must be an exact nonempty string")
    if any(ord(char) > 127 for char in canonical_json):
        raise ValueError("canonical_json must use ASCII bytes")
    decoded = json.loads(
        canonical_json, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
    )
    if not isinstance(decoded, OrderedDict):
        raise ValueError("canonical JSON root must be an object")
    if _canonical_bytes(decoded).decode("ascii") != canonical_json:
        raise ValueError("JSON must use the exact canonical byte encoding")
    return decoded


@dataclass(frozen=True, slots=True)
class ScenarioEvidence:
    readiness: ScenarioReadiness
    provider: str | None
    evidence_sha256: str | None
    unavailability_reason: str | None

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ScenarioEvidence must not be subclassed")

    def __post_init__(self) -> None:
        if type(self.readiness) is not ScenarioReadiness:
            raise TypeError("readiness must be an exact ScenarioReadiness")
        paired = (self.provider is None) == (self.evidence_sha256 is None)
        if not paired:
            raise ValueError("provider and evidence_sha256 must be present together")
        if self.readiness is ScenarioReadiness.READY:
            if type(self.provider) is not str or _TOKEN_PATTERN.fullmatch(self.provider) is None:
                raise ValueError("READY evidence requires one canonical provider")
            if type(self.evidence_sha256) is not str or _SHA256_PATTERN.fullmatch(self.evidence_sha256) is None:
                raise ValueError("READY evidence requires one exact SHA-256")
            if self.unavailability_reason is not None:
                raise ValueError("READY evidence cannot contain an unavailability reason")
        else:
            if self.provider is not None or self.evidence_sha256 is not None:
                raise ValueError("NOT_READY evidence must not claim provider or evidence bytes")
            _nonempty_reason(self.unavailability_reason)


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    scenario_id: ScenarioId
    execution_domain: ExecutionDomain
    requirement_class: RequirementClass
    evidence: ScenarioEvidence
    forwarded_actor_fields: tuple[str, ...]
    generator_privileged_fields: tuple[str, ...]
    required_run_log_fields: tuple[str, ...]
    metric_obligations: tuple[str, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ScenarioSpec must not be subclassed")

    def __post_init__(self) -> None:
        if type(self.scenario_id) is not ScenarioId:
            raise TypeError("scenario_id must be an exact ScenarioId")
        if type(self.execution_domain) is not ExecutionDomain:
            raise TypeError("execution_domain must be an exact ExecutionDomain")
        if type(self.requirement_class) is not RequirementClass:
            raise TypeError("requirement_class must be an exact RequirementClass")
        if type(self.evidence) is not ScenarioEvidence:
            raise TypeError("evidence must be an exact ScenarioEvidence")
        _exact_string_tuple("forwarded_actor_fields", self.forwarded_actor_fields)
        _exact_string_tuple("generator_privileged_fields", self.generator_privileged_fields)
        _exact_string_tuple("required_run_log_fields", self.required_run_log_fields)
        _exact_string_tuple("metric_obligations", self.metric_obligations)
        if self.forwarded_actor_fields != FORWARDED_ACTOR_FIELDS:
            raise ValueError("scenario generator may forward only the body command")
        if self.generator_privileged_fields != GENERATOR_PRIVILEGED_FIELDS:
            raise ValueError("scenario privilege fields must remain exact and evaluation-only")
        if self.required_run_log_fields != REQUIRED_RUN_LOG_FIELDS:
            raise ValueError("required run-log identity fields are not exact")
        expected_metrics = ACCEPTED_METRIC_NAMES + (
            (REAL_ROBOT_REPEATABILITY_METRIC,)
            if self.execution_domain is ExecutionDomain.REAL_ROBOT else ()
        )
        if self.metric_obligations != expected_metrics:
            raise ValueError("metric obligations do not match the exact domain contract")


@dataclass(frozen=True, slots=True)
class ScenarioSelection:
    scenario_id: ScenarioId
    seed: int

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ScenarioSelection must not be subclassed")

    def __post_init__(self) -> None:
        if type(self.scenario_id) is not ScenarioId:
            raise TypeError("scenario_id must be an exact ScenarioId")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be an exact nonnegative integer")


def _evidence_clone(value: ScenarioEvidence) -> ScenarioEvidence:
    return ScenarioEvidence(
        value.readiness, value.provider, value.evidence_sha256, value.unavailability_reason,
    )


def _spec_clone(value: ScenarioSpec) -> ScenarioSpec:
    return ScenarioSpec(
        value.scenario_id, value.execution_domain, value.requirement_class,
        _evidence_clone(value.evidence), tuple(value.forwarded_actor_fields),
        tuple(value.generator_privileged_fields), tuple(value.required_run_log_fields),
        tuple(value.metric_obligations),
    )


def _not_ready(reason: str) -> ScenarioEvidence:
    return ScenarioEvidence(ScenarioReadiness.NOT_READY, None, None, reason)


def _spec(
    scenario_id: ScenarioId,
    domain: ExecutionDomain,
    requirement: RequirementClass,
    reason: str,
) -> ScenarioSpec:
    metrics = ACCEPTED_METRIC_NAMES + (
        (REAL_ROBOT_REPEATABILITY_METRIC,) if domain is ExecutionDomain.REAL_ROBOT else ()
    )
    return ScenarioSpec(
        scenario_id, domain, requirement, _not_ready(reason), FORWARDED_ACTOR_FIELDS,
        GENERATOR_PRIVILEGED_FIELDS, REQUIRED_RUN_LOG_FIELDS, metrics,
    )


_SCENARIO_REGISTRY = (
    _spec(ScenarioId.CORNER, ExecutionDomain.SIMULATOR, RequirementClass.REQUIRED, "no accepted corner scenario fixture or evidence"),
    _spec(ScenarioId.FLAT_WALL, ExecutionDomain.SIMULATOR, RequirementClass.REQUIRED, "no accepted flat-wall scenario fixture or evidence"),
    _spec(ScenarioId.CORRIDOR_TUNNEL, ExecutionDomain.SIMULATOR, RequirementClass.REQUIRED, "no accepted corridor/tunnel scenario fixture or evidence"),
    _spec(ScenarioId.OPEN_FIELD, ExecutionDomain.SIMULATOR, RequirementClass.REQUIRED, "no accepted open-field scenario fixture or evidence"),
    _spec(ScenarioId.CLUTTERED_INDOOR, ExecutionDomain.SIMULATOR, RequirementClass.REQUIRED, "no accepted cluttered-indoor scenario fixture or evidence"),
    _spec(ScenarioId.UNSAFE_COMMAND_TOWARD_WALL, ExecutionDomain.SIMULATOR, RequirementClass.REQUIRED, "no accepted unsafe-command-toward-wall scenario fixture or evidence"),
    _spec(ScenarioId.GAP_DOORWAY, ExecutionDomain.SIMULATOR, RequirementClass.REQUIRED, "no accepted gap/doorway scenario fixture or evidence"),
    _spec(ScenarioId.DYNAMIC_OBSTACLE, ExecutionDomain.SIMULATOR, RequirementClass.OPTIONAL_WHEN_AVAILABLE, "optional dynamic-obstacle scenario has no accepted fixture or evidence"),
    _spec(ScenarioId.REAL_ROBOT_OBSERVABLE_HOLDING, ExecutionDomain.REAL_ROBOT, RequirementClass.REQUIRED, "no accepted real-robot observable-holding access, fixture, or evidence"),
    _spec(ScenarioId.REAL_ROBOT_UNSAFE_COMMAND_APPROACH, ExecutionDomain.REAL_ROBOT, RequirementClass.REQUIRED, "no accepted real-robot unsafe-command-approach access, fixture, or evidence"),
)


def _validate_registry(registry: object) -> tuple[ScenarioSpec, ...]:
    if type(registry) is not tuple or any(type(item) is not ScenarioSpec for item in registry):
        raise TypeError("scenario registry must be one exact tuple of ScenarioSpec")
    if tuple(item.scenario_id.value for item in registry) != SCENARIO_IDS:
        raise ValueError("scenario registry IDs/order are not canonical")
    if len({item.scenario_id for item in registry}) != 10:
        raise ValueError("scenario IDs must not duplicate")
    if tuple(item.execution_domain for item in registry[:8]) != (ExecutionDomain.SIMULATOR,) * 8 or tuple(
        item.execution_domain for item in registry[8:]
    ) != (ExecutionDomain.REAL_ROBOT,) * 2:
        raise ValueError("scenario domain split must be eight simulator and two real-robot rows")
    optional = tuple(item.scenario_id for item in registry if item.requirement_class is RequirementClass.OPTIONAL_WHEN_AVAILABLE)
    if optional != (ScenarioId.DYNAMIC_OBSTACLE,):
        raise ValueError("dynamic_obstacle must be the sole optional row")
    if any(item.evidence.readiness is not ScenarioReadiness.NOT_READY for item in registry):
        raise ValueError("M7-013 scenario registry must remain entirely NOT_READY")
    return registry


_validate_registry(_SCENARIO_REGISTRY)
SCENARIO_SPEC_FIELD_ORDER: Final = tuple(item.name for item in fields(ScenarioSpec))
SCENARIO_EVIDENCE_FIELD_ORDER: Final = tuple(item.name for item in fields(ScenarioEvidence))
SCENARIO_SELECTION_FIELD_ORDER: Final = tuple(item.name for item in fields(ScenarioSelection))


def scenario_registry() -> tuple[ScenarioSpec, ...]:
    """Return one independently immutable copy of the exact catalog rows."""
    return tuple(_spec_clone(item) for item in _SCENARIO_REGISTRY)


def resolve_scenario(scenario_id: ScenarioId | str) -> ScenarioSpec:
    """Resolve one exact ID without aliases, normalization or fallback."""
    if type(scenario_id) is ScenarioId:
        key = scenario_id.value
    elif type(scenario_id) is str:
        key = scenario_id
    else:
        raise TypeError("scenario_id must be an exact ScenarioId or string")
    for item in _SCENARIO_REGISTRY:
        if item.scenario_id.value == key:
            return _spec_clone(item)
    raise KeyError(f"unknown canonical scenario_id: {key!r}")


def select_scenario(*, scenario_id: ScenarioId | str, seed: int) -> ScenarioSelection:
    """Create one exact immutable selection with no readiness or execution path."""
    spec = resolve_scenario(scenario_id)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact nonnegative integer")
    return ScenarioSelection(spec.scenario_id, seed)


def _row_mapping(spec: ScenarioSpec) -> OrderedDict[str, object]:
    evidence = spec.evidence
    return OrderedDict((
        ("scenario_id", spec.scenario_id.value),
        ("execution_domain", spec.execution_domain.value),
        ("requirement_class", spec.requirement_class.value),
        ("readiness", evidence.readiness.value),
        ("provider", evidence.provider),
        ("evidence_sha256", evidence.evidence_sha256),
        ("unavailability_reason", evidence.unavailability_reason),
        ("forwarded_actor_fields", list(spec.forwarded_actor_fields)),
        ("generator_privileged_fields", list(spec.generator_privileged_fields)),
        ("required_run_log_fields", list(spec.required_run_log_fields)),
        ("metric_obligations", list(spec.metric_obligations)),
        ("executable", False),
    ))


def _catalog_mapping() -> OrderedDict[str, object]:
    rows = scenario_registry()
    return OrderedDict((
        ("schema_version", CATALOG_SCHEMA_VERSION),
        ("platform", PLATFORM_ID),
        ("sensor", SENSOR_ID),
        ("scenario_order", list(SCENARIO_IDS)),
        ("forwarded_actor_fields", list(FORWARDED_ACTOR_FIELDS)),
        ("generator_privileged_fields", list(GENERATOR_PRIVILEGED_FIELDS)),
        ("required_run_log_fields", list(REQUIRED_RUN_LOG_FIELDS)),
        ("accepted_metric_names", list(ACCEPTED_METRIC_NAMES)),
        ("rows", [_row_mapping(item) for item in rows]),
    ))


def _validate_catalog_mapping(mapping: OrderedDict[str, object]) -> None:
    if tuple(mapping) != CATALOG_FIELD_ORDER:
        raise ValueError("catalog keys are missing, extra, or reordered")
    expected = _catalog_mapping()
    if mapping != expected:
        raise ValueError("catalog JSON does not encode the exact current registry")
    rows = mapping["rows"]
    if type(rows) is not list or any(not isinstance(row, OrderedDict) or tuple(row) != SCENARIO_ROW_FIELD_ORDER for row in rows):
        raise ValueError("catalog row keys are missing, extra, or reordered")


@dataclass(frozen=True, slots=True)
class ScenarioCatalog:
    canonical_json: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ScenarioCatalog must not be subclassed")

    def __post_init__(self) -> None:
        mapping = _decode_canonical_json(self.canonical_json)
        _validate_catalog_mapping(mapping)

    @classmethod
    def from_json(cls, canonical_json: str) -> ScenarioCatalog:
        return cls(canonical_json)

    def to_bytes(self) -> bytes:
        return self.canonical_json.encode("ascii")

    def to_mapping(self) -> dict[str, object]:
        return json.loads(self.canonical_json)

    @property
    def catalog_sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()


def build_scenario_catalog() -> ScenarioCatalog:
    """Build one independently owned exact canonical catalog snapshot."""
    return ScenarioCatalog(_canonical_bytes(_catalog_mapping()).decode("ascii"))


def _run_plan_mapping(selection: ScenarioSelection, manifest: BaselineRunManifest) -> OrderedDict[str, object]:
    spec = resolve_scenario(selection.scenario_id)
    evidence = spec.evidence
    baseline = manifest.to_mapping()
    return OrderedDict((
        ("schema_version", RUN_PLAN_SCHEMA_VERSION),
        ("catalog_sha256", build_scenario_catalog().catalog_sha256),
        ("manifest_sha256", manifest.manifest_sha256),
        ("product_commit", baseline["product_commit"]),
        ("platform", baseline["platform"]),
        ("sensor", baseline["sensor"]),
        ("baseline_id", baseline["baseline_id"]),
        ("scenario_id", selection.scenario_id.value),
        ("seed", selection.seed),
        ("actor_audit_status", baseline["actor_audit_status"]),
        ("baseline_execution_status", baseline["execution_status"]),
        ("execution_domain", spec.execution_domain.value),
        ("requirement_class", spec.requirement_class.value),
        ("scenario_readiness", evidence.readiness.value),
        ("scenario_provider", evidence.provider),
        ("scenario_evidence_sha256", evidence.evidence_sha256),
        ("scenario_unavailability_reason", evidence.unavailability_reason),
        ("forwarded_actor_fields", list(spec.forwarded_actor_fields)),
        ("required_run_log_fields", list(spec.required_run_log_fields)),
        ("metric_obligations", list(spec.metric_obligations)),
        ("executable", False),
    ))


def _validate_run_plan_mapping(mapping: OrderedDict[str, object], manifest: BaselineRunManifest) -> None:
    if tuple(mapping) != RUN_PLAN_FIELD_ORDER:
        raise ValueError("run-plan keys are missing, extra, or reordered")
    scenario_id = mapping["scenario_id"]
    seed = mapping["seed"]
    selection = select_scenario(scenario_id=scenario_id, seed=seed)
    if mapping != _run_plan_mapping(selection, manifest):
        raise ValueError("run-plan JSON does not bind the exact catalog selection and manifest")


@dataclass(frozen=True, slots=True)
class ScenarioRunPlan:
    canonical_json: str
    manifest: BaselineRunManifest

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ScenarioRunPlan must not be subclassed")

    def __post_init__(self) -> None:
        if type(self.manifest) is not BaselineRunManifest:
            raise TypeError("manifest must be an exact BaselineRunManifest")
        owned_manifest = BaselineRunManifest.from_json(self.manifest.canonical_json)
        mapping = _decode_canonical_json(self.canonical_json)
        _validate_run_plan_mapping(mapping, owned_manifest)
        object.__setattr__(self, "manifest", owned_manifest)

    @classmethod
    def from_json(cls, canonical_json: str, manifest: BaselineRunManifest) -> ScenarioRunPlan:
        return cls(canonical_json, manifest)

    def to_bytes(self) -> bytes:
        return self.canonical_json.encode("ascii")

    def to_mapping(self) -> dict[str, object]:
        return json.loads(self.canonical_json)

    @property
    def run_plan_sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()


def bind_scenario_run_plan(
    *, selection: ScenarioSelection, manifest: BaselineRunManifest,
) -> ScenarioRunPlan:
    """Bind exact identities without changing readiness or exposing execution."""
    if type(selection) is not ScenarioSelection:
        raise TypeError("selection must be an exact ScenarioSelection")
    if type(manifest) is not BaselineRunManifest:
        raise TypeError("manifest must be an exact BaselineRunManifest")
    owned_manifest = BaselineRunManifest.from_json(manifest.canonical_json)
    baseline = owned_manifest.to_mapping()
    if baseline["scenario_id"] != selection.scenario_id.value or baseline["seed"] != selection.seed:
        raise ValueError("manifest scenario_id and seed must equal the exact selection")
    if baseline["platform"] != PLATFORM_ID or baseline["sensor"] != SENSOR_ID:
        raise ValueError("manifest platform or sensor violates the locked identity")
    mapping = _run_plan_mapping(selection, owned_manifest)
    return ScenarioRunPlan(_canonical_bytes(mapping).decode("ascii"), owned_manifest)


__all__ = [
    "ACCEPTED_METRIC_NAMES", "CATALOG_FIELD_ORDER", "CATALOG_SCHEMA_VERSION",
    "ExecutionDomain", "FORWARDED_ACTOR_FIELDS", "GENERATOR_PRIVILEGED_FIELDS",
    "REAL_ROBOT_REPEATABILITY_METRIC", "REQUIRED_RUN_LOG_FIELDS",
    "RUN_PLAN_FIELD_ORDER", "RUN_PLAN_SCHEMA_VERSION", "RequirementClass",
    "SCENARIO_EVIDENCE_FIELD_ORDER", "SCENARIO_IDS", "SCENARIO_ROW_FIELD_ORDER",
    "SCENARIO_SELECTION_FIELD_ORDER", "SCENARIO_SPEC_FIELD_ORDER", "ScenarioCatalog",
    "ScenarioEvidence", "ScenarioId", "ScenarioReadiness", "ScenarioRunPlan",
    "ScenarioSelection", "ScenarioSpec", "bind_scenario_run_plan",
    "build_scenario_catalog", "resolve_scenario", "scenario_registry",
    "select_scenario",
]
