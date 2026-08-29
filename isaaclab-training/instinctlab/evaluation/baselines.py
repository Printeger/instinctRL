"""Immutable B0-B8 identity, capability, and run-manifest contracts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import re
from typing import Final


MANIFEST_SCHEMA_VERSION: Final = "m7.baseline-run-manifest.v1"
PLATFORM_ID: Final = "TASLAB_UAV"
SENSOR_ID: Final = "Livox MID360"


class BaselineId(str, Enum):
    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    B4 = "B4"
    B5 = "B5"
    B6 = "B6"
    B7 = "B7"
    B8 = "B8"


class ExecutionFamily(str, Enum):
    DIRECT_CONTROL = "direct_control"
    LEARNED_GOVERNOR = "learned_governor"
    DETERMINISTIC_FILTER = "deterministic_filter"
    HEURISTIC_FILTER = "heuristic_filter"
    MODEL_BASED_FILTER = "model_based_filter"


class PrivilegeClass(str, Enum):
    ACTOR_CLEAN = "actor_clean"
    PRIVILEGED_UPPER_BOUND = "evaluation_only_privileged_upper_bound"


class ReadinessState(str, Enum):
    READY = "READY"
    NOT_READY = "NOT_READY"


class ActorAuditStatus(str, Enum):
    PASS = "PASS"
    NOT_RUN = "NOT_RUN"


class ExecutionStatus(str, Enum):
    READY = "READY"
    NOT_READY = "NOT_READY"
    EXECUTED = "EXECUTED"
    SKIPPED = "SKIPPED"


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _require_token(name: str, value: object) -> str:
    if type(value) is not str or _TOKEN_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a nonempty canonical ASCII token")
    return value


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _require_string_tuple(name: str, value: object, *, allow_empty: bool = True) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable ordered tuple")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    if any(type(item) is not str or not item for item in value):
        raise TypeError(f"{name} entries must be nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")
    return value


@dataclass(frozen=True)
class ComponentToggles:
    history: bool
    imu: bool
    anchor: bool
    ics: bool

    def __post_init__(self) -> None:
        if any(type(getattr(self, item.name)) is not bool for item in fields(self)):
            raise TypeError("component toggles must be exact booleans")


@dataclass(frozen=True)
class CapabilityRequirement:
    name: str
    accepted_provider: str | None
    accepted_evidence_sha256: str | None

    def __post_init__(self) -> None:
        _require_token("capability name", self.name)
        if (self.accepted_provider is None) != (self.accepted_evidence_sha256 is None):
            raise ValueError("accepted capability provider and hash must be present together")
        if self.accepted_provider is not None:
            _require_token("accepted capability provider", self.accepted_provider)
            _require_sha256("accepted capability evidence", self.accepted_evidence_sha256)


@dataclass(frozen=True)
class CapabilityBinding:
    name: str
    provider: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        _require_token("capability name", self.name)
        _require_token("capability provider", self.provider)
        _require_sha256("capability evidence", self.evidence_sha256)


@dataclass(frozen=True)
class BaselineSpec:
    baseline_id: BaselineId
    config_switch: str
    execution_family: ExecutionFamily
    description: str
    contribution_tested: str
    deployable: bool
    privilege_class: PrivilegeClass
    actor_input_fields: tuple[str, ...]
    evaluation_only_fields: tuple[str, ...]
    component_toggles: ComponentToggles
    capability_requirements: tuple[CapabilityRequirement, ...]
    optional: bool
    readiness: ReadinessState

    def __post_init__(self) -> None:
        if type(self.baseline_id) is not BaselineId:
            raise TypeError("baseline_id must be BaselineId")
        _require_token("config_switch", self.config_switch)
        if type(self.execution_family) is not ExecutionFamily:
            raise TypeError("execution_family must be ExecutionFamily")
        for name in ("description", "contribution_tested"):
            value = getattr(self, name)
            if type(value) is not str or not value or any(ord(char) < 32 for char in value):
                raise ValueError(f"{name} must be a nonempty control-free string")
        if type(self.deployable) is not bool or type(self.optional) is not bool:
            raise TypeError("deployable and optional must be exact booleans")
        if type(self.privilege_class) is not PrivilegeClass:
            raise TypeError("privilege_class must be PrivilegeClass")
        _require_string_tuple("actor_input_fields", self.actor_input_fields, allow_empty=False)
        _require_string_tuple("evaluation_only_fields", self.evaluation_only_fields, allow_empty=False)
        if type(self.component_toggles) is not ComponentToggles:
            raise TypeError("component_toggles must be ComponentToggles")
        if type(self.capability_requirements) is not tuple or not self.capability_requirements:
            raise TypeError("capability_requirements must be a nonempty immutable tuple")
        if any(type(item) is not CapabilityRequirement for item in self.capability_requirements):
            raise TypeError("capability_requirements entries must be CapabilityRequirement")
        names = tuple(item.name for item in self.capability_requirements)
        if len(names) != len(set(names)):
            raise ValueError("capability requirement names must not duplicate")
        if type(self.readiness) is not ReadinessState:
            raise TypeError("readiness must be ReadinessState")


BASELINE_SPEC_FIELD_ORDER: Final = tuple(item.name for item in fields(BaselineSpec))
CAPABILITY_BINDING_FIELD_ORDER: Final = tuple(item.name for item in fields(CapabilityBinding))
COMPONENT_TOGGLE_FIELD_ORDER: Final = tuple(item.name for item in fields(ComponentToggles))

FULL_ACTOR_INPUT_FIELDS: Final = (
    "ranges_m",
    "validity",
    "sensor_frame_time_s",
    "policy_time_s",
    "body_angular_velocity_rad_s",
    "gravity_direction_body",
    "commanded_body_velocity_m_s",
    "previous_issued_body_velocity_m_s",
)

EVALUATION_ONLY_FIELDS: Final = (
    "commanded_velocity_b_m_s",
    "actual_velocity_b_m_s",
    "issued_velocity_b_m_s",
    "position_w_m",
    "station_reference_position_w_m",
    "anchor_error_m",
    "anchor_valid",
    "minimum_clearance_m",
    "clearance_valid",
    "collision",
    "ics_beta",
    "ics_emergency",
    "ics_worst_margin_m_s",
    "terminated",
    "truncated",
    "termination_reason",
    "termination_collision",
    "termination_below_height",
    "termination_above_height",
    "termination_timeout",
    "observability_value",
    "observability_valid",
    "observability_source",
)

_DEPLOYABLE_ACTOR_FIELDS = frozenset(FULL_ACTOR_INPUT_FIELDS + ("minimum_range_m",))
_B0_REQUIREMENTS = (
    CapabilityRequirement(
        "m5.b0_direct_velocity",
        "instinctlab.control.governor.b0_governor",
        "cd5cf6c73578bc69dc5f696ac6d294d9d4c8db0baaf12265e6715646554b7197",
    ),
    CapabilityRequirement(
        "m5.b0_body_to_world",
        "instinctlab.control.command_adapter.rotate_body_to_world",
        "a79ea8d88fb0391762a9efcebbfa783d5287ee1393d3b6e2757812032c32991d",
    ),
    CapabilityRequirement(
        "m7.b0_direct_adapter",
        "instinctlab.control.baseline_b0.execute_b0_direct_velocity",
        "5fa2b8d746eb231a1bb12f331fe1a76e8f988005e9a315e1dbf622e0bf1ed3d1",
    ),
    CapabilityRequirement(
        "m7.b0_manifest_execution",
        "instinctlab.evaluation.baseline_execution.execute_ready_b0",
        "db7b73ee7e9c4b39689814530363bafb6a1ef48c2513a0208c0ff29c7073c108",
    ),
    CapabilityRequirement(
        "m7.b0_taslab_controller_handoff",
        "instinctlab.evaluation.baseline_controller_execution.execute_ready_b0_controller",
        "72d104850f8c8fede59998f14661b5b7ea7cd2c113ec11218fe232337732ab52",
    ),
    CapabilityRequirement(
        "m7.b0_run_lifecycle",
        "instinctlab.evaluation.baseline_run.begin_ready_b0_run",
        "5210566e987b737657d30ba44075a4dc835e379eae438726449dbc76002ab847",
    ),
    CapabilityRequirement(
        "m7.b0_taslab_environment",
        "instinctlab.evaluation.baseline_environment.TaslabB0EvaluationEnv",
        "b1ecdaa8b7d01ff0f9dd49891c2266908cea5a2540e8a91c33b9626464ded584",
    ),
)


def _unaccepted(name: str) -> tuple[CapabilityRequirement, ...]:
    return (CapabilityRequirement(name, None, None),)


BASELINE_REGISTRY: Final = (
    BaselineSpec(
        BaselineId.B0, "direct_velocity", ExecutionFamily.DIRECT_CONTROL,
        "Direct velocity execution with no learned governor, anchor, or intervention.",
        "TASLAB velocity controller alone.", True, PrivilegeClass.ACTOR_CLEAN,
        ("commanded_body_velocity_m_s",), EVALUATION_ONLY_FIELDS,
        ComponentToggles(False, False, False, False), _B0_REQUIREMENTS, False,
        ReadinessState.READY,
    ),
    BaselineSpec(
        BaselineId.B1, "state_velocity_upper_bound", ExecutionFamily.LEARNED_GOVERNOR,
        "State/velocity-aided governor used only as a privileged diagnostic upper bound.",
        "Gap to the forbidden state-aided policy.", False,
        PrivilegeClass.PRIVILEGED_UPPER_BOUND,
        FULL_ACTOR_INPUT_FIELDS + ("explicit_body_velocity_m_s", "local_odometry"),
        EVALUATION_ONLY_FIELDS, ComponentToggles(True, True, True, True),
        _unaccepted("m7.b1_state_velocity_upper_bound_adapter"), False,
        ReadinessState.NOT_READY,
    ),
    BaselineSpec(
        BaselineId.B2, "no_history", ExecutionFamily.LEARNED_GOVERNOR,
        "Governor using one current MID360, IMU, and command frame without history.",
        "Value of temporal history for motion inference.", True, PrivilegeClass.ACTOR_CLEAN,
        FULL_ACTOR_INPUT_FIELDS[:-1], EVALUATION_ONLY_FIELDS,
        ComponentToggles(False, True, True, True), _unaccepted("m7.b2_no_history_adapter"),
        False, ReadinessState.NOT_READY,
    ),
    BaselineSpec(
        BaselineId.B3, "no_anchor", ExecutionFamily.LEARNED_GOVERNOR,
        "Full actor inputs with measurement-space anchor loss and features disabled.",
        "Measurement-space anchor contribution to station keeping.", True,
        PrivilegeClass.ACTOR_CLEAN, FULL_ACTOR_INPUT_FIELDS, EVALUATION_ONLY_FIELDS,
        ComponentToggles(True, True, False, True), _unaccepted("m7.b3_no_anchor_adapter"),
        False, ReadinessState.NOT_READY,
    ),
    BaselineSpec(
        BaselineId.B4, "no_imu", ExecutionFamily.LEARNED_GOVERNOR,
        "Governor using MID360 history, commands, and previous output without IMU cues.",
        "IMU contribution to short-horizon inference and robustness.", True,
        PrivilegeClass.ACTOR_CLEAN,
        ("ranges_m", "validity", "sensor_frame_time_s", "policy_time_s",
         "commanded_body_velocity_m_s", "previous_issued_body_velocity_m_s"),
        EVALUATION_ONLY_FIELDS, ComponentToggles(True, False, True, True),
        _unaccepted("m7.b4_no_imu_adapter"), False, ReadinessState.NOT_READY,
    ),
    BaselineSpec(
        BaselineId.B5, "no_ics", ExecutionFamily.LEARNED_GOVERNOR,
        "Full learned governor with ICS-inspired command attenuation disabled.",
        "ICS attenuation contribution to unsafe-command outcomes.", True,
        PrivilegeClass.ACTOR_CLEAN, FULL_ACTOR_INPUT_FIELDS, EVALUATION_ONLY_FIELDS,
        ComponentToggles(True, True, True, False), _unaccepted("m7.b5_no_ics_adapter"),
        False, ReadinessState.NOT_READY,
    ),
    BaselineSpec(
        BaselineId.B6, "distance_only_filter", ExecutionFamily.DETERMINISTIC_FILTER,
        "Deterministic command attenuation based only on minimum range.",
        "Distance-only reactivity compared with braking feasibility.", True,
        PrivilegeClass.ACTOR_CLEAN,
        ("minimum_range_m", "commanded_body_velocity_m_s"), EVALUATION_ONLY_FIELDS,
        ComponentToggles(False, False, False, False),
        _unaccepted("m7.b6_distance_only_filter"), False, ReadinessState.NOT_READY,
    ),
    BaselineSpec(
        BaselineId.B7, "braking_heuristic", ExecutionFamily.HEURISTIC_FILTER,
        "Non-learned braking-feasibility attenuation using MID360 range-rate history.",
        "Gain from learning beyond the hand-designed braking rule.", True,
        PrivilegeClass.ACTOR_CLEAN,
        ("ranges_m", "validity", "sensor_frame_time_s", "policy_time_s",
         "commanded_body_velocity_m_s"),
        EVALUATION_ONLY_FIELDS, ComponentToggles(True, False, False, True),
        _unaccepted("m7.b7_braking_heuristic"), False, ReadinessState.NOT_READY,
    ),
    BaselineSpec(
        BaselineId.B8, "simplified_cbf_vo_optional", ExecutionFamily.MODEL_BASED_FILTER,
        "Optional local model-based CBF/VO-style command filter.",
        "Comparison with a conventional local safety-filter style baseline.", True,
        PrivilegeClass.ACTOR_CLEAN,
        ("ranges_m", "validity", "sensor_frame_time_s", "policy_time_s",
         "commanded_body_velocity_m_s"),
        EVALUATION_ONLY_FIELDS, ComponentToggles(False, False, False, False),
        _unaccepted("m7.b8_simplified_cbf_vo_filter"), True, ReadinessState.NOT_READY,
    ),
)

BASELINE_IDS: Final = tuple(item.value for item in BaselineId)
BASELINE_CONFIG_SWITCHES: Final = tuple(item.config_switch for item in BASELINE_REGISTRY)


def validate_baseline_registry(registry: tuple[BaselineSpec, ...]) -> None:
    """Validate one complete canonical registry without normalization or fallback."""
    if type(registry) is not tuple:
        raise TypeError("registry must be an immutable ordered tuple")
    if len(registry) != len(BASELINE_IDS) or any(type(row) is not BaselineSpec for row in registry):
        raise ValueError("registry must contain exactly nine BaselineSpec rows")
    ids = tuple(row.baseline_id.value for row in registry)
    switches = tuple(row.config_switch for row in registry)
    if ids != BASELINE_IDS:
        raise ValueError(f"registry IDs must be exactly ordered as {BASELINE_IDS}")
    expected_switches = (
        "direct_velocity", "state_velocity_upper_bound", "no_history", "no_anchor",
        "no_imu", "no_ics", "distance_only_filter", "braking_heuristic",
        "simplified_cbf_vo_optional",
    )
    if switches != expected_switches or len(set(switches)) != len(switches):
        raise ValueError("registry config switches must be exact, ordered, and unique")
    capability_names: set[str] = set()
    accepted_providers: set[str] = set()
    for row in registry:
        if row.optional != (row.baseline_id is BaselineId.B8):
            raise ValueError("only B8 may be optional")
        if row.baseline_id is BaselineId.B1:
            if row.deployable or row.privilege_class is not PrivilegeClass.PRIVILEGED_UPPER_BOUND:
                raise ValueError("B1 must be a nondeployable privileged upper bound")
        elif not row.deployable or row.privilege_class is not PrivilegeClass.ACTOR_CLEAN:
            raise ValueError("every non-B1 row must be deployable and actor-clean")
        if row.deployable and not set(row.actor_input_fields).issubset(_DEPLOYABLE_ACTOR_FIELDS):
            raise ValueError(f"{row.baseline_id.value} contains a forbidden deployable actor field")
        accepted = all(
            item.accepted_provider is not None and item.accepted_evidence_sha256 is not None
            for item in row.capability_requirements
        )
        if (row.readiness is ReadinessState.READY) != accepted:
            raise ValueError("readiness must equal accepted capability evidence")
        if (row.baseline_id is BaselineId.B0) != (row.readiness is ReadinessState.READY):
            raise ValueError("only accepted B0 may currently be READY")
        for requirement in row.capability_requirements:
            if requirement.name in capability_names:
                raise ValueError("capability requirement names must be globally unique")
            capability_names.add(requirement.name)
            if requirement.accepted_provider is not None:
                if requirement.accepted_provider in accepted_providers:
                    raise ValueError("accepted capability providers must be globally unique")
                accepted_providers.add(requirement.accepted_provider)
    for row, canonical_row in zip(registry, BASELINE_REGISTRY, strict=True):
        for field in fields(BaselineSpec):
            if getattr(row, field.name) != getattr(canonical_row, field.name):
                raise ValueError(
                    f"registry {row.baseline_id.value}.{field.name} does not match "
                    "the canonical baseline row"
                )


validate_baseline_registry(BASELINE_REGISTRY)


def resolve_baseline(baseline_id: BaselineId | str) -> BaselineSpec:
    """Resolve only an exact B0-B8 ID; aliases and normalization are forbidden."""
    if type(baseline_id) is BaselineId:
        exact = baseline_id.value
    elif type(baseline_id) is str:
        exact = baseline_id
    else:
        raise TypeError("baseline_id must be BaselineId or an exact string")
    for row in BASELINE_REGISTRY:
        if row.baseline_id.value == exact:
            return row
    raise KeyError(f"unknown exact baseline ID: {exact!r}")


def resolve_config_switch(config_switch: str) -> BaselineSpec:
    """Resolve only an exact canonical config value without aliases."""
    if type(config_switch) is not str:
        raise TypeError("config_switch must be an exact string")
    for row in BASELINE_REGISTRY:
        if row.config_switch == config_switch:
            return row
    raise KeyError(f"unknown exact baseline config switch: {config_switch!r}")


def validate_capability_bindings(
    baseline: BaselineSpec,
    bindings: tuple[CapabilityBinding, ...],
    execution_status: ExecutionStatus,
) -> None:
    """Reject unsupported, aliased, duplicate, reordered, or false-ready claims."""
    if type(baseline) is not BaselineSpec:
        raise TypeError("baseline must be BaselineSpec")
    if type(bindings) is not tuple:
        raise TypeError("capability bindings must be an immutable ordered tuple")
    if any(type(binding) is not CapabilityBinding for binding in bindings):
        raise TypeError("capability bindings must contain only CapabilityBinding")
    if len({id(binding) for binding in bindings}) != len(bindings):
        raise ValueError("capability binding objects must not alias")
    names = tuple(binding.name for binding in bindings)
    providers = tuple(binding.provider for binding in bindings)
    if len(names) != len(set(names)) or len(providers) != len(set(providers)):
        raise ValueError("capability names and providers must not duplicate")
    if type(execution_status) is not ExecutionStatus:
        raise TypeError("execution_status must be ExecutionStatus")
    ready_claim = execution_status in (ExecutionStatus.READY, ExecutionStatus.EXECUTED)
    if not ready_claim:
        if bindings:
            raise ValueError("NOT_READY or SKIPPED manifests must not claim capability bindings")
        return
    if baseline.readiness is not ReadinessState.READY:
        raise ValueError(f"{baseline.baseline_id.value} has no accepted executable capability")
    expected_names = tuple(item.name for item in baseline.capability_requirements)
    if names != expected_names:
        raise ValueError(f"capability bindings must be exactly ordered as {expected_names}")
    for requirement, binding in zip(baseline.capability_requirements, bindings, strict=True):
        if binding.provider != requirement.accepted_provider or (
            binding.evidence_sha256 != requirement.accepted_evidence_sha256
        ):
            raise ValueError(f"capability binding for {binding.name} does not match accepted evidence")


MANIFEST_PAYLOAD_FIELD_ORDER: Final = (
    "schema_version", "product_commit", "platform", "sensor", "baseline_id",
    "config_switch", "execution_family", "contribution_tested", "deployable",
    "privilege_class", "actor_input_fields", "evaluation_only_fields",
    "component_toggles", "capability_bindings", "scenario_id", "seed",
    "actor_audit_status", "execution_status", "skip_reason",
)
MANIFEST_FIELD_ORDER: Final = MANIFEST_PAYLOAD_FIELD_ORDER + ("manifest_sha256",)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("ascii")


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _unique_ordered_object(pairs: list[tuple[str, object]]) -> OrderedDict[str, object]:
    output: OrderedDict[str, object] = OrderedDict()
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        output[key] = value
    return output


def _decode_canonical_json(canonical_json: str) -> OrderedDict[str, object]:
    if type(canonical_json) is not str or not canonical_json:
        raise TypeError("canonical_json must be a nonempty string")
    if any(ord(char) > 127 for char in canonical_json):
        raise ValueError("canonical_json must contain ASCII bytes only")
    decoded = json.loads(
        canonical_json, object_pairs_hook=_unique_ordered_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(decoded, OrderedDict):
        raise ValueError("manifest JSON root must be an object")
    if _canonical_json_bytes(decoded).decode("ascii") != canonical_json:
        raise ValueError("manifest JSON must use the exact canonical byte encoding")
    return decoded


def _binding_mapping(binding: CapabilityBinding) -> OrderedDict[str, object]:
    return OrderedDict(
        (("name", binding.name), ("provider", binding.provider),
         ("evidence_sha256", binding.evidence_sha256))
    )


def _component_mapping(toggles: ComponentToggles) -> OrderedDict[str, object]:
    return OrderedDict((item.name, getattr(toggles, item.name)) for item in fields(toggles))


def _validate_status_contract(
    baseline: BaselineSpec,
    bindings: tuple[CapabilityBinding, ...],
    actor_audit_status: ActorAuditStatus,
    execution_status: ExecutionStatus,
    skip_reason: str | None,
) -> None:
    validate_capability_bindings(baseline, bindings, execution_status)
    if type(actor_audit_status) is not ActorAuditStatus:
        raise TypeError("actor_audit_status must be ActorAuditStatus")
    if execution_status in (ExecutionStatus.READY, ExecutionStatus.EXECUTED):
        if actor_audit_status is not ActorAuditStatus.PASS:
            raise ValueError("READY and EXECUTED require actor audit PASS")
        if skip_reason is not None:
            raise ValueError("READY and EXECUTED cannot contain a skip reason")
        return
    if actor_audit_status is not ActorAuditStatus.NOT_RUN:
        raise ValueError("NOT_READY and SKIPPED require actor audit NOT_RUN")
    if execution_status is ExecutionStatus.SKIPPED:
        if not baseline.optional:
            raise ValueError("only an optional baseline may be SKIPPED")
        _require_token("skip_reason", skip_reason)
        return
    if baseline.optional:
        raise ValueError("unavailable B8 must be explicitly SKIPPED with a reason")
    if skip_reason is not None:
        raise ValueError("NOT_READY cannot contain a skip reason")


def _manifest_payload(
    baseline: BaselineSpec,
    product_commit: str,
    bindings: tuple[CapabilityBinding, ...],
    scenario_id: str,
    seed: int,
    actor_audit_status: ActorAuditStatus,
    execution_status: ExecutionStatus,
    skip_reason: str | None,
) -> OrderedDict[str, object]:
    return OrderedDict(
        (
            ("schema_version", MANIFEST_SCHEMA_VERSION),
            ("product_commit", product_commit),
            ("platform", PLATFORM_ID),
            ("sensor", SENSOR_ID),
            ("baseline_id", baseline.baseline_id.value),
            ("config_switch", baseline.config_switch),
            ("execution_family", baseline.execution_family.value),
            ("contribution_tested", baseline.contribution_tested),
            ("deployable", baseline.deployable),
            ("privilege_class", baseline.privilege_class.value),
            ("actor_input_fields", list(baseline.actor_input_fields)),
            ("evaluation_only_fields", list(baseline.evaluation_only_fields)),
            ("component_toggles", _component_mapping(baseline.component_toggles)),
            ("capability_bindings", [_binding_mapping(binding) for binding in bindings]),
            ("scenario_id", scenario_id),
            ("seed", seed),
            ("actor_audit_status", actor_audit_status.value),
            ("execution_status", execution_status.value),
            ("skip_reason", skip_reason),
        )
    )


def _validate_manifest_mapping(mapping: OrderedDict[str, object]) -> None:
    if tuple(mapping) != MANIFEST_FIELD_ORDER:
        raise ValueError(f"manifest keys must be exactly ordered as {MANIFEST_FIELD_ORDER}")
    if mapping["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unknown manifest schema_version")
    product_commit = mapping["product_commit"]
    if type(product_commit) is not str or _COMMIT_PATTERN.fullmatch(product_commit) is None:
        raise ValueError("product_commit must be an exact lowercase 40-hex Git commit")
    if mapping["platform"] != PLATFORM_ID or mapping["sensor"] != SENSOR_ID:
        raise ValueError("manifest platform or sensor violates the locked identity")
    baseline = resolve_baseline(mapping["baseline_id"])
    if resolve_config_switch(mapping["config_switch"]) is not baseline:
        raise ValueError("baseline ID and config switch do not identify the same row")
    if type(mapping["deployable"]) is not bool:
        raise TypeError("manifest deployable must be an exact JSON boolean")
    component_toggles = mapping["component_toggles"]
    if not isinstance(component_toggles, OrderedDict) or tuple(
        component_toggles
    ) != COMPONENT_TOGGLE_FIELD_ORDER:
        raise ValueError("component toggle keys are missing, extra, or reordered")
    for name in COMPONENT_TOGGLE_FIELD_ORDER:
        if type(component_toggles[name]) is not bool:
            raise TypeError(
                f"manifest component_toggles.{name} must be an exact JSON boolean"
            )
    exact_values = {
        "execution_family": baseline.execution_family.value,
        "contribution_tested": baseline.contribution_tested,
        "deployable": baseline.deployable,
        "privilege_class": baseline.privilege_class.value,
        "actor_input_fields": list(baseline.actor_input_fields),
        "evaluation_only_fields": list(baseline.evaluation_only_fields),
        "component_toggles": _component_mapping(baseline.component_toggles),
    }
    for name, expected in exact_values.items():
        if mapping[name] != expected:
            raise ValueError(f"manifest {name} does not match the canonical baseline row")
    raw_bindings = mapping["capability_bindings"]
    if type(raw_bindings) is not list:
        raise TypeError("manifest capability_bindings must be a JSON array")
    bindings: list[CapabilityBinding] = []
    for item in raw_bindings:
        if not isinstance(item, OrderedDict) or tuple(item) != CAPABILITY_BINDING_FIELD_ORDER:
            raise ValueError("capability binding keys are missing, extra, or reordered")
        bindings.append(CapabilityBinding(item["name"], item["provider"], item["evidence_sha256"]))
    _require_token("scenario_id", mapping["scenario_id"])
    seed = mapping["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    try:
        actor_status = ActorAuditStatus(mapping["actor_audit_status"])
        execution_status = ExecutionStatus(mapping["execution_status"])
    except (TypeError, ValueError) as error:
        raise ValueError("manifest contains an unknown audit or execution status") from error
    _validate_status_contract(
        baseline, tuple(bindings), actor_status, execution_status, mapping["skip_reason"]
    )
    payload = OrderedDict((name, mapping[name]) for name in MANIFEST_PAYLOAD_FIELD_ORDER)
    expected_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if mapping["manifest_sha256"] != expected_hash:
        raise ValueError("manifest_sha256 does not bind the exact canonical payload bytes")


@dataclass(frozen=True)
class BaselineRunManifest:
    """Owned canonical JSON bytes with a self-verifying deterministic SHA-256."""

    canonical_json: str

    def __post_init__(self) -> None:
        _validate_manifest_mapping(_decode_canonical_json(self.canonical_json))

    @classmethod
    def from_json(cls, canonical_json: str) -> BaselineRunManifest:
        return cls(canonical_json)

    def to_bytes(self) -> bytes:
        return self.canonical_json.encode("ascii")

    def to_mapping(self) -> dict[str, object]:
        return json.loads(self.canonical_json)

    @property
    def manifest_sha256(self) -> str:
        return self.to_mapping()["manifest_sha256"]


def build_run_manifest(
    *,
    product_commit: str,
    baseline_id: BaselineId | str,
    config_switch: str,
    capability_bindings: tuple[CapabilityBinding, ...],
    scenario_id: str,
    seed: int,
    actor_audit_status: ActorAuditStatus,
    execution_status: ExecutionStatus,
    skip_reason: str | None,
) -> BaselineRunManifest:
    """Build one exact manifest without defaults, aliases, or readiness fallback."""
    if type(product_commit) is not str or _COMMIT_PATTERN.fullmatch(product_commit) is None:
        raise ValueError("product_commit must be an exact lowercase 40-hex Git commit")
    baseline = resolve_baseline(baseline_id)
    if resolve_config_switch(config_switch) is not baseline:
        raise ValueError("baseline ID and config switch do not identify the same row")
    if type(capability_bindings) is not tuple:
        raise TypeError("capability_bindings must be an immutable ordered tuple")
    _require_token("scenario_id", scenario_id)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    _validate_status_contract(
        baseline, capability_bindings, actor_audit_status, execution_status, skip_reason
    )
    payload = _manifest_payload(
        baseline, product_commit, capability_bindings, scenario_id, seed,
        actor_audit_status, execution_status, skip_reason,
    )
    manifest_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    manifest = OrderedDict(payload)
    manifest["manifest_sha256"] = manifest_hash
    return BaselineRunManifest(_canonical_json_bytes(manifest).decode("ascii"))


__all__ = [
    "ActorAuditStatus", "BASELINE_CONFIG_SWITCHES", "BASELINE_IDS", "BASELINE_REGISTRY",
    "BASELINE_SPEC_FIELD_ORDER", "BaselineId", "BaselineRunManifest", "BaselineSpec",
    "CAPABILITY_BINDING_FIELD_ORDER", "COMPONENT_TOGGLE_FIELD_ORDER", "CapabilityBinding",
    "CapabilityRequirement", "ComponentToggles", "EVALUATION_ONLY_FIELDS", "ExecutionFamily",
    "ExecutionStatus", "FULL_ACTOR_INPUT_FIELDS", "MANIFEST_FIELD_ORDER",
    "MANIFEST_PAYLOAD_FIELD_ORDER", "MANIFEST_SCHEMA_VERSION", "PLATFORM_ID",
    "PrivilegeClass", "ReadinessState", "SENSOR_ID", "build_run_manifest",
    "resolve_baseline", "resolve_config_switch", "validate_baseline_registry",
    "validate_capability_bindings",
]
