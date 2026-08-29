"""Exact command-only B0 environment-to-runtime evaluation collection."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import re

import torch

from .baseline_environment import B0EnvironmentResult
from .baselines import BaselineRunManifest
from .metrics import (
    EVALUATION_INPUT_KEYS,
    AggregateMetricResult,
    EvaluationMetricConfig,
    ObservabilitySource,
    aggregate_metrics,
    compute_step_metrics,
)
from .runtime import ExactRuntimeObservability


PRE_RESET_KEYS = (
    "reset_mask", "episode_length", "root_state_w", "joint_pos", "joint_vel",
    "control", "lee", "wrench", "reward", "task",
)
CONTROL_KEYS = (
    "alpha", "v_corr_b", "v_gov_b", "v_final_b", "v_final_w",
    "anchor_metrics", "ics_metrics",
)
ANCHOR_KEYS = (
    "active", "reset_code", "usable_fraction", "mean_absolute_error_m", "huber_loss",
)
ICS_KEYS = (
    "beta", "active_beam_count", "minimum_clearance_m", "worst_margin_m_s", "emergency",
)
TASK_KEYS = ("root_state_w", "actual_velocity_b", "exact_clearance_m", "collision", "termination")
TERMINATION_KEYS = ("terminated", "truncated", "done", "reason_code", "masks")
TERMINATION_MASK_KEYS = ("collision", "below_height", "above_height", "timeout")
OBSERVABILITY_DETAIL_KEYS = (
    "numerical_rank", "singular_values", "condition_number", "condition_valid",
    "weak_direction_w", "weak_direction_valid", "valid_ray_count", "total_ray_count",
    "valid_beam_fraction", "hit_max_error_m", "distance_max_error_m", "source_step",
    "source_reset_mask", "source_identity_sha256", "result_identity_sha256",
)
SOURCE_METADATA_KEYS = (
    "source", "source_shape", "record_shape", "dtype", "units", "frame",
    "validity", "ownership", "privilege",
)
PROVENANCE_KEYS = ("product_commit", "fixture_id", "generator_id", "seed")
RECORD_KEYS = (
    "schema_version", "platform", "sensor", "baseline_id", "claim", "actor_surface",
    "manifest", "receipt", "coverage", "evaluation_input_order", "source_manifest",
    "metric_config", "observability_source_counts", "metrics", "termination_counts",
    "provenance", "gates",
)
_SOURCE_SPEC = (
    ("commanded_velocity_b_m_s", ("collector.pending_command", "[N, 3]", "[N, 3]", "torch.float32", "m/s", "body", "exact detached finite command prepared before the sole environment step", "record-owned detached clone", "actor-approved command")),
    ("actual_velocity_b_m_s", ("pre_reset.task.actual_velocity_b", "[N, 3]", "[N, 3]", "torch.float32", "m/s", "body", "finite accepted same-state task producer", "record-owned detached clone", "evaluation-only privileged")),
    ("issued_velocity_b_m_s", ("pre_reset.control.v_final_b", "[N, 3]", "[N, 3]", "torch.float32", "m/s", "body", "finite accepted disabled-B0 controller producer", "record-owned detached clone", "evaluation-only diagnostic")),
    ("position_w_m", ("pre_reset.task.root_state_w[:,0:3]", "[N, 13] -> [N, 3]", "[N, 3]", "torch.float32", "m", "world", "same-state task root equals top-level pre_reset root", "record-owned detached clone", "evaluation-only privileged")),
    ("station_reference_position_w_m", ("collector.initial_critic[:,0:3]", "[N, 18] -> [N, 3]", "[N, 3]", "torch.float32", "m", "world", "fixed initial no-reset station reference", "record-owned clone of collector-owned state", "evaluation-only derived")),
    ("anchor_error_m", ("pre_reset.control.anchor_metrics.mean_absolute_error_m[:,0]", "[N, 1] -> [N]", "[N]", "torch.float32", "m", "body scan geometry", "exact zero while B0 anchor is disabled", "record-owned detached clone", "evaluation-only diagnostic")),
    ("anchor_valid", ("disabled B0 anchor contract", "scalar contract -> [N]", "[N]", "torch.bool", "boolean", "none", "always false", "record-owned constructed tensor", "evaluation-only validity")),
    ("minimum_clearance_m", ("pre_reset.task.exact_clearance_m[:,0]", "[N, 1] -> [N]", "[N]", "torch.float32", "m", "world geometry", "finite nonnegative accepted exact task producer", "record-owned detached clone", "evaluation-only privileged")),
    ("clearance_valid", ("accepted exact task producer availability", "scalar contract -> [N]", "[N]", "torch.bool", "boolean", "none", "always true", "record-owned constructed tensor", "evaluation-only validity")),
    ("collision", ("pre_reset.task.collision[:,0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "exact false in bounded no-reset mode", "record-owned detached clone", "evaluation-only privileged")),
    ("ics_beta", ("pre_reset.control.ics_metrics.beta[:,0]", "[N, 1] -> [N]", "[N]", "torch.float32", "ratio", "none", "exact one while B0 ICS is disabled", "record-owned detached clone", "evaluation-only diagnostic")),
    ("ics_emergency", ("pre_reset.control.ics_metrics.emergency[:,0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "exact false while B0 ICS is disabled", "record-owned derived tensor", "evaluation-only diagnostic")),
    ("ics_worst_margin_m_s", ("pre_reset.control.ics_metrics.worst_margin_m_s[:,0]", "[N, 1] -> [N]", "[N]", "torch.float32", "m/s", "body radial", "exact zero while B0 ICS is disabled", "record-owned detached clone", "evaluation-only diagnostic")),
    ("terminated", ("pre_reset.task.termination.terminated[:,0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "exact false", "record-owned detached clone", "evaluation-only termination")),
    ("truncated", ("pre_reset.task.termination.truncated[:,0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "exact false", "record-owned detached clone", "evaluation-only termination")),
    ("termination_reason", ("pre_reset.task.termination.reason_code[:,0]", "[N, 1] -> [N]", "[N]", "torch.int64", "enum", "none", "exact no-termination value zero", "record-owned detached clone", "evaluation-only termination")),
    ("termination_collision", ("pre_reset.task.termination.masks.collision[:,0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "exact false", "record-owned detached clone", "evaluation-only termination")),
    ("termination_below_height", ("pre_reset.task.termination.masks.below_height[:,0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "exact false", "record-owned detached clone", "evaluation-only termination")),
    ("termination_above_height", ("pre_reset.task.termination.masks.above_height[:,0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "exact false", "record-owned detached clone", "evaluation-only termination")),
    ("termination_timeout", ("pre_reset.task.termination.masks.timeout[:,0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "exact false", "record-owned detached clone", "evaluation-only termination")),
    ("observability_value", ("exact_observability.sigma_min", "[N]", "[N]", "torch.float32", "dimensionless", "world translational range Jacobian", "finite exact value including rank-deficient zero", "record-owned detached clone", "evaluation-only exact")),
    ("observability_valid", ("successful guarded exact acquisition", "scalar contract -> [N]", "[N]", "torch.bool", "boolean", "none", "always true", "record-owned constructed tensor", "evaluation-only validity")),
    ("observability_source", ("ObservabilitySource.EXACT", "scalar contract -> [N]", "[N]", "torch.int64", "enum", "none", "exact source classification", "record-owned constructed tensor", "evaluation-only exact")),
)
_SHA_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_ACTIVE = "ACTIVE"
_FINALIZED = "FINALIZED"
_COLLECTOR_TOKEN = object()


def _source_manifest() -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    return tuple(
        (name, tuple(zip(SOURCE_METADATA_KEYS, values, strict=True)))
        for name, values in _SOURCE_SPEC
    )


B0_SOURCE_MANIFEST = _source_manifest()


def _owns_exact_storage(value: torch.Tensor) -> bool:
    return (
        value._base is None
        and value.storage_offset() == 0
        and value.untyped_storage().nbytes() == value.numel() * value.element_size()
    )


def _ordered(name: str, value: object, keys: tuple[str, ...]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or tuple(value) != keys:
        raise ValueError(f"{name} keys must be exactly ordered as {keys}")
    return value


def _tensor(
    name: str,
    value: object,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    *,
    finite: bool = True,
) -> torch.Tensor:
    if type(value) is not torch.Tensor or value.shape != shape or value.dtype != dtype:
        raise TypeError(f"{name} must be an exact {dtype} tensor with shape {shape}")
    if value.device != device or value.requires_grad or value.grad_fn is not None:
        raise ValueError(f"{name} must be detached on {device}")
    if not _owns_exact_storage(value):
        raise ValueError(f"{name} must own one exact contiguous allocation")
    if finite and value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")
    return value


def _reject_aliases(values: Mapping[str, torch.Tensor]) -> None:
    owners: dict[tuple[torch.device, int], str] = {}
    for name, value in values.items():
        key = (value.device, value.untyped_storage().data_ptr())
        if key in owners:
            raise ValueError(f"{owners[key]} and {name} alias storage")
        owners[key] = name


def _row_hashes(
    tensors: Mapping[str, torch.Tensor],
    constants: tuple[tuple[str, str], ...] = (),
) -> torch.Tensor:
    if not tensors:
        raise ValueError("identity tensors must not be empty")
    first = next(iter(tensors.values()))
    batch = first.shape[0]
    rows = []
    for index in range(batch):
        digest = hashlib.sha256()
        for name, value in tensors.items():
            if value.shape[0] != batch:
                raise ValueError("identity tensors must share one leading batch")
            item = value[index].detach().contiguous().cpu()
            digest.update(name.encode("ascii"))
            digest.update(f"{item.dtype}|{tuple(item.shape)}|".encode("ascii"))
            digest.update(item.numpy().tobytes(order="C"))
        for name, value in constants:
            digest.update(name.encode("ascii")); digest.update(value.encode("ascii"))
        rows.append(torch.tensor(tuple(digest.digest()), dtype=torch.uint8, device=first.device))
    return torch.stack(rows).detach().clone()


def _metric_record(aggregate: AggregateMetricResult) -> tuple[tuple[str, tuple[tuple[str, object], ...]], ...]:
    return tuple(
        (metric.name, (
            ("reduction", metric.reduction), ("unit", metric.unit),
            ("available", metric.available), ("value", metric.value),
            ("numerator", metric.numerator), ("denominator", metric.denominator),
            ("valid_count", metric.valid_count), ("total_count", metric.total_count),
        ))
        for metric in aggregate.metrics
    )


@dataclass(frozen=True, slots=True)
class B0RuntimeEvaluationProvenance:
    product_commit: str
    fixture_id: str
    generator_id: str
    seed: int

    def __post_init__(self) -> None:
        if type(self.product_commit) is not str or _COMMIT_PATTERN.fullmatch(self.product_commit) is None:
            raise ValueError("product_commit must be an exact lowercase 40-hex Git commit")
        for name in ("fixture_id", "generator_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or any(ord(char) < 32 for char in value):
                raise ValueError(f"{name} must be one nonempty printable string")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative exact integer")


@dataclass(frozen=True, slots=True)
class B0RuntimeEvaluationConfig:
    expected_policy_steps: int
    num_envs: int
    rays_per_environment: int
    metric_config: EvaluationMetricConfig
    provenance: B0RuntimeEvaluationProvenance

    def __post_init__(self) -> None:
        if type(self.expected_policy_steps) is not int or self.expected_policy_steps != 8:
            raise ValueError("B0 runtime evaluation requires exactly eight policy steps")
        if type(self.num_envs) is not int or self.num_envs != 2:
            raise ValueError("B0 runtime evaluation requires exactly two environments")
        if type(self.rays_per_environment) is not int or self.rays_per_environment != 21240:
            raise ValueError("B0 runtime evaluation requires exactly 21240 rays per environment")
        if type(self.metric_config) is not EvaluationMetricConfig or self.metric_config != EvaluationMetricConfig():
            raise ValueError("metric_config must be the exact accepted M7 configuration")
        if type(self.provenance) is not B0RuntimeEvaluationProvenance:
            raise TypeError("provenance must be an exact B0RuntimeEvaluationProvenance")
        metric = EvaluationMetricConfig(
            self.metric_config.schema_version,
            self.metric_config.null_command_threshold_m_s,
            self.metric_config.full_compliance_beta,
        )
        provenance = B0RuntimeEvaluationProvenance(
            self.provenance.product_commit, self.provenance.fixture_id,
            self.provenance.generator_id, self.provenance.seed,
        )
        object.__setattr__(self, "metric_config", metric)
        object.__setattr__(self, "provenance", provenance)


@dataclass(frozen=True, slots=True)
class B0RuntimeEvaluationContract:
    num_envs: int
    policy_steps: int
    metric_samples: int
    physics_steps: int
    lifecycle_calls: int
    wrench_applications: int
    environment_samples: int
    rays_per_environment: int
    ray_samples: int


@dataclass(frozen=True, slots=True)
class B0RuntimeEvaluationRecord:
    canonical_json: str

    def __post_init__(self) -> None:
        if type(self.canonical_json) is not str or not self.canonical_json:
            raise TypeError("canonical_json must be a nonempty exact string")
        value = json.loads(self.canonical_json, object_pairs_hook=OrderedDict)
        if tuple(value) != RECORD_KEYS:
            raise ValueError("canonical record keys are missing, extra, or reordered")
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        if encoded != self.canonical_json:
            raise ValueError("record must use exact canonical JSON bytes")

    def to_mapping(self) -> dict[str, object]:
        return json.loads(self.canonical_json)


@dataclass(frozen=True, slots=True)
class B0RuntimeEvaluationResult:
    schema_version: str
    contract: B0RuntimeEvaluationContract
    ready_manifest_sha256: str
    executed_manifest_sha256: str
    trajectory_sha256: str
    command_identity_sha256: torch.Tensor
    source_identity_sha256: torch.Tensor
    result_identity_sha256: torch.Tensor
    evaluation_inputs: tuple[tuple[str, torch.Tensor], ...]
    observability_details: tuple[tuple[str, torch.Tensor], ...]
    aggregate_metrics: AggregateMetricResult
    record: B0RuntimeEvaluationRecord
    gates: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if self.schema_version != "m7.b0-runtime-evaluation-result.v1":
            raise ValueError("unknown B0 runtime evaluation result schema")
        if type(self.contract) is not B0RuntimeEvaluationContract:
            raise TypeError("contract must be an exact B0RuntimeEvaluationContract")
        for name in ("ready_manifest_sha256", "executed_manifest_sha256", "trajectory_sha256"):
            if type(getattr(self, name)) is not str or _SHA_PATTERN.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be an exact lowercase SHA-256")
        if type(self.record) is not B0RuntimeEvaluationRecord:
            raise TypeError("record must be an exact B0RuntimeEvaluationRecord")


def _owned_ready_manifest(manifest: BaselineRunManifest) -> BaselineRunManifest:
    if type(manifest) is not BaselineRunManifest:
        raise TypeError("manifest must be an exact BaselineRunManifest")
    owned = BaselineRunManifest.from_json(manifest.canonical_json)
    mapping = owned.to_mapping()
    if mapping["baseline_id"] != "B0" or mapping["config_switch"] != "direct_velocity":
        raise ValueError("manifest must identify the exact B0 direct-velocity row")
    if mapping["execution_status"] != "READY" or mapping["actor_audit_status"] != "PASS":
        raise ValueError("B0 runtime evaluation requires one actor-audited READY manifest")
    if mapping["actor_input_fields"] != ["commanded_body_velocity_m_s"]:
        raise ValueError("B0 manifest actor surface must remain command-only")
    if mapping["component_toggles"] != {
        "history": False, "imu": False, "anchor": False, "ics": False,
    }:
        raise ValueError("B0 manifest components must remain exactly disabled")
    if len(mapping["capability_bindings"]) != 8 or mapping["capability_bindings"][-1]["name"] != "m7.b0_runtime_evaluation":
        raise ValueError("B0 runtime evaluation requires the exact ordered eight-capability manifest")
    return owned


def _validate_exact_observability(
    value: object,
    batch: int,
    device: torch.device,
    expected_step: int,
    rays_per_environment: int,
) -> ExactRuntimeObservability:
    if type(value) is not ExactRuntimeObservability:
        raise TypeError("exact_observability must be an exact ExactRuntimeObservability")
    expected = {
        "numerical_rank": ((batch,), torch.long),
        "singular_values": ((batch, 3), torch.float32),
        "sigma_min": ((batch,), torch.float32),
        "condition_number": ((batch,), torch.float32),
        "condition_valid": ((batch,), torch.bool),
        "weak_direction_w": ((batch, 3), torch.float32),
        "weak_direction_valid": ((batch,), torch.bool),
        "valid_ray_count": ((batch,), torch.long),
        "total_ray_count": ((batch,), torch.long),
        "valid_beam_fraction": ((batch,), torch.float32),
        "hit_max_error_m": ((batch,), torch.float32),
        "distance_max_error_m": ((batch,), torch.float32),
        "source_step": ((batch,), torch.long),
        "source_reset_mask": ((batch,), torch.bool),
        "source_identity_sha256": ((batch, 32), torch.uint8),
        "result_identity_sha256": ((batch, 32), torch.uint8),
    }
    values = OrderedDict((field.name, getattr(value, field.name)) for field in fields(value))
    for name, tensor in values.items():
        shape, dtype = expected[name]
        _tensor(f"exact_observability.{name}", tensor, shape, dtype, device)
    _reject_aliases(values)
    if not torch.equal(value.source_step, torch.full_like(value.source_step, expected_step)):
        raise ValueError("exact observability source is stale, duplicated, skipped, or reordered")
    if bool(value.source_reset_mask.any().item()):
        raise ValueError("bounded B0 exact observability must bind an all-false reset mask")
    if bool(((value.numerical_rank < 0) | (value.numerical_rank > 3)).any().item()):
        raise ValueError("exact observability rank is outside [0,3]")
    if bool((value.singular_values < 0.0).any().item()) or not bool(
        (value.singular_values[:, 1:] >= value.singular_values[:, :-1]).all().item()
    ) or not torch.equal(value.sigma_min, value.singular_values[:, 0]):
        raise ValueError("exact observability spectrum is inconsistent")
    if not torch.equal(value.condition_valid, value.numerical_rank == 3):
        raise ValueError("exact observability condition validity is inconsistent")
    if bool((value.condition_number[~value.condition_valid] != 0.0).any().item()):
        raise ValueError("invalid condition numbers must be finite zero placeholders")
    if bool((value.weak_direction_w[~value.weak_direction_valid] != 0.0).any().item()):
        raise ValueError("invalid weak directions must be finite zero placeholders")
    if not torch.equal(value.total_ray_count, torch.full_like(value.total_ray_count, rays_per_environment)):
        raise ValueError("exact observability total-ray count is inconsistent")
    if bool(((value.valid_ray_count < 0) | (value.valid_ray_count > value.total_ray_count)).any().item()):
        raise ValueError("exact observability valid-ray count is inconsistent")
    fraction = value.valid_ray_count.to(torch.float32) / value.total_ray_count.to(torch.float32)
    if not torch.equal(value.valid_beam_fraction, fraction):
        raise ValueError("exact observability valid fraction is inconsistent")
    if bool((value.hit_max_error_m < 0.0).any().item()) or bool((value.distance_max_error_m < 0.0).any().item()):
        raise ValueError("exact observability errors must be nonnegative")
    sealed = OrderedDict((name, tensor) for name, tensor in values.items() if name != "result_identity_sha256")
    if not torch.equal(value.result_identity_sha256, _row_hashes(sealed)):
        raise ValueError("exact observability result identity is forged or mutated")
    return value


class B0RuntimeEvaluationCollector:
    """Factory-owned exact B0 command-to-evaluation state machine."""

    __slots__ = (
        "config", "_manifest", "_station_reference", "_state", "_pending_command",
        "_pending_snapshot", "_rows", "_details", "_command_identities",
        "_source_identities", "_result_identities", "_seen_sources",
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("B0RuntimeEvaluationCollector must not be subclassed")

    def __init__(
        self,
        manifest: BaselineRunManifest,
        initial_critic: torch.Tensor,
        config: B0RuntimeEvaluationConfig,
        token: object,
    ) -> None:
        if token is not _COLLECTOR_TOKEN:
            raise TypeError("collector must be created by begin_ready_b0_runtime_evaluation")
        if type(config) is not B0RuntimeEvaluationConfig:
            raise TypeError("config must be an exact B0RuntimeEvaluationConfig")
        if type(initial_critic) is not torch.Tensor:
            raise TypeError("initial_critic must be an exact torch.Tensor")
        owned_manifest = _owned_ready_manifest(manifest)
        critic = _tensor("initial_critic", initial_critic, (2, 18), torch.float32, initial_critic.device)
        manifest_mapping = owned_manifest.to_mapping()
        if config.provenance.product_commit != manifest_mapping["product_commit"] or config.provenance.seed != manifest_mapping["seed"]:
            raise ValueError("provenance product_commit/seed must bind the owned READY manifest")
        self.config = B0RuntimeEvaluationConfig(
            config.expected_policy_steps, config.num_envs, config.rays_per_environment,
            config.metric_config, config.provenance,
        )
        self._manifest = owned_manifest
        self._station_reference = critic[:, :3].detach().clone()
        self._state = _ACTIVE
        self._pending_command: torch.Tensor | None = None
        self._pending_snapshot: torch.Tensor | None = None
        self._rows: list[OrderedDict[str, torch.Tensor]] = []
        self._details: list[OrderedDict[str, torch.Tensor]] = []
        self._command_identities: list[torch.Tensor] = []
        self._source_identities: list[torch.Tensor] = []
        self._result_identities: list[torch.Tensor] = []
        self._seen_sources: set[bytes] = set()

    @property
    def state_token(self) -> str:
        return self._state

    @property
    def row_count(self) -> int:
        return len(self._rows)

    @property
    def has_pending_command(self) -> bool:
        return self._pending_command is not None

    def prepare_command(self, command: torch.Tensor) -> torch.Tensor:
        if type(self) is not B0RuntimeEvaluationCollector:
            raise TypeError("prepare_command requires an exact B0RuntimeEvaluationCollector")
        if self._state != _ACTIVE:
            raise RuntimeError("finalized B0 runtime evaluation cannot prepare a command")
        if len(self._rows) >= self.config.expected_policy_steps:
            raise RuntimeError("B0 runtime evaluation reached exact row coverage")
        if self._pending_command is not None:
            raise RuntimeError("one prepared command is already pending")
        value = _tensor("command", command, (2, 3), torch.float32, self._station_reference.device)
        station_pointer = self._station_reference.untyped_storage().data_ptr()
        if value.untyped_storage().data_ptr() == station_pointer:
            raise ValueError("command must not alias collector-private station state")
        staged = value.detach().clone()
        self._pending_command = staged
        self._pending_snapshot = staged.detach().clone()
        return staged

    def commit_transition(
        self,
        *,
        staged_command: torch.Tensor,
        pre_reset: Mapping[str, object],
        exact_observability: ExactRuntimeObservability,
        returned_critic: torch.Tensor,
        returned_episode_length: torch.Tensor,
    ) -> None:
        if type(self) is not B0RuntimeEvaluationCollector:
            raise TypeError("commit_transition requires an exact B0RuntimeEvaluationCollector")
        if self._state != _ACTIVE:
            raise RuntimeError("finalized B0 runtime evaluation cannot commit a row")
        if len(self._rows) >= self.config.expected_policy_steps:
            raise RuntimeError("B0 runtime evaluation row coverage overrun")
        if self._pending_command is None or self._pending_snapshot is None:
            raise RuntimeError("one prepared command is required before row commit")
        if staged_command is not self._pending_command:
            raise ValueError("row must consume the exact staged command returned by prepare_command")
        command = _tensor("staged_command", staged_command, (2, 3), torch.float32, self._station_reference.device)
        if not torch.equal(command, self._pending_snapshot):
            raise ValueError("staged command changed during the environment transition")

        batch, device = 2, command.device
        pre = _ordered("pre_reset", pre_reset, PRE_RESET_KEYS)
        control = _ordered("pre_reset.control", pre["control"], CONTROL_KEYS)
        anchor = _ordered("pre_reset.control.anchor_metrics", control["anchor_metrics"], ANCHOR_KEYS)
        ics = _ordered("pre_reset.control.ics_metrics", control["ics_metrics"], ICS_KEYS)
        task = _ordered("pre_reset.task", pre["task"], TASK_KEYS)
        termination = _ordered("pre_reset.task.termination", task["termination"], TERMINATION_KEYS)
        masks = _ordered("pre_reset.task.termination.masks", termination["masks"], TERMINATION_MASK_KEYS)
        for name in ("lee", "wrench", "reward"):
            if not isinstance(pre[name], Mapping):
                raise TypeError(f"pre_reset.{name} must remain an accepted detached mapping")

        reset_mask = _tensor("pre_reset.reset_mask", pre["reset_mask"], (batch,), torch.bool, device)
        episode = _tensor("pre_reset.episode_length", pre["episode_length"], (batch,), torch.long, device)
        root = _tensor("pre_reset.root_state_w", pre["root_state_w"], (batch, 13), torch.float32, device)
        task_root = _tensor("pre_reset.task.root_state_w", task["root_state_w"], (batch, 13), torch.float32, device)
        actual = _tensor("pre_reset.task.actual_velocity_b", task["actual_velocity_b"], (batch, 3), torch.float32, device)
        clearance = _tensor("pre_reset.task.exact_clearance_m", task["exact_clearance_m"], (batch, 1), torch.float32, device)
        collision = _tensor("pre_reset.task.collision", task["collision"], (batch, 1), torch.bool, device)
        alpha = _tensor("pre_reset.control.alpha", control["alpha"], (batch, 1), torch.float32, device)
        correction = _tensor("pre_reset.control.v_corr_b", control["v_corr_b"], (batch, 3), torch.float32, device)
        governed = _tensor("pre_reset.control.v_gov_b", control["v_gov_b"], (batch, 3), torch.float32, device)
        issued = _tensor("pre_reset.control.v_final_b", control["v_final_b"], (batch, 3), torch.float32, device)
        world = _tensor("pre_reset.control.v_final_w", control["v_final_w"], (batch, 3), torch.float32, device)
        anchor_active = _tensor("anchor.active", anchor["active"], (batch, 1), torch.float32, device)
        anchor_reset = _tensor("anchor.reset_code", anchor["reset_code"], (batch, 1), torch.long, device)
        anchor_usable = _tensor("anchor.usable_fraction", anchor["usable_fraction"], (batch, 1), torch.float32, device)
        anchor_error = _tensor("anchor.mean_absolute_error_m", anchor["mean_absolute_error_m"], (batch, 1), torch.float32, device)
        anchor_loss = _tensor("anchor.huber_loss", anchor["huber_loss"], (batch, 1), torch.float32, device)
        beta = _tensor("ics.beta", ics["beta"], (batch, 1), torch.float32, device)
        active_beams = _tensor("ics.active_beam_count", ics["active_beam_count"], (batch, 1), torch.float32, device)
        ics_clearance = _tensor("ics.minimum_clearance_m", ics["minimum_clearance_m"], (batch, 1), torch.float32, device)
        worst = _tensor("ics.worst_margin_m_s", ics["worst_margin_m_s"], (batch, 1), torch.float32, device)
        emergency = _tensor("ics.emergency", ics["emergency"], (batch, 1), torch.float32, device)
        terminated = _tensor("termination.terminated", termination["terminated"], (batch, 1), torch.bool, device)
        truncated = _tensor("termination.truncated", termination["truncated"], (batch, 1), torch.bool, device)
        done = _tensor("termination.done", termination["done"], (batch, 1), torch.bool, device)
        reason = _tensor("termination.reason_code", termination["reason_code"], (batch, 1), torch.long, device)
        mask_values = OrderedDict(
            (name, _tensor(f"termination.masks.{name}", masks[name], (batch, 1), torch.bool, device))
            for name in TERMINATION_MASK_KEYS
        )
        returned = _tensor("returned_critic", returned_critic, (batch, 18), torch.float32, device)
        returned_episode = _tensor("returned_episode_length", returned_episode_length, (batch,), torch.long, device)
        joint_pos = pre["joint_pos"]
        joint_vel = pre["joint_vel"]
        if type(joint_pos) is not torch.Tensor or type(joint_vel) is not torch.Tensor:
            raise TypeError("pre_reset joint state must contain exact tensors")
        if joint_pos.ndim != 2 or joint_pos.shape[0] != batch or joint_vel.shape != joint_pos.shape:
            raise ValueError("pre_reset joint tensors must share exact [N,J] shape")
        if joint_pos.dtype != torch.float32 or joint_vel.dtype != torch.float32:
            raise TypeError("pre_reset joint tensors must use float32")
        _tensor("pre_reset.joint_pos", joint_pos, tuple(joint_pos.shape), torch.float32, device)
        _tensor("pre_reset.joint_vel", joint_vel, tuple(joint_vel.shape), torch.float32, device)

        sources = OrderedDict((
            ("command", command), ("root", root), ("task_root", task_root),
            ("actual", actual), ("clearance", clearance), ("collision", collision),
            ("alpha", alpha), ("correction", correction), ("governed", governed),
            ("issued", issued), ("world", world), ("anchor_active", anchor_active),
            ("anchor_reset", anchor_reset), ("anchor_usable", anchor_usable),
            ("anchor_error", anchor_error), ("anchor_loss", anchor_loss),
            ("beta", beta), ("active_beams", active_beams), ("ics_clearance", ics_clearance),
            ("worst", worst), ("emergency", emergency), ("terminated", terminated),
            ("truncated", truncated), ("done", done), ("reason", reason),
            *((f"mask_{name}", value) for name, value in mask_values.items()),
            ("reset_mask", reset_mask), ("episode", episode), ("returned", returned),
            ("returned_episode", returned_episode), ("joint_pos", joint_pos), ("joint_vel", joint_vel),
        ))
        _reject_aliases(sources)
        expected_step = len(self._rows) + 1
        if bool(reset_mask.any().item()) or bool(done.any().item()) or bool(terminated.any().item()) or bool(truncated.any().item()):
            raise ValueError("bounded B0 runtime evaluation rejects every reset/done/termination")
        if any(bool(value.any().item()) for value in mask_values.values()) or bool(reason.any().item()):
            raise ValueError("bounded B0 termination masks and reasons must remain exact zero")
        if not torch.equal(root, task_root):
            raise ValueError("top-level and task same-state roots must be exact and independently owned")
        expected_critic = torch.cat((task_root, actual, clearance, collision.to(torch.float32)), dim=1)
        if not torch.equal(returned, expected_critic):
            raise ValueError("returned critic must bind the same-state task fields exactly")
        expected_episode = torch.full_like(episode, expected_step)
        if not torch.equal(episode, expected_episode) or not torch.equal(returned_episode, expected_episode):
            raise ValueError("episode length must bind the exact ordered no-reset policy row")
        if not torch.equal(alpha, torch.ones_like(alpha)) or bool(correction.any().item()):
            raise ValueError("disabled B0 governor must preserve alpha one and correction zero")
        if not torch.equal(governed, issued):
            raise ValueError("disabled B0 ICS must preserve governed and issued body velocity")
        if not bool(torch.isfinite(world).all().item()):
            raise ValueError("issued world velocity must remain finite")
        if any(bool(value.any().item()) for value in (
            anchor_active, anchor_reset, anchor_usable, anchor_error, anchor_loss,
            active_beams, ics_clearance, worst, emergency,
        )) or not torch.equal(beta, torch.ones_like(beta)):
            raise ValueError("anchor and ICS metrics must preserve exact disabled B0 semantics")
        if bool((clearance < 0.0).any().item()) or bool(collision.any().item()):
            raise ValueError("bounded B0 task must preserve nonnegative clearance and no collision")

        exact = _validate_exact_observability(
            exact_observability, batch, device, expected_step,
            self.config.rays_per_environment,
        )
        source_rows = tuple(bytes(row.tolist()) for row in exact.source_identity_sha256)
        if len(set(source_rows)) != batch or any(identity in self._seen_sources for identity in source_rows):
            raise ValueError("exact observability source identity is duplicated")
        cross_sources = OrderedDict(sources)
        cross_sources.update(
            (f"exact.{field.name}", getattr(exact, field.name)) for field in fields(exact)
        )
        _reject_aliases(cross_sources)

        row = OrderedDict((
            ("commanded_velocity_b_m_s", command.detach().clone()),
            ("actual_velocity_b_m_s", actual.detach().clone()),
            ("issued_velocity_b_m_s", issued.detach().clone()),
            ("position_w_m", task_root[:, :3].detach().clone()),
            ("station_reference_position_w_m", self._station_reference.detach().clone()),
            ("anchor_error_m", anchor_error[:, 0].detach().clone()),
            ("anchor_valid", torch.zeros(batch, dtype=torch.bool, device=device)),
            ("minimum_clearance_m", clearance[:, 0].detach().clone()),
            ("clearance_valid", torch.ones(batch, dtype=torch.bool, device=device)),
            ("collision", collision[:, 0].detach().clone()),
            ("ics_beta", beta[:, 0].detach().clone()),
            ("ics_emergency", torch.zeros(batch, dtype=torch.bool, device=device)),
            ("ics_worst_margin_m_s", worst[:, 0].detach().clone()),
            ("terminated", terminated[:, 0].detach().clone()),
            ("truncated", truncated[:, 0].detach().clone()),
            ("termination_reason", reason[:, 0].detach().clone()),
            ("termination_collision", mask_values["collision"][:, 0].detach().clone()),
            ("termination_below_height", mask_values["below_height"][:, 0].detach().clone()),
            ("termination_above_height", mask_values["above_height"][:, 0].detach().clone()),
            ("termination_timeout", mask_values["timeout"][:, 0].detach().clone()),
            ("observability_value", exact.sigma_min.detach().clone()),
            ("observability_valid", torch.ones(batch, dtype=torch.bool, device=device)),
            ("observability_source", torch.full((batch,), int(ObservabilitySource.EXACT), dtype=torch.long, device=device)),
        ))
        if tuple(row) != EVALUATION_INPUT_KEYS:
            raise RuntimeError("internal B0 evaluation input ordering violation")
        detail = OrderedDict(
            (name, getattr(exact, name).detach().clone()) for name in OBSERVABILITY_DETAIL_KEYS
        )
        command_identity = _row_hashes(OrderedDict((("commanded_velocity_b_m_s", command),)), (
            ("policy_step", str(expected_step)),
        ))

        self._rows.append(row)
        self._details.append(detail)
        self._command_identities.append(command_identity)
        self._source_identities.append(exact.source_identity_sha256.detach().clone())
        self._result_identities.append(exact.result_identity_sha256.detach().clone())
        self._seen_sources.update(source_rows)
        self._pending_command = None
        self._pending_snapshot = None

    def finalize(self, environment_result: B0EnvironmentResult) -> B0RuntimeEvaluationResult:
        if type(self) is not B0RuntimeEvaluationCollector:
            raise TypeError("finalize requires an exact B0RuntimeEvaluationCollector")
        if self._state != _ACTIVE:
            raise RuntimeError("B0 runtime evaluation may be finalized exactly once")
        if self._pending_command is not None or len(self._rows) != 8:
            raise RuntimeError("B0 runtime evaluation coverage is incomplete or has a pending command")
        if type(environment_result) is not B0EnvironmentResult:
            raise TypeError("environment_result must be an exact B0EnvironmentResult")
        receipt = environment_result.receipt
        coverage = environment_result.coverage
        if receipt.ready_manifest.canonical_json != self._manifest.canonical_json:
            raise ValueError("environment READY identity does not match collector ownership")
        expected_coverage = (8, 160, 160, 160, 320, 2, 20)
        actual_coverage = (
            coverage.policy_steps, coverage.physics_steps, coverage.lifecycle_calls,
            coverage.wrench_applications, coverage.environment_samples,
            coverage.num_envs, coverage.decimation,
        )
        if actual_coverage != expected_coverage or receipt.step_count != 160:
            raise ValueError("environment receipt or scalar coverage is not exact")
        ready_mapping = receipt.ready_manifest.to_mapping()
        executed_mapping = receipt.executed_manifest.to_mapping()
        if executed_mapping["execution_status"] != "EXECUTED":
            raise ValueError("environment receipt must contain the exact EXECUTED successor")
        for name, value in ready_mapping.items():
            if name not in ("execution_status", "manifest_sha256") and executed_mapping[name] != value:
                raise ValueError(f"environment EXECUTED successor changed {name}")
        if receipt.ready_manifest.manifest_sha256 == receipt.executed_manifest.manifest_sha256:
            raise ValueError("READY and EXECUTED manifest hashes must differ")

        evaluation = OrderedDict(
            (name, torch.stack([row[name] for row in self._rows])) for name in EVALUATION_INPUT_KEYS
        )
        details = OrderedDict(
            (name, torch.stack([row[name] for row in self._details])) for name in OBSERVABILITY_DETAIL_KEYS
        )
        step = compute_step_metrics(evaluation, self.config.metric_config)
        aggregate = aggregate_metrics(step)
        source_counts = (
            ("UNAVAILABLE", int((evaluation["observability_source"] == int(ObservabilitySource.UNAVAILABLE)).sum().item())),
            ("PROXY", int((evaluation["observability_source"] == int(ObservabilitySource.PROXY)).sum().item())),
            ("EXACT", int((evaluation["observability_source"] == int(ObservabilitySource.EXACT)).sum().item())),
        )
        termination_counts = (
            ("collision", int(evaluation["termination_collision"].sum().item())),
            ("below_height", int(evaluation["termination_below_height"].sum().item())),
            ("above_height", int(evaluation["termination_above_height"].sum().item())),
            ("timeout", int(evaluation["termination_timeout"].sum().item())),
            ("terminated", int(evaluation["terminated"].sum().item())),
            ("truncated", int(evaluation["truncated"].sum().item())),
            ("done", int((evaluation["terminated"] | evaluation["truncated"]).sum().item())),
            ("reason_none", int((evaluation["termination_reason"] == 0).sum().item())),
        )
        if source_counts != (("UNAVAILABLE", 0), ("PROXY", 0), ("EXACT", 16)):
            raise RuntimeError("observability source coverage is not exact")
        if termination_counts != (
            ("collision", 0), ("below_height", 0), ("above_height", 0),
            ("timeout", 0), ("terminated", 0), ("truncated", 0),
            ("done", 0), ("reason_none", 16),
        ):
            raise RuntimeError("bounded B0 termination counts are not exact")
        contract = B0RuntimeEvaluationContract(2, 8, 16, 160, 160, 160, 320, 21240, 339840)
        provenance = self.config.provenance
        record_mapping = OrderedDict((
            ("schema_version", "m7.b0-runtime-evaluation.v1"),
            ("platform", "TASLAB_UAV"),
            ("sensor", "Livox MID360"),
            ("baseline_id", "B0"),
            ("claim", "exact command-only wiring and aggregation evidence; no policy-performance claim"),
            ("actor_surface", OrderedDict((
                ("kind", "command-only"),
                ("input_fields", ["commanded_body_velocity_m_s"]),
                ("learned_actor_called", False),
                ("normalized_action_present", False),
            ))),
            ("manifest", OrderedDict((
                ("ready_sha256", receipt.ready_manifest.manifest_sha256),
                ("executed_sha256", receipt.executed_manifest.manifest_sha256),
            ))),
            ("receipt", OrderedDict((
                ("schema_version", receipt.schema_version),
                ("step_count", receipt.step_count),
                ("trajectory_sha256", receipt.trajectory_sha256),
            ))),
            ("coverage", OrderedDict(asdict(contract).items())),
            ("evaluation_input_order", list(EVALUATION_INPUT_KEYS)),
            ("source_manifest", OrderedDict(
                (name, OrderedDict(metadata)) for name, metadata in B0_SOURCE_MANIFEST
            )),
            ("metric_config", OrderedDict(asdict(self.config.metric_config).items())),
            ("observability_source_counts", OrderedDict(source_counts)),
            ("metrics", OrderedDict((name, OrderedDict(values)) for name, values in _metric_record(aggregate))),
            ("termination_counts", OrderedDict(termination_counts)),
            ("provenance", OrderedDict((
                ("product_commit", provenance.product_commit),
                ("fixture_id", provenance.fixture_id),
                ("generator_id", provenance.generator_id),
                ("seed", provenance.seed),
            ))),
            ("gates", OrderedDict((
                ("coverage_exact", True), ("metrics_unchanged", True),
                ("command_only_actor_surface", True), ("observability_normal_free", True),
                ("no_reset", True), ("all_finite", True),
            ))),
        ))
        if tuple(record_mapping) != RECORD_KEYS or tuple(record_mapping["provenance"]) != PROVENANCE_KEYS:
            raise RuntimeError("canonical B0 record ordering violation")
        canonical = json.dumps(record_mapping, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        record = B0RuntimeEvaluationRecord(canonical)
        output_inputs = tuple((name, value.detach().clone()) for name, value in evaluation.items())
        output_details = tuple((name, value.detach().clone()) for name, value in details.items())
        result = B0RuntimeEvaluationResult(
            "m7.b0-runtime-evaluation-result.v1", contract,
            receipt.ready_manifest.manifest_sha256,
            receipt.executed_manifest.manifest_sha256,
            receipt.trajectory_sha256,
            torch.stack(self._command_identities).detach().clone(),
            torch.stack(self._source_identities).detach().clone(),
            torch.stack(self._result_identities).detach().clone(),
            output_inputs, output_details, aggregate, record,
            tuple((name, bool(value)) for name, value in record_mapping["gates"].items()),
        )
        self._pending_command = None
        self._pending_snapshot = None
        self._station_reference = None
        self._rows.clear(); self._details.clear(); self._command_identities.clear()
        self._source_identities.clear(); self._result_identities.clear(); self._seen_sources.clear()
        self._state = _FINALIZED
        return result


def begin_ready_b0_runtime_evaluation(
    manifest: BaselineRunManifest,
    initial_critic: torch.Tensor,
    config: B0RuntimeEvaluationConfig,
) -> B0RuntimeEvaluationCollector:
    """Own exact inputs and construct one bounded B0 runtime evaluation collector."""
    if type(config) is not B0RuntimeEvaluationConfig:
        raise TypeError("config must be an exact B0RuntimeEvaluationConfig")
    return B0RuntimeEvaluationCollector(manifest, initial_critic, config, _COLLECTOR_TOKEN)


__all__ = [
    "ANCHOR_KEYS", "B0RuntimeEvaluationCollector", "B0RuntimeEvaluationConfig",
    "B0RuntimeEvaluationContract", "B0RuntimeEvaluationProvenance",
    "B0RuntimeEvaluationRecord", "B0RuntimeEvaluationResult", "B0_SOURCE_MANIFEST",
    "CONTROL_KEYS", "ICS_KEYS", "OBSERVABILITY_DETAIL_KEYS", "PRE_RESET_KEYS",
    "PROVENANCE_KEYS", "RECORD_KEYS", "SOURCE_METADATA_KEYS", "TASK_KEYS",
    "TERMINATION_KEYS", "TERMINATION_MASK_KEYS", "begin_ready_b0_runtime_evaluation",
]
