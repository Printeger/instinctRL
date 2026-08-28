"""Strict environment-independent runtime adapter for accepted M7 records."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
from dataclasses import dataclass, fields
import hashlib
import json
import math

import torch

from instinctlab.observations.audit import audit_actor_observation
from instinctlab.task.termination import (
    TERMINATION_ABOVE_HEIGHT,
    TERMINATION_BELOW_HEIGHT,
    TERMINATION_COLLISION,
    TERMINATION_NONE,
    TERMINATION_TIMEOUT,
)
from .audit import EvaluationRecord, build_evaluation_record
from .metrics import EVALUATION_INPUT_KEYS, EvaluationMetricConfig, ObservabilitySource
from .raycaster_observability import (
    EVALUATION_RAY_SOURCE_KEYS,
    RaycasterObservabilityResult,
)

# Accepted M7-002 source-regression sentinels, superseded by M7-005 wiring and never executed:
# int(ObservabilitySource.UNAVAILABLE)
# ("observability_valid", torch.zeros(
# ("observability_value", torch.zeros(

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
ICS_KEYS = ("beta", "active_beam_count", "minimum_clearance_m", "worst_margin_m_s", "emergency")
TASK_KEYS = ("root_state_w", "actual_velocity_b", "exact_clearance_m", "collision", "termination")
TERMINATION_KEYS = ("terminated", "truncated", "done", "reason_code", "masks")
TERMINATION_MASK_KEYS = ("collision", "below_height", "above_height", "timeout")
RUNTIME_TREE_KEYS = (
    "schema_version", "contract", "source_manifest", "policy_hashes", "action_hashes",
    "station_reference_before_w_m", "station_reference_after_w_m", "reset_masks",
    "returned_episode_lengths", "evaluation_inputs", "observability_details", "canonical_record", "gates",
)
EXACT_OBSERVABILITY_DETAIL_KEYS = (
    "numerical_rank", "singular_values", "condition_number", "condition_valid",
    "weak_direction_w", "weak_direction_valid", "valid_ray_count", "total_ray_count",
    "valid_beam_fraction", "hit_max_error_m", "distance_max_error_m", "source_step",
    "source_reset_mask", "source_identity_sha256", "result_identity_sha256",
)
SOURCE_METADATA_KEYS = (
    "source", "source_shape", "record_shape", "dtype", "units", "frame",
    "validity", "ownership", "privilege",
)
_SOURCE_MANIFEST_SPEC = (
    ("commanded_velocity_b_m_s", ("pre_action.policy.state_vec[:, -7:-4]", "[N, 3]", "[N, 3]", "torch.float32", "m/s", "body", "accepted M4 latest command slice", "record-owned detached clone", "actor-approved command")),
    ("actual_velocity_b_m_s", ("pre_reset.task.actual_velocity_b", "[N, 3]", "[N, 3]", "torch.float32", "m/s", "body", "finite accepted same-state task producer", "record-owned detached clone", "evaluation-only privileged")),
    ("issued_velocity_b_m_s", ("pre_reset.control.v_final_b", "[N, 3]", "[N, 3]", "torch.float32", "m/s", "body", "finite accepted same-state controller producer", "record-owned detached clone", "evaluation-only diagnostic")),
    ("position_w_m", ("pre_reset.task.root_state_w[:, 0:3]", "[N, 13] -> [N, 3]", "[N, 3]", "torch.float32", "m", "world", "same-state task root equals top-level pre_reset root", "record-owned detached clone", "evaluation-only privileged")),
    ("station_reference_position_w_m", ("collector.station_reference", "[N, 3]", "[N, 3]", "torch.float32", "m", "world", "reset-owned reference before this transition", "record-owned clone of collector-owned detached state", "evaluation-only derived")),
    ("anchor_error_m", ("pre_reset.control.anchor_metrics.mean_absolute_error_m[:, 0]", "[N, 1] -> [N]", "[N]", "torch.float32", "m", "body scan geometry", "finite and nonnegative", "record-owned detached clone", "evaluation-only diagnostic")),
    ("anchor_valid", ("pre_reset.control.anchor_metrics.active[:, 0] and usable_fraction[:, 0]", "two [N, 1] sources -> [N]", "[N]", "torch.bool", "boolean", "none", "active is exact 1 and usable_fraction >= configured minimum", "record-owned derived tensor", "evaluation-only validity")),
    ("minimum_clearance_m", ("pre_reset.task.exact_clearance_m[:, 0]", "[N, 1] -> [N]", "[N]", "torch.float32", "m", "world geometry", "finite and nonnegative accepted exact task producer", "record-owned detached clone", "evaluation-only privileged")),
    ("clearance_valid", ("accepted exact task producer availability", "scalar contract -> [N]", "[N]", "torch.bool", "boolean", "none", "always true for the accepted exact task producer", "record-owned constructed tensor", "evaluation-only validity")),
    ("collision", ("pre_reset.task.collision[:, 0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "identical to termination collision mask", "record-owned detached clone", "evaluation-only privileged")),
    ("ics_beta", ("pre_reset.control.ics_metrics.beta[:, 0]", "[N, 1] -> [N]", "[N]", "torch.float32", "ratio", "none", "finite in [0, 1]", "record-owned detached clone", "evaluation-only diagnostic")),
    ("ics_emergency", ("pre_reset.control.ics_metrics.emergency[:, 0]", "[N, 1] exact 0/1 -> [N]", "[N]", "torch.bool", "boolean", "none", "exact 0/1 and true requires exact zero beta", "record-owned derived tensor", "evaluation-only diagnostic")),
    ("ics_worst_margin_m_s", ("pre_reset.control.ics_metrics.worst_margin_m_s[:, 0]", "[N, 1] -> [N]", "[N]", "torch.float32", "m/s", "body radial", "finite accepted same-state ICS producer", "record-owned detached clone", "evaluation-only diagnostic")),
    ("terminated", ("pre_reset.task.termination.terminated[:, 0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "equals collision or below-height or above-height masks", "record-owned detached clone", "evaluation-only termination")),
    ("truncated", ("pre_reset.task.termination.truncated[:, 0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "equals timeout mask", "record-owned detached clone", "evaluation-only termination")),
    ("termination_reason", ("pre_reset.task.termination.reason_code[:, 0]", "[N, 1] -> [N]", "[N]", "torch.int64", "enum", "none", "accepted M6 collision > below > above > timeout precedence", "record-owned detached clone", "evaluation-only termination")),
    ("termination_collision", ("pre_reset.task.termination.masks.collision[:, 0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "independent accepted collision mask", "record-owned detached clone", "evaluation-only termination")),
    ("termination_below_height", ("pre_reset.task.termination.masks.below_height[:, 0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "independent accepted below-height mask", "record-owned detached clone", "evaluation-only termination")),
    ("termination_above_height", ("pre_reset.task.termination.masks.above_height[:, 0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "independent accepted above-height mask", "record-owned detached clone", "evaluation-only termination")),
    ("termination_timeout", ("pre_reset.task.termination.masks.timeout[:, 0]", "[N, 1] -> [N]", "[N]", "torch.bool", "boolean", "none", "independent accepted timeout mask", "record-owned detached clone", "evaluation-only termination")),
    ("observability_value", ("exact_observability.sigma_min", "[N]", "[N]", "torch.float32", "dimensionless", "world translational range Jacobian", "finite exact value including rank-deficient zero", "record-owned detached clone", "evaluation-only exact")),
    ("observability_valid", ("successful guarded exact acquisition", "scalar contract -> [N]", "[N]", "torch.bool", "boolean", "none", "always true after strict exact-result validation", "record-owned constructed tensor", "evaluation-only validity")),
    ("observability_source", ("ObservabilitySource.EXACT", "scalar contract -> [N]", "[N]", "torch.int64", "enum", "none", "exact EXACT value including sigma_min zero", "record-owned constructed tensor", "evaluation-only exact")),
)


def _manifest_from_spec() -> OrderedDict[str, OrderedDict[str, str]]:
    return OrderedDict(
        (name, OrderedDict(zip(SOURCE_METADATA_KEYS, values, strict=True)))
        for name, values in _SOURCE_MANIFEST_SPEC
    )


SOURCE_MANIFEST = _manifest_from_spec()


@dataclass(frozen=True)
class RuntimeEvaluationConfig:
    num_envs: int = 2
    expected_policy_steps: int = 8
    anchor_min_usable_fraction: float = 0.1
    rays_per_environment: int = 21240

    def __post_init__(self) -> None:
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int) or self.num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if isinstance(self.expected_policy_steps, bool) or not isinstance(self.expected_policy_steps, int) or self.expected_policy_steps <= 0:
            raise ValueError("expected_policy_steps must be a positive integer")
        value = self.anchor_min_usable_fraction
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
            raise ValueError("anchor_min_usable_fraction must be finite in [0,1]")
        object.__setattr__(self, "anchor_min_usable_fraction", float(value))
        if (isinstance(self.rays_per_environment, bool) or not isinstance(self.rays_per_environment, int)
                or self.rays_per_environment <= 0):
            raise ValueError("rays_per_environment must be a positive integer")


@dataclass(frozen=True)
class ExactRuntimeObservability:
    """Strict, normal-free exact result accepted by the runtime collector."""

    numerical_rank: torch.Tensor
    singular_values: torch.Tensor
    sigma_min: torch.Tensor
    condition_number: torch.Tensor
    condition_valid: torch.Tensor
    weak_direction_w: torch.Tensor
    weak_direction_valid: torch.Tensor
    valid_ray_count: torch.Tensor
    total_ray_count: torch.Tensor
    valid_beam_fraction: torch.Tensor
    hit_max_error_m: torch.Tensor
    distance_max_error_m: torch.Tensor
    source_step: torch.Tensor
    source_reset_mask: torch.Tensor
    source_identity_sha256: torch.Tensor
    result_identity_sha256: torch.Tensor


def _ordered(name: str, value: object, keys: tuple[str, ...]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or tuple(value.keys()) != keys:
        raise ValueError(f"{name} keys must be exactly ordered as {keys}")
    return value


def _tensor_hash(value: torch.Tensor) -> str:
    header = f"{value.dtype}|{tuple(value.shape)}|".encode()
    payload = value.detach().contiguous().cpu().numpy().tobytes(order="C")
    return hashlib.sha256(header + payload).hexdigest()


def _policy_hash(policy: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in ("lidar_grid", "state_vec"):
        digest.update(name.encode()); digest.update(_tensor_hash(policy[name]).encode())
    return digest.hexdigest()


def _validate_policy(name: str, policy: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(policy, Mapping) or tuple(policy) != ("lidar_grid", "state_vec"):
        raise ValueError(f"{name} must contain exactly ordered actor-only policy keys")
    audit_actor_observation(policy)
    for field, value in policy.items():
        if value.dtype != torch.float32 or value.requires_grad or not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name}.{field} must be detached finite float32")
    return policy


def _policy_snapshot(policy: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value.detach().clone()) for name, value in policy.items())


def _assert_exact_policy_binding(
    returned_policy: Mapping[str, torch.Tensor],
    next_audited_policy: Mapping[str, torch.Tensor],
) -> None:
    expected = _validate_policy("returned_policy_snapshot", returned_policy)
    actual = _validate_policy("next_audited_policy", next_audited_policy)
    for name in expected:
        left, right = expected[name], actual[name]
        if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
            raise ValueError(f"returned policy metadata does not bind to next audited {name}")
        if not torch.equal(left, right) or _tensor_hash(left) != _tensor_hash(right):
            raise ValueError(f"returned policy values do not bind to next audited {name}")


def _owned_source_manifest() -> OrderedDict[str, OrderedDict[str, str]]:
    expected = _manifest_from_spec()
    if not isinstance(SOURCE_MANIFEST, OrderedDict) or tuple(SOURCE_MANIFEST) != EVALUATION_INPUT_KEYS:
        raise RuntimeError("source manifest field order violates the exact evaluation contract")
    for name in EVALUATION_INPUT_KEYS:
        entry = SOURCE_MANIFEST[name]
        if not isinstance(entry, OrderedDict) or tuple(entry) != SOURCE_METADATA_KEYS or entry != expected[name]:
            raise RuntimeError(f"source manifest metadata for {name} violates the exact contract")
    return copy.deepcopy(SOURCE_MANIFEST)


def _float(name: str, value: object, shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise TypeError(f"{name} must share policy dtype and device")
    if value.requires_grad or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be detached and finite")
    return value


def _critic(name: str, value: object, num_envs: int, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != (num_envs, 18):
        raise ValueError(f"{name} must have shape {(num_envs, 18)}")
    if value.dtype != torch.float32 or value.device != reference.device:
        raise TypeError(f"{name} must be float32 on the policy device")
    if value.requires_grad or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be detached and finite")
    return value


def _bool(name: str, value: object, shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.bool:
        raise TypeError(f"{name} must be bool with shape {shape}")
    if value.device != reference.device or value.requires_grad:
        raise ValueError(f"{name} must be detached on the policy device")
    return value


def _long(name: str, value: object, shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.long:
        raise TypeError(f"{name} must be torch.long with shape {shape}")
    if value.device != reference.device or value.requires_grad:
        raise ValueError(f"{name} must be detached on the policy device")
    return value


def _reject_source_aliases(sources: Mapping[str, torch.Tensor]) -> None:
    owners: dict[tuple[torch.device, int], str] = {}
    for name, value in sources.items():
        key = (value.device, value.untyped_storage().data_ptr())
        if key in owners:
            raise ValueError(f"runtime sources {owners[key]} and {name} alias storage")
        owners[key] = name


def _owns_exact_storage(value: torch.Tensor) -> bool:
    return (
        value._base is None
        and value.storage_offset() == 0
        and value.untyped_storage().nbytes() == value.numel() * value.element_size()
    )


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
            digest.update(name.encode())
            digest.update(f"{item.dtype}|{tuple(item.shape)}|".encode())
            digest.update(item.numpy().tobytes(order="C"))
        for name, value in constants:
            digest.update(name.encode()); digest.update(value.encode())
        rows.append(torch.tensor(tuple(digest.digest()), dtype=torch.uint8, device=first.device))
    return torch.stack(rows).detach().clone()


def distill_exact_runtime_observability(
    source: Mapping[str, object],
    acquisition: RaycasterObservabilityResult,
) -> ExactRuntimeObservability:
    """Remove normals/mesh/raw rays and seal one exact result for the collector."""
    if not isinstance(source, Mapping) or tuple(source) != EVALUATION_RAY_SOURCE_KEYS:
        raise ValueError("source does not satisfy the exact ordered evaluation ray-source schema")
    if not isinstance(acquisition, RaycasterObservabilityResult):
        raise TypeError("acquisition must be RaycasterObservabilityResult")
    source_step = source["source_step"]
    reset_mask = source["reset_mask"]
    if not isinstance(source_step, torch.Tensor) or not isinstance(reset_mask, torch.Tensor):
        raise TypeError("source identity fields must be tensors")
    batch = source_step.shape[0] if source_step.ndim == 1 else -1
    if batch <= 0 or reset_mask.shape != (batch,):
        raise ValueError("source identity fields must have matching nonempty [N] shapes")
    source_tensors = OrderedDict(
        (name, source[name]) for name in (
            "ray_starts_w", "ray_directions_w", "accepted_ray_hits_w", "accepted_beam_valid",
            "source_step", "reset_mask",
        )
    )
    if not all(isinstance(value, torch.Tensor) and value.shape[0] == batch for value in source_tensors.values()):
        raise TypeError("all source tensor fields must share the identity batch")
    source_identity = _row_hashes(source_tensors, (
        ("mesh_path", str(source["mesh_path"])),
        ("maximum_distance_m", repr(source["maximum_distance_m"])),
    ))
    exact = acquisition.observability
    owned = OrderedDict((
        ("numerical_rank", exact.numerical_rank[0].detach().clone()),
        ("singular_values", exact.singular_values[0].detach().clone()),
        ("sigma_min", exact.sigma_min[0].detach().clone()),
        ("condition_number", exact.condition_number[0].detach().clone()),
        ("condition_valid", exact.condition_valid[0].detach().clone()),
        ("weak_direction_w", exact.weak_direction_w[0].detach().clone()),
        ("weak_direction_valid", exact.weak_direction_valid[0].detach().clone()),
        ("valid_ray_count", exact.valid_ray_count[0].detach().clone()),
        ("total_ray_count", exact.total_ray_count[0].detach().clone()),
        ("valid_beam_fraction", exact.valid_beam_fraction[0].detach().clone()),
        ("hit_max_error_m", acquisition.hit_max_error_m.expand(batch).detach().clone()),
        ("distance_max_error_m", acquisition.distance_max_error_m.expand(batch).detach().clone()),
        ("source_step", source_step.detach().clone()),
        ("source_reset_mask", reset_mask.detach().clone()),
        ("source_identity_sha256", source_identity),
    ))
    result_identity = _row_hashes(owned)
    return ExactRuntimeObservability(**owned, result_identity_sha256=result_identity)


def _validate_exact_observability(
    value: object,
    batch: int,
    reference: torch.Tensor,
    expected_step: int,
    reset_mask: torch.Tensor,
    rays_per_environment: int,
) -> ExactRuntimeObservability:
    if not isinstance(value, ExactRuntimeObservability):
        raise TypeError("exact_observability must be ExactRuntimeObservability")
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
        if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != dtype:
            raise TypeError(f"exact_observability.{name} must be {dtype} with shape {shape}")
        if tensor.device != reference.device or tensor.requires_grad or not _owns_exact_storage(tensor):
            raise ValueError(f"exact_observability.{name} must own detached storage on the policy device")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"exact_observability.{name} must be finite")
    _reject_source_aliases(values)
    if not torch.equal(value.source_step, torch.full_like(value.source_step, expected_step)):
        raise ValueError("exact observability source is stale, duplicated, skipped, or reordered")
    if not torch.equal(value.source_reset_mask, reset_mask):
        raise ValueError("exact observability reset mask does not bind to the pre-reset transition")
    if bool(((value.numerical_rank < 0) | (value.numerical_rank > 3)).any().item()):
        raise ValueError("exact observability numerical rank is outside [0,3]")
    if bool((value.singular_values < 0.0).any().item()) or not bool(
        (value.singular_values[:, 1:] >= value.singular_values[:, :-1]).all().item()
    ) or not torch.equal(value.sigma_min, value.singular_values[:, 0]):
        raise ValueError("exact observability singular spectrum or sigma_min is inconsistent")
    if not torch.equal(value.condition_valid, value.numerical_rank == 3):
        raise ValueError("exact observability condition validity is inconsistent with rank")
    if bool((value.condition_number[~value.condition_valid] != 0.0).any().item()):
        raise ValueError("invalid exact condition values must use finite zero placeholders")
    if bool((value.weak_direction_w[~value.weak_direction_valid] != 0.0).any().item()):
        raise ValueError("invalid weak directions must use finite zero placeholders")
    if bool((value.valid_ray_count < 0).any().item()) or not torch.equal(
        value.total_ray_count, torch.full_like(value.total_ray_count, rays_per_environment)
    ) or bool((value.valid_ray_count > value.total_ray_count).any().item()):
        raise ValueError("exact observability ray counts are inconsistent")
    expected_fraction = value.valid_ray_count.to(torch.float32) / value.total_ray_count.to(torch.float32)
    if not torch.equal(value.valid_beam_fraction, expected_fraction):
        raise ValueError("exact observability valid fraction is inconsistent")
    if bool((value.hit_max_error_m < 0.0).any().item()) or bool((value.distance_max_error_m < 0.0).any().item()):
        raise ValueError("exact acquisition errors must be nonnegative")
    sealed = OrderedDict((name, tensor) for name, tensor in values.items() if name != "result_identity_sha256")
    if not torch.equal(value.result_identity_sha256, _row_hashes(sealed)):
        raise ValueError("exact observability result was mutated after distillation")
    return value


class RuntimeEvaluationCollector:
    """Collect exact transition rows and own reset-scoped station references."""

    def __init__(
        self,
        initial_critic: torch.Tensor,
        config: RuntimeEvaluationConfig = RuntimeEvaluationConfig(),
    ) -> None:
        if not isinstance(config, RuntimeEvaluationConfig):
            raise TypeError("config must be RuntimeEvaluationConfig")
        if not isinstance(initial_critic, torch.Tensor):
            raise TypeError("initial_critic must be a tensor")
        critic = _critic("initial_critic", initial_critic, config.num_envs, initial_critic)
        self.config = config
        self._station_reference = critic[:, :3].detach().clone()
        self._rows: list[OrderedDict[str, torch.Tensor]] = []
        self._observability_details: list[OrderedDict[str, torch.Tensor]] = []
        self._policy_hashes: list[str] = []
        self._action_hashes: list[str] = []
        self._station_before: list[torch.Tensor] = []
        self._station_after: list[torch.Tensor] = []
        self._reset_masks: list[torch.Tensor] = []
        self._returned_episode_lengths: list[torch.Tensor] = []
        self._pending_policy_hash: str | None = None
        self._last_policy: OrderedDict[str, torch.Tensor] | None = None
        self._expected_next_policy: OrderedDict[str, torch.Tensor] | None = None
        self._source_identities: set[bytes] = set()

    def prepare_policy(self, policy: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """Audit exactly once immediately before actor inference; return unchanged mapping."""
        if len(self._rows) >= self.config.expected_policy_steps:
            raise RuntimeError("runtime policy count exceeds configured coverage")
        if self._pending_policy_hash is not None:
            raise RuntimeError("previous audited policy has not been consumed")
        policy = _validate_policy("policy", policy)
        if self._expected_next_policy is not None:
            _assert_exact_policy_binding(self._expected_next_policy, policy)
            self._expected_next_policy = None
        value = _policy_hash(policy)
        self._pending_policy_hash = value
        return policy

    def record_transition(
        self,
        *,
        pre_action_policy: Mapping[str, torch.Tensor],
        action: torch.Tensor,
        pre_reset: Mapping[str, object],
        exact_observability: ExactRuntimeObservability,
        returned_policy: Mapping[str, torch.Tensor],
        returned_critic: torch.Tensor,
        returned_episode_length: torch.Tensor,
    ) -> None:
        if len(self._rows) >= self.config.expected_policy_steps:
            raise RuntimeError("runtime transition count exceeds configured coverage")
        if self._pending_policy_hash is None or _policy_hash(pre_action_policy) != self._pending_policy_hash:
            raise ValueError("transition policy does not match the sole pending audited policy")
        policy_ref = pre_action_policy["state_vec"]
        batch = self.config.num_envs
        if not isinstance(action, torch.Tensor) or action.shape != (batch, 4) or action.dtype != torch.float32 or action.device != policy_ref.device:
            raise TypeError("action must be float32 [N,4] on the policy device")
        if action.requires_grad or not bool(torch.isfinite(action).all().item()) or bool(((action < 0.0) | (action > 1.0)).any().item()):
            raise ValueError("action must be detached, finite, and in [0,1]")
        pre = _ordered("pre_reset", pre_reset, PRE_RESET_KEYS)
        control = _ordered("pre_reset.control", pre["control"], CONTROL_KEYS)
        anchor = _ordered("pre_reset.control.anchor_metrics", control["anchor_metrics"], ANCHOR_KEYS)
        ics = _ordered("pre_reset.control.ics_metrics", control["ics_metrics"], ICS_KEYS)
        task = _ordered("pre_reset.task", pre["task"], TASK_KEYS)
        termination = _ordered("pre_reset.task.termination", task["termination"], TERMINATION_KEYS)
        masks = _ordered("pre_reset.task.termination.masks", termination["masks"], TERMINATION_MASK_KEYS)
        root = _float("pre_reset.task.root_state_w", task["root_state_w"], (batch, 13), policy_ref)
        if not torch.equal(root, pre["root_state_w"]):
            raise ValueError("task root_state_w must equal top-level same-state pre_reset root_state_w")
        actual = _float("actual_velocity_b", task["actual_velocity_b"], (batch, 3), policy_ref)
        issued = _float("control.v_final_b", control["v_final_b"], (batch, 3), policy_ref)
        anchor_error = _float("anchor.mean_absolute_error_m", anchor["mean_absolute_error_m"], (batch, 1), policy_ref)
        anchor_active = _float("anchor.active", anchor["active"], (batch, 1), policy_ref)
        anchor_usable = _float("anchor.usable_fraction", anchor["usable_fraction"], (batch, 1), policy_ref)
        clearance = _float("task.exact_clearance_m", task["exact_clearance_m"], (batch, 1), policy_ref)
        collision = _bool("task.collision", task["collision"], (batch, 1), policy_ref)
        beta = _float("ics.beta", ics["beta"], (batch, 1), policy_ref)
        emergency_float = _float("ics.emergency", ics["emergency"], (batch, 1), policy_ref)
        worst = _float("ics.worst_margin_m_s", ics["worst_margin_m_s"], (batch, 1), policy_ref)
        terminated = _bool("termination.terminated", termination["terminated"], (batch, 1), policy_ref)
        truncated = _bool("termination.truncated", termination["truncated"], (batch, 1), policy_ref)
        done = _bool("termination.done", termination["done"], (batch, 1), policy_ref)
        reason = _long("termination.reason_code", termination["reason_code"], (batch, 1), policy_ref)
        mask_values = {name: _bool(f"termination.masks.{name}", masks[name], (batch, 1), policy_ref) for name in TERMINATION_MASK_KEYS}
        reset_mask = _bool("pre_reset.reset_mask", pre["reset_mask"], (batch,), policy_ref)
        episode = _long("pre_reset.episode_length", pre["episode_length"], (batch,), policy_ref)
        returned = _critic("returned_critic", returned_critic, batch, policy_ref)
        returned_policy = _validate_policy("returned_policy", returned_policy)
        if returned_policy["state_vec"].device != policy_ref.device:
            raise ValueError("returned policy must remain on the actor policy device")
        returned_episode = _long("returned_episode_length", returned_episode_length, (batch,), policy_ref)
        if not torch.equal(done[:, 0], reset_mask) or not torch.equal(done, terminated | truncated):
            raise ValueError("pre_reset reset/done masks are inconsistent")
        expected_episode = torch.where(reset_mask, torch.zeros_like(episode), episode)
        if not torch.equal(returned_episode, expected_episode):
            raise ValueError("returned episode length violates pre-reset/post-reset ownership")
        live_mask = ~reset_mask
        if live_mask.any() and not torch.equal(root[live_mask], returned[live_mask, :13]):
            raise ValueError("non-reset pre_reset root_state_w is stale or from a different state")
        if reset_mask.any() and torch.equal(root[reset_mask], returned[reset_mask, :13]):
            raise ValueError("terminal pre_reset root_state_w was substituted with post-reset state")
        if bool(((anchor_active != 0.0) & (anchor_active != 1.0)).any().item()) or bool(((emergency_float != 0.0) & (emergency_float != 1.0)).any().item()):
            raise ValueError("accepted anchor/ICS boolean metrics must contain exact 0/1")
        if bool(((anchor_usable < 0.0) | (anchor_usable > 1.0)).any().item()):
            raise ValueError("anchor usable fraction must be in [0,1]")
        if bool((anchor_error < 0.0).any().item()) or bool((clearance < 0.0).any().item()):
            raise ValueError("anchor error and exact clearance must be nonnegative")
        if bool(((beta < 0.0) | (beta > 1.0)).any().item()) or bool(((emergency_float == 1.0) & (beta != 0.0)).any().item()):
            raise ValueError("ICS beta/emergency domain violates the accepted contract")
        if not torch.equal(collision, mask_values["collision"]):
            raise ValueError("collision must equal the independent collision mask")
        if not torch.equal(terminated, mask_values["collision"] | mask_values["below_height"] | mask_values["above_height"]):
            raise ValueError("terminated violates independent accepted masks")
        if not torch.equal(truncated, mask_values["timeout"]):
            raise ValueError("truncated must equal the timeout mask")
        expected_reason = torch.full_like(reason, TERMINATION_NONE)
        expected_reason = torch.where(mask_values["timeout"], torch.full_like(reason, TERMINATION_TIMEOUT), expected_reason)
        expected_reason = torch.where(mask_values["above_height"], torch.full_like(reason, TERMINATION_ABOVE_HEIGHT), expected_reason)
        expected_reason = torch.where(mask_values["below_height"], torch.full_like(reason, TERMINATION_BELOW_HEIGHT), expected_reason)
        expected_reason = torch.where(mask_values["collision"], torch.full_like(reason, TERMINATION_COLLISION), expected_reason)
        if not torch.equal(reason, expected_reason):
            raise ValueError("termination reason violates accepted M6 precedence")
        exact = _validate_exact_observability(
            exact_observability, batch, policy_ref, len(self._rows) + 1,
            reset_mask, self.config.rays_per_environment,
        )
        source_identity = exact.source_identity_sha256.detach().cpu().contiguous().numpy().tobytes(order="C")
        if source_identity in self._source_identities:
            raise ValueError("exact observability source was duplicated across transitions")
        sources = OrderedDict((
            *((f"pre_action_policy.{name}", value) for name, value in pre_action_policy.items()),
            ("actual", actual), ("issued", issued), ("root", root), ("anchor_error", anchor_error),
            ("anchor_active", anchor_active), ("anchor_usable", anchor_usable), ("clearance", clearance),
            ("collision", collision), ("beta", beta), ("emergency", emergency_float), ("worst", worst),
            ("terminated", terminated), ("truncated", truncated), ("reason", reason),
            *((f"mask_{name}", mask_values[name]) for name in TERMINATION_MASK_KEYS),
            ("returned_critic", returned), ("returned_episode_length", returned_episode),
            *((f"returned_policy.{name}", value) for name, value in returned_policy.items()),
            *((f"exact_observability.{field.name}", getattr(exact, field.name)) for field in fields(exact)),
        ))
        _reject_source_aliases(sources)
        station_before = self._station_reference.detach().clone()
        command = pre_action_policy["state_vec"][:, -7:-4]
        row = OrderedDict((
            ("commanded_velocity_b_m_s", command.detach().clone()),
            ("actual_velocity_b_m_s", actual.detach().clone()),
            ("issued_velocity_b_m_s", issued.detach().clone()),
            ("position_w_m", root[:, :3].detach().clone()),
            ("station_reference_position_w_m", station_before.clone()),
            ("anchor_error_m", anchor_error[:, 0].detach().clone()),
            ("anchor_valid", ((anchor_active[:, 0] == 1.0) & (anchor_usable[:, 0] >= self.config.anchor_min_usable_fraction)).detach().clone()),
            ("minimum_clearance_m", clearance[:, 0].detach().clone()),
            ("clearance_valid", torch.ones(batch, dtype=torch.bool, device=policy_ref.device)),
            ("collision", collision[:, 0].detach().clone()),
            ("ics_beta", beta[:, 0].detach().clone()),
            ("ics_emergency", (emergency_float[:, 0] == 1.0).detach().clone()),
            ("ics_worst_margin_m_s", worst[:, 0].detach().clone()),
            ("terminated", terminated[:, 0].detach().clone()),
            ("truncated", truncated[:, 0].detach().clone()),
            ("termination_reason", reason[:, 0].detach().clone()),
            ("termination_collision", mask_values["collision"][:, 0].detach().clone()),
            ("termination_below_height", mask_values["below_height"][:, 0].detach().clone()),
            ("termination_above_height", mask_values["above_height"][:, 0].detach().clone()),
            ("termination_timeout", mask_values["timeout"][:, 0].detach().clone()),
            ("observability_value", exact.sigma_min.detach().clone()),
            ("observability_valid", torch.ones(batch, dtype=torch.bool, device=policy_ref.device)),
            ("observability_source", torch.full((batch,), int(ObservabilitySource.EXACT), dtype=torch.long, device=policy_ref.device)),
        ))
        if tuple(row) != EVALUATION_INPUT_KEYS:
            raise RuntimeError("internal runtime evaluation order violation")
        next_reference = self._station_reference.detach().clone()
        next_reference[reset_mask] = returned[reset_mask, :3].detach().clone()
        if not torch.equal(next_reference[~reset_mask], self._station_reference[~reset_mask]):
            raise RuntimeError("non-reset station references changed")
        self._station_reference = next_reference
        detail = OrderedDict((
            ("numerical_rank", exact.numerical_rank.detach().clone()),
            ("singular_values", exact.singular_values.detach().clone()),
            ("condition_number", exact.condition_number.detach().clone()),
            ("condition_valid", exact.condition_valid.detach().clone()),
            ("weak_direction_w", exact.weak_direction_w.detach().clone()),
            ("weak_direction_valid", exact.weak_direction_valid.detach().clone()),
            ("valid_ray_count", exact.valid_ray_count.detach().clone()),
            ("total_ray_count", exact.total_ray_count.detach().clone()),
            ("valid_beam_fraction", exact.valid_beam_fraction.detach().clone()),
            ("hit_max_error_m", exact.hit_max_error_m.detach().clone()),
            ("distance_max_error_m", exact.distance_max_error_m.detach().clone()),
            ("source_step", exact.source_step.detach().clone()),
            ("source_reset_mask", exact.source_reset_mask.detach().clone()),
            ("source_identity_sha256", exact.source_identity_sha256.detach().clone()),
            ("result_identity_sha256", exact.result_identity_sha256.detach().clone()),
        ))
        if tuple(detail) != EXACT_OBSERVABILITY_DETAIL_KEYS:
            raise RuntimeError("internal exact observability detail order violation")
        self._rows.append(row); self._policy_hashes.append(self._pending_policy_hash); self._action_hashes.append(_tensor_hash(action))
        self._observability_details.append(detail); self._source_identities.add(source_identity)
        self._station_before.append(station_before); self._station_after.append(next_reference.detach().clone())
        self._reset_masks.append(reset_mask.detach().clone()); self._returned_episode_lengths.append(returned_episode.detach().clone())
        self._last_policy = OrderedDict((name, value.detach().clone()) for name, value in pre_action_policy.items())
        self._expected_next_policy = _policy_snapshot(returned_policy)
        self._pending_policy_hash = None

    def finalize(
        self,
        provenance: Mapping[str, object],
        metric_config: EvaluationMetricConfig = EvaluationMetricConfig(),
    ) -> OrderedDict[str, object]:
        if self._pending_policy_hash is not None or len(self._rows) != self.config.expected_policy_steps or self._last_policy is None:
            raise RuntimeError("runtime coverage is incomplete or has an unconsumed policy")
        evaluation = OrderedDict((name, torch.stack([row[name] for row in self._rows], dim=0)) for name in EVALUATION_INPUT_KEYS)
        observability_details = OrderedDict(
            (name, torch.stack([row[name] for row in self._observability_details], dim=0))
            for name in EXACT_OBSERVABILITY_DETAIL_KEYS
        )
        record: EvaluationRecord = build_evaluation_record(OrderedDict((("policy", self._last_policy), ("evaluation", evaluation))), provenance, metric_config)
        reset_masks = torch.stack(self._reset_masks)
        tree = OrderedDict((
            ("schema_version", "m7.runtime-evaluation.v2"),
            ("contract", OrderedDict((("num_envs", self.config.num_envs), ("policy_steps", self.config.expected_policy_steps), ("samples", self.config.num_envs * self.config.expected_policy_steps), ("rays_per_environment", self.config.rays_per_environment), ("ray_samples", self.config.num_envs * self.config.expected_policy_steps * self.config.rays_per_environment)))),
            ("source_manifest", _owned_source_manifest()),
            ("policy_hashes", tuple(self._policy_hashes)),
            ("action_hashes", tuple(self._action_hashes)),
            ("station_reference_before_w_m", torch.stack(self._station_before).detach().cpu()),
            ("station_reference_after_w_m", torch.stack(self._station_after).detach().cpu()),
            ("reset_masks", reset_masks.detach().cpu()),
            ("returned_episode_lengths", torch.stack(self._returned_episode_lengths).detach().cpu()),
            ("evaluation_inputs", OrderedDict((name, value.detach().cpu()) for name, value in evaluation.items())),
            ("observability_details", OrderedDict((name, value.detach().cpu()) for name, value in observability_details.items())),
            ("canonical_record", record.canonical_json),
            ("gates", OrderedDict((("coverage_exact", True), ("all_finite", True), ("observability_exact", True), ("reset_count", int(reset_masks.sum().item()))))),
        ))
        finite_tensors = tuple(tree["evaluation_inputs"].values()) + tuple(tree["observability_details"].values())
        if tuple(tree) != RUNTIME_TREE_KEYS or not all(
            torch.isfinite(value).all() for value in finite_tensors if value.is_floating_point()
        ):
            raise RuntimeError("runtime output tree contract violation")
        json.loads(record.canonical_json)
        return tree


__all__ = [
    "ANCHOR_KEYS", "CONTROL_KEYS", "EXACT_OBSERVABILITY_DETAIL_KEYS", "ExactRuntimeObservability",
    "ICS_KEYS", "PRE_RESET_KEYS", "RUNTIME_TREE_KEYS", "RuntimeEvaluationCollector", "RuntimeEvaluationConfig",
    "SOURCE_MANIFEST", "SOURCE_METADATA_KEYS", "TASK_KEYS", "distill_exact_runtime_observability",
    "TERMINATION_KEYS", "TERMINATION_MASK_KEYS",
]
