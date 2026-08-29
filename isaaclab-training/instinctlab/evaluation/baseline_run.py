"""Exact environment-independent multi-step B0 run lifecycle and receipt."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import struct

import torch

from instinctlab.control.baseline_b0 import B0DirectConfig

from .baseline_controller_execution import (
    B0ControllerExecutionEnvelope,
    execute_ready_b0_controller,
)
from .baselines import (
    ActorAuditStatus,
    BaselineId,
    BaselineRunManifest,
    CapabilityBinding,
    ComponentToggles,
    ExecutionStatus,
    MANIFEST_PAYLOAD_FIELD_ORDER,
    PrivilegeClass,
    ReadinessState,
    resolve_baseline,
    validate_capability_bindings,
)


B0_RUN_RECEIPT_SCHEMA_VERSION = "m7.b0-run-receipt.v1"
B0_RUN_TRAJECTORY_SCHEMA_VERSION = "m7.b0-run-trajectory.v1"
_ACTIVE = "ACTIVE"
_FINALIZED = "FINALIZED"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SESSION_TOKEN = object()
_LEE_FIELDS = (
    "normalized_rotor_commands",
    "held_position_w_m",
    "held_yaw_rad",
    "desired_rotation_w",
    "collective_thrust_n",
)


def _owned_direct_config(config: B0DirectConfig) -> B0DirectConfig:
    if type(config) is not B0DirectConfig:
        raise TypeError("direct_config must be an exact B0DirectConfig")
    return B0DirectConfig(config.velocity_limit_m_s)


@dataclass(frozen=True, slots=True)
class B0RunConfig:
    """Exact positive run length and immutable B0 direct configuration."""

    expected_steps: int
    direct_config: B0DirectConfig

    def __post_init__(self) -> None:
        if type(self.expected_steps) is not int:
            raise TypeError("expected_steps must be an exact integer")
        if self.expected_steps <= 0:
            raise ValueError("expected_steps must be positive")
        object.__setattr__(self, "direct_config", _owned_direct_config(self.direct_config))


def _bindings_from_mapping(mapping: dict[str, object]) -> tuple[CapabilityBinding, ...]:
    raw_bindings = mapping["capability_bindings"]
    if type(raw_bindings) is not list:
        raise TypeError("B0 capability_bindings must be a canonical JSON array")
    return tuple(
        CapabilityBinding(item["name"], item["provider"], item["evidence_sha256"])
        for item in raw_bindings
    )


def _require_ready_manifest(manifest: BaselineRunManifest) -> BaselineRunManifest:
    if type(manifest) is not BaselineRunManifest:
        raise TypeError("manifest must be an exact BaselineRunManifest")
    owned = BaselineRunManifest.from_json(manifest.canonical_json)
    mapping = owned.to_mapping()
    baseline = resolve_baseline(BaselineId.B0)
    if mapping["baseline_id"] != BaselineId.B0.value:
        raise ValueError("run manifest must identify exact B0")
    if mapping["config_switch"] != baseline.config_switch:
        raise ValueError("B0 run manifest must use the exact direct_velocity switch")
    if (
        baseline.readiness is not ReadinessState.READY
        or not baseline.deployable
        or baseline.privilege_class is not PrivilegeClass.ACTOR_CLEAN
    ):
        raise ValueError("B0 registry row must remain READY, deployable, and actor-clean")
    if baseline.actor_input_fields != ("commanded_body_velocity_m_s",):
        raise ValueError("B0 run actor schema must remain command-only")
    if baseline.component_toggles != ComponentToggles(False, False, False, False):
        raise ValueError("B0 run component toggles must all remain false")
    bindings = _bindings_from_mapping(mapping)
    validate_capability_bindings(baseline, bindings, ExecutionStatus.READY)
    if mapping["actor_audit_status"] != ActorAuditStatus.PASS.value:
        raise ValueError("B0 run requires actor audit PASS")
    if mapping["execution_status"] != ExecutionStatus.READY.value:
        raise ValueError("B0 run accepts only a READY manifest")
    if mapping["skip_reason"] is not None:
        raise ValueError("B0 READY run cannot contain a skip reason")
    return owned


def _require_executed_successor(
    ready_manifest: BaselineRunManifest,
    executed_manifest: BaselineRunManifest,
) -> BaselineRunManifest:
    if type(executed_manifest) is not BaselineRunManifest:
        raise TypeError("executed successor must be an exact BaselineRunManifest")
    owned = BaselineRunManifest.from_json(executed_manifest.canonical_json)
    ready_mapping = ready_manifest.to_mapping()
    executed_mapping = owned.to_mapping()
    if executed_mapping["baseline_id"] != BaselineId.B0.value:
        raise ValueError("executed successor must identify exact B0")
    if executed_mapping["execution_status"] != ExecutionStatus.EXECUTED.value:
        raise ValueError("run successor must be EXECUTED")
    for name in MANIFEST_PAYLOAD_FIELD_ORDER:
        if name != "execution_status" and executed_mapping[name] != ready_mapping[name]:
            raise ValueError(f"run successor changed immutable payload field {name}")
    if owned.manifest_sha256 == ready_manifest.manifest_sha256:
        raise ValueError("READY and EXECUTED run manifests must have distinct hashes")
    return owned


@dataclass(frozen=True, slots=True)
class B0RunReceipt:
    """Owned manifest identities and deterministic hash for one completed run."""

    schema_version: str
    ready_manifest: BaselineRunManifest
    executed_manifest: BaselineRunManifest
    step_count: int
    trajectory_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or (
            self.schema_version != B0_RUN_RECEIPT_SCHEMA_VERSION
        ):
            raise ValueError("unknown B0 run receipt schema_version")
        ready = _require_ready_manifest(self.ready_manifest)
        executed = _require_executed_successor(ready, self.executed_manifest)
        if type(self.step_count) is not int:
            raise TypeError("receipt step_count must be an exact integer")
        if self.step_count <= 0:
            raise ValueError("receipt step_count must be positive")
        if type(self.trajectory_sha256) is not str or (
            _SHA256_PATTERN.fullmatch(self.trajectory_sha256) is None
        ):
            raise ValueError("trajectory_sha256 must be an exact lowercase SHA-256")
        object.__setattr__(self, "ready_manifest", ready)
        object.__setattr__(self, "executed_manifest", executed)


def _u64(value: int) -> bytes:
    if type(value) is not int or value < 0 or value >= 2**64:
        raise ValueError("framed integer must fit unsigned 64-bit")
    return struct.pack("<Q", value)


def _blob(payload: bytes) -> bytes:
    if type(payload) is not bytes:
        raise TypeError("framed payload must be exact bytes")
    return _u64(len(payload)) + payload


def _field(name: str, payload: bytes) -> bytes:
    if type(name) is not str or not name or not name.isascii():
        raise ValueError("frame field name must be nonempty ASCII")
    return _blob(name.encode("ascii")) + _blob(payload)


def _tensor_payload(tensor: torch.Tensor) -> bytes:
    if type(tensor) is not torch.Tensor:
        raise TypeError("trajectory tensor must be an exact torch.Tensor")
    if tensor.dtype is torch.float32:
        dtype_token, format_code = b"float32-le", "f"
    elif tensor.dtype is torch.float64:
        dtype_token, format_code = b"float64-le", "d"
    else:
        raise TypeError("trajectory tensors must use float32 or float64")
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise ValueError("trajectory tensors must be detached and gradient-free")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("trajectory tensors must be finite")
    owned = tensor.detach().to(device="cpu").contiguous()
    values = owned.reshape(-1).tolist()
    raw = struct.pack(f"<{len(values)}{format_code}", *values)
    shape = _u64(owned.ndim) + b"".join(_u64(dimension) for dimension in owned.shape)
    return _blob(dtype_token) + shape + _blob(raw)


def _tensor_field(name: str, tensor: torch.Tensor) -> bytes:
    return _field(name, _tensor_payload(tensor))


def _initial_digest(
    ready_manifest: BaselineRunManifest,
    config: B0RunConfig,
) -> object:
    digest = hashlib.sha256()
    digest.update(_field("schema_version", B0_RUN_TRAJECTORY_SCHEMA_VERSION.encode("ascii")))
    digest.update(_field("ready_manifest_sha256", ready_manifest.manifest_sha256.encode("ascii")))
    digest.update(_field("expected_steps", _u64(config.expected_steps)))
    digest.update(
        _field(
            "velocity_limit_m_s",
            struct.pack("<d", config.direct_config.velocity_limit_m_s),
        )
    )
    return digest


def _encode_step(
    step_index: int,
    ready_manifest: BaselineRunManifest,
    executed_manifest: BaselineRunManifest,
    commanded_body_velocity_m_s: torch.Tensor,
    controller_private_root_state_w: torch.Tensor,
    envelope: B0ControllerExecutionEnvelope,
) -> bytes:
    frame = bytearray()
    frame.extend(_field("step_index", _u64(step_index)))
    frame.extend(
        _field("ready_manifest_sha256", ready_manifest.manifest_sha256.encode("ascii"))
    )
    frame.extend(
        _field(
            "executed_manifest_sha256",
            executed_manifest.manifest_sha256.encode("ascii"),
        )
    )
    frame.extend(
        _tensor_field("actor.commanded_body_velocity_m_s", commanded_body_velocity_m_s)
    )
    frame.extend(
        _tensor_field("controller_private.root_state_w", controller_private_root_state_w)
    )
    frame.extend(
        _tensor_field(
            "b0.issued_body_velocity_m_s",
            envelope.b0_execution.result.issued_body_velocity_m_s,
        )
    )
    frame.extend(
        _tensor_field(
            "b0.issued_world_velocity_m_s",
            envelope.b0_execution.result.issued_world_velocity_m_s,
        )
    )
    for name in _LEE_FIELDS:
        frame.extend(_tensor_field(f"lee.{name}", getattr(envelope.lee_result, name)))
    return bytes(frame)


class B0RunSession:
    """Exact-type, bounded, atomic B0 run state machine."""

    __slots__ = (
        "_ready_manifest",
        "_config",
        "_expected_steps",
        "_next_step",
        "_state",
        "_digest",
        "_executed_manifest",
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("B0RunSession must not be subclassed")

    def __init__(
        self,
        ready_manifest: BaselineRunManifest,
        config: B0RunConfig,
        token: object,
    ) -> None:
        if token is not _SESSION_TOKEN:
            raise TypeError("B0RunSession must be created by begin_ready_b0_run")
        self._ready_manifest = ready_manifest
        self._config = config
        self._expected_steps = config.expected_steps
        self._next_step = 0
        self._state = _ACTIVE
        self._digest = _initial_digest(ready_manifest, config)
        self._executed_manifest = None

    @property
    def state_token(self) -> str:
        return self._state

    @property
    def next_step(self) -> int:
        return self._next_step

    @property
    def expected_steps(self) -> int:
        return self._expected_steps

    @property
    def finalized(self) -> bool:
        return self._state == _FINALIZED

    def step(
        self,
        commanded_body_velocity_m_s: torch.Tensor,
        controller_private_root_state_w: torch.Tensor,
    ) -> B0ControllerExecutionEnvelope:
        if type(self) is not B0RunSession:
            raise TypeError("step requires an exact B0RunSession")
        if self._state != _ACTIVE:
            raise RuntimeError("finalized B0 run cannot execute another step")
        if self._next_step >= self._expected_steps:
            raise RuntimeError("B0 run reached expected_steps and must be finalized")
        envelope = execute_ready_b0_controller(
            self._ready_manifest,
            commanded_body_velocity_m_s,
            controller_private_root_state_w,
            self._config.direct_config,
        )
        if type(envelope) is not B0ControllerExecutionEnvelope:
            raise TypeError("controller handoff must return exact B0ControllerExecutionEnvelope")
        if type(envelope.b0_execution.ready_manifest) is not BaselineRunManifest or (
            envelope.b0_execution.ready_manifest.canonical_json
            != self._ready_manifest.canonical_json
        ):
            raise ValueError("controller handoff changed the owned READY manifest")
        staged_successor = _require_executed_successor(
            self._ready_manifest,
            envelope.b0_execution.executed_manifest,
        )
        if self._executed_manifest is not None and (
            staged_successor.canonical_json != self._executed_manifest.canonical_json
        ):
            raise ValueError("controller handoff EXECUTED successor drifted within run")
        command_snapshot = commanded_body_velocity_m_s.detach().clone()
        root_snapshot = controller_private_root_state_w.detach().clone()
        staged_digest = self._digest.copy()
        staged_digest.update(
            _encode_step(
                self._next_step,
                self._ready_manifest,
                staged_successor,
                command_snapshot,
                root_snapshot,
                envelope,
            )
        )
        self._digest = staged_digest
        if self._executed_manifest is None:
            self._executed_manifest = staged_successor
        self._next_step += 1
        return envelope

    def finalize(self) -> B0RunReceipt:
        if type(self) is not B0RunSession:
            raise TypeError("finalize requires an exact B0RunSession")
        if self._state != _ACTIVE:
            raise RuntimeError("B0 run may be finalized exactly once")
        if self._next_step != self._expected_steps:
            raise RuntimeError("B0 run cannot finalize before expected_steps")
        if self._executed_manifest is None:
            raise RuntimeError("B0 run cannot finalize without a successor")
        receipt = B0RunReceipt(
            schema_version=B0_RUN_RECEIPT_SCHEMA_VERSION,
            ready_manifest=BaselineRunManifest.from_json(
                self._ready_manifest.canonical_json
            ),
            executed_manifest=BaselineRunManifest.from_json(
                self._executed_manifest.canonical_json
            ),
            step_count=self._next_step,
            trajectory_sha256=self._digest.hexdigest(),
        )
        self._state = _FINALIZED
        return receipt


def begin_ready_b0_run(
    manifest: BaselineRunManifest,
    config: B0RunConfig,
) -> B0RunSession:
    """Own one exact READY manifest and begin a zero-step active B0 run."""

    if type(config) is not B0RunConfig:
        raise TypeError("config must be an exact B0RunConfig")
    owned_manifest = _require_ready_manifest(manifest)
    owned_config = B0RunConfig(config.expected_steps, config.direct_config)
    return B0RunSession(owned_manifest, owned_config, _SESSION_TOKEN)


__all__ = [
    "B0_RUN_RECEIPT_SCHEMA_VERSION",
    "B0_RUN_TRAJECTORY_SCHEMA_VERSION",
    "B0RunConfig",
    "B0RunReceipt",
    "B0RunSession",
    "begin_ready_b0_run",
]
