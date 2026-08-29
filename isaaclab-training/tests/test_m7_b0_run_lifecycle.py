"""Focused exact B0 multi-step run lifecycle and receipt tests."""

from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import FrozenInstanceError, fields
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import struct
import weakref

import pytest
import torch

import instinctlab.evaluation as evaluation
import instinctlab.evaluation.baseline_run as run_module
from instinctlab.control.baseline_b0 import B0DirectConfig
from instinctlab.evaluation.baseline_controller_execution import (
    B0ControllerExecutionEnvelope,
    execute_ready_b0_controller,
)
from instinctlab.evaluation.baseline_execution import B0ExecutionEnvelope
from instinctlab.evaluation.baseline_run import (
    B0_RUN_RECEIPT_SCHEMA_VERSION,
    B0_RUN_TRAJECTORY_SCHEMA_VERSION,
    B0RunConfig,
    B0RunReceipt,
    B0RunSession,
    begin_ready_b0_run,
)
from instinctlab.evaluation.baselines import (
    ActorAuditStatus,
    BaselineRunManifest,
    CapabilityBinding,
    ExecutionStatus,
    MANIFEST_PAYLOAD_FIELD_ORDER,
    build_run_manifest,
    resolve_baseline,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_COMMIT = "7f14cf2c1e71ac275ba9e2092a948d2ac0c504a4"
LEE_FIELDS = (
    "normalized_rotor_commands",
    "held_position_w_m",
    "held_yaw_rad",
    "desired_rotation_w",
    "collective_thrust_n",
)


def _bindings() -> tuple[CapabilityBinding, ...]:
    return tuple(
        CapabilityBinding(item.name, item.accepted_provider, item.accepted_evidence_sha256)
        for item in resolve_baseline("B0").capability_requirements
    )


def _manifest(
    status: ExecutionStatus = ExecutionStatus.READY,
    *,
    seed: int = 31,
    product_commit: str = PRODUCT_COMMIT,
) -> BaselineRunManifest:
    row = resolve_baseline("B0")
    ready = status in (ExecutionStatus.READY, ExecutionStatus.EXECUTED)
    return build_run_manifest(
        product_commit=product_commit,
        baseline_id="B0",
        config_switch=row.config_switch,
        capability_bindings=_bindings() if ready else (),
        scenario_id="M7-010.lifecycle",
        seed=seed,
        actor_audit_status=ActorAuditStatus.PASS if ready else ActorAuditStatus.NOT_RUN,
        execution_status=status,
        skip_reason=None,
    )


def _root(dtype: torch.dtype, *, batch: int = 1, yaw_degrees: float = 0.0) -> torch.Tensor:
    root = torch.zeros((batch, 13), dtype=dtype)
    half = math.radians(yaw_degrees) * 0.5
    root[:, 3] = math.cos(half)
    root[:, 6] = math.sin(half)
    root[:, 7:10] = torch.tensor((0.1, -0.2, 0.05), dtype=dtype)
    return root


def _command(dtype: torch.dtype, values: tuple[float, float, float], batch: int = 1) -> torch.Tensor:
    return torch.tensor((values,) * batch, dtype=dtype)


def _state_fingerprint(session: B0RunSession) -> tuple[object, ...]:
    successor = session._executed_manifest
    return (
        session.state_token,
        session.next_step,
        session.expected_steps,
        session.finalized,
        session._digest.copy().hexdigest(),
        successor.canonical_json if successor is not None else None,
        session._ready_manifest.canonical_json,
        session._config.expected_steps,
        session._config.direct_config.velocity_limit_m_s,
    )


def _oracle_u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _oracle_blob(value: bytes) -> bytes:
    return _oracle_u64(len(value)) + value


def _oracle_field(name: str, value: bytes) -> bytes:
    return _oracle_blob(name.encode("ascii")) + _oracle_blob(value)


def _oracle_tensor(tensor: torch.Tensor) -> bytes:
    owned = tensor.detach().cpu().contiguous()
    if owned.dtype is torch.float32:
        token, code = b"float32-le", "f"
    elif owned.dtype is torch.float64:
        token, code = b"float64-le", "d"
    else:
        raise TypeError
    values = owned.reshape(-1).tolist()
    data = struct.pack(f"<{len(values)}{code}", *values)
    shape = _oracle_u64(owned.ndim) + b"".join(_oracle_u64(item) for item in owned.shape)
    return _oracle_blob(token) + shape + _oracle_blob(data)


def _independent_hash(
    manifest: BaselineRunManifest,
    config: B0RunConfig,
    trajectory: list[tuple[torch.Tensor, torch.Tensor, B0ControllerExecutionEnvelope]],
    *,
    command_field_name: str = "actor.commanded_body_velocity_m_s",
) -> str:
    digest = hashlib.sha256()
    digest.update(_oracle_field("schema_version", B0_RUN_TRAJECTORY_SCHEMA_VERSION.encode("ascii")))
    digest.update(_oracle_field("ready_manifest_sha256", manifest.manifest_sha256.encode("ascii")))
    digest.update(_oracle_field("expected_steps", _oracle_u64(config.expected_steps)))
    digest.update(_oracle_field("velocity_limit_m_s", struct.pack("<d", config.direct_config.velocity_limit_m_s)))
    for index, (command, root, envelope) in enumerate(trajectory):
        digest.update(_oracle_field("step_index", _oracle_u64(index)))
        digest.update(_oracle_field("ready_manifest_sha256", manifest.manifest_sha256.encode("ascii")))
        digest.update(_oracle_field("executed_manifest_sha256", envelope.b0_execution.executed_manifest.manifest_sha256.encode("ascii")))
        tensors = (
            (command_field_name, command),
            ("controller_private.root_state_w", root),
            ("b0.issued_body_velocity_m_s", envelope.b0_execution.result.issued_body_velocity_m_s),
            ("b0.issued_world_velocity_m_s", envelope.b0_execution.result.issued_world_velocity_m_s),
            *((f"lee.{name}", getattr(envelope.lee_result, name)) for name in LEE_FIELDS),
        )
        for name, tensor in tensors:
            digest.update(_oracle_field(name, _oracle_tensor(tensor)))
    return digest.hexdigest()


def _execute(
    commands: list[torch.Tensor],
    roots: list[torch.Tensor],
    *,
    manifest: BaselineRunManifest | None = None,
    limit: float = 2.0,
) -> tuple[B0RunSession, list[B0ControllerExecutionEnvelope], B0RunReceipt]:
    ready = _manifest() if manifest is None else manifest
    session = begin_ready_b0_run(ready, B0RunConfig(len(commands), B0DirectConfig(limit)))
    envelopes = [session.step(command, root) for command, root in zip(commands, roots, strict=True)]
    return session, envelopes, session.finalize()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_one_step_and_multi_step_hand_oracles_direct_parity(dtype: torch.dtype) -> None:
    ready = _manifest()
    commands = [
        _command(dtype, (0.0, 0.0, 0.0)),
        _command(dtype, (0.5, -0.25, 0.75), batch=2),
        _command(dtype, (3.0, 4.0, 0.0)),
    ]
    roots = [
        _root(dtype),
        _root(dtype, batch=2, yaw_degrees=90.0),
        _root(dtype, yaw_degrees=35.0),
    ]
    config = B0RunConfig(3, B0DirectConfig(2.0))
    session = begin_ready_b0_run(ready, config)
    assert type(session) is B0RunSession
    assert (session.state_token, session.next_step, session.expected_steps, session.finalized) == (
        "ACTIVE", 0, 3, False,
    )
    trajectory = []
    successor_json = None
    for index, (command, root) in enumerate(zip(commands, roots, strict=True)):
        envelope = session.step(command, root)
        direct = execute_ready_b0_controller(ready, command, root, config.direct_config)
        assert type(envelope) is B0ControllerExecutionEnvelope
        assert envelope.b0_execution.executed_manifest.canonical_json == direct.b0_execution.executed_manifest.canonical_json
        assert torch.equal(envelope.b0_execution.result.issued_body_velocity_m_s, direct.b0_execution.result.issued_body_velocity_m_s)
        assert torch.equal(envelope.b0_execution.result.issued_world_velocity_m_s, direct.b0_execution.result.issued_world_velocity_m_s)
        for name in LEE_FIELDS:
            assert torch.equal(getattr(envelope.lee_result, name), getattr(direct.lee_result, name))
        successor_json = successor_json or envelope.b0_execution.executed_manifest.canonical_json
        assert envelope.b0_execution.executed_manifest.canonical_json == successor_json
        assert session.next_step == index + 1 and not session.finalized
        trajectory.append((command, root, envelope))
    receipt = session.finalize()
    assert type(receipt) is B0RunReceipt
    assert receipt.schema_version == B0_RUN_RECEIPT_SCHEMA_VERSION
    assert receipt.step_count == 3
    assert receipt.ready_manifest.canonical_json == ready.canonical_json
    assert receipt.ready_manifest is not ready
    assert receipt.executed_manifest.canonical_json == successor_json
    assert receipt.ready_manifest is not receipt.executed_manifest
    assert receipt.trajectory_sha256 == _independent_hash(ready, config, trajectory)
    assert (session.state_token, session.next_step, session.finalized) == ("FINALIZED", 3, True)

    one_command, one_root = commands[:1], roots[:1]
    _, one_envelope, one_receipt = _execute(one_command, one_root)
    one_config = B0RunConfig(1, B0DirectConfig(2.0))
    assert one_receipt.trajectory_sha256 == _independent_hash(
        ready, one_config, [(one_command[0], one_root[0], one_envelope[0])]
    )


def test_begin_and_lifecycle_guards_have_exact_zero_or_one_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    accepted = run_module.execute_ready_b0_controller

    def spy(*args):
        calls.append(args)
        return accepted(*args)

    monkeypatch.setattr(run_module, "execute_ready_b0_controller", spy)
    session = begin_ready_b0_run(_manifest(), B0RunConfig(1, B0DirectConfig()))
    assert calls == []
    with pytest.raises(RuntimeError, match="before expected_steps"):
        session.finalize()
    assert calls == []
    envelope = session.step(_command(torch.float64, (0.1, 0.2, 0.3)), _root(torch.float64))
    assert len(calls) == 1 and type(envelope) is B0ControllerExecutionEnvelope
    with pytest.raises(RuntimeError, match="must be finalized"):
        session.step(_command(torch.float64, (0.0, 0.0, 0.0)), _root(torch.float64))
    assert len(calls) == 1
    session.finalize()
    with pytest.raises(RuntimeError, match="exactly once"):
        session.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        session.step(_command(torch.float64, (0.0, 0.0, 0.0)), _root(torch.float64))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mutation",
    ("command_type", "root_shape", "dtype", "grad", "nonfinite", "alias"),
)
def test_accepted_handoff_rejection_is_one_call_and_atomic(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = run_module.execute_ready_b0_controller
    calls = []

    def spy(*args):
        calls.append(args)
        return accepted(*args)

    monkeypatch.setattr(run_module, "execute_ready_b0_controller", spy)
    session = begin_ready_b0_run(_manifest(), B0RunConfig(1, B0DirectConfig()))
    before = _state_fingerprint(session)
    command: object = _command(torch.float64, (0.1, 0.2, 0.3))
    root: object = _root(torch.float64)
    if mutation == "command_type": command = [[0.1, 0.2, 0.3]]
    elif mutation == "root_shape": root = torch.zeros((1, 12), dtype=torch.float64)
    elif mutation == "dtype": root = _root(torch.float32)
    elif mutation == "grad": command = command.requires_grad_()
    elif mutation == "nonfinite": root[0, 0] = float("nan")
    elif mutation == "alias":
        backing = torch.zeros((1, 13), dtype=torch.float64); backing[:, 3] = 1.0
        command, root = backing[:, :3], backing
    with pytest.raises((TypeError, ValueError)):
        session.step(command, root)  # type: ignore[arg-type]
    assert len(calls) == 1 and _state_fingerprint(session) == before
    valid_command, valid_root = _command(torch.float64, (0.1, 0.2, 0.3)), _root(torch.float64)
    session.step(valid_command, valid_root)
    receipt = session.finalize()
    _, _, clean = _execute([valid_command], [valid_root])
    assert receipt.trajectory_sha256 == clean.trajectory_sha256


def test_handoff_exception_hash_failure_and_successor_drift_are_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command, root = _command(torch.float64, (0.25, -0.5, 0.75)), _root(torch.float64)
    session = begin_ready_b0_run(_manifest(), B0RunConfig(1, B0DirectConfig()))
    before = _state_fingerprint(session)
    accepted = run_module.execute_ready_b0_controller
    failed_calls = []
    def fail_handoff(*args):
        failed_calls.append(args)
        raise RuntimeError("handoff failure")
    monkeypatch.setattr(run_module, "execute_ready_b0_controller", fail_handoff)
    with pytest.raises(RuntimeError, match="handoff failure"):
        session.step(command, root)
    assert len(failed_calls) == 1 and _state_fingerprint(session) == before
    monkeypatch.setattr(run_module, "execute_ready_b0_controller", accepted)
    original_encoder = run_module._encode_step
    monkeypatch.setattr(run_module, "_encode_step", lambda *args: (_ for _ in ()).throw(RuntimeError("frame failure")))
    with pytest.raises(RuntimeError, match="frame failure"):
        session.step(command, root)
    assert _state_fingerprint(session) == before
    monkeypatch.setattr(run_module, "_encode_step", original_encoder)
    session.step(command, root)
    receipt = session.finalize()
    _, _, clean = _execute([command], [root])
    assert receipt.trajectory_sha256 == clean.trajectory_sha256

    drift_session = begin_ready_b0_run(_manifest(), B0RunConfig(2, B0DirectConfig()))
    drift_session.step(command, root)
    drift_before = _state_fingerprint(drift_session)

    def drift(*args):
        result = accepted(*args)
        wrong = _manifest(ExecutionStatus.EXECUTED, seed=99)
        b0 = B0ExecutionEnvelope(
            result.b0_execution.ready_manifest,
            wrong,
            result.b0_execution.result,
        )
        return B0ControllerExecutionEnvelope(b0, result.lee_result)

    monkeypatch.setattr(run_module, "execute_ready_b0_controller", drift)
    with pytest.raises(ValueError, match="successor changed|drifted"):
        drift_session.step(command, root)
    assert _state_fingerprint(drift_session) == drift_before
    monkeypatch.setattr(run_module, "execute_ready_b0_controller", accepted)
    drift_session.step(command, root)
    drift_receipt = drift_session.finalize()
    _, _, clean_two = _execute([command, command], [root, root])
    assert drift_receipt.trajectory_sha256 == clean_two.trajectory_sha256


def test_invalid_construction_and_receipt_values_fail_closed_without_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(run_module, "execute_ready_b0_controller", lambda *args: calls.append(args))
    forged = object.__new__(BaselineRunManifest)
    object.__setattr__(forged, "canonical_json", "{}")
    for manifest in (None, object(), forged, _manifest(ExecutionStatus.NOT_READY), _manifest(ExecutionStatus.EXECUTED)):
        with pytest.raises((TypeError, ValueError)):
            begin_ready_b0_run(manifest, B0RunConfig(1, B0DirectConfig()))  # type: ignore[arg-type]
    for config in (None, object(), B0DirectConfig()):
        with pytest.raises(TypeError):
            begin_ready_b0_run(_manifest(), config)  # type: ignore[arg-type]
    assert calls == []
    for value in (True, False, 1.0, "1", None):
        with pytest.raises(TypeError):
            B0RunConfig(value, B0DirectConfig())  # type: ignore[arg-type]
    for value in (0, -1):
        with pytest.raises(ValueError):
            B0RunConfig(value, B0DirectConfig())
    with pytest.raises(TypeError):
        B0RunConfig(1, object())  # type: ignore[arg-type]


def test_config_session_and_receipt_are_exact_frozen_owned_records() -> None:
    direct = B0DirectConfig(2.0)
    config = B0RunConfig(1, direct)
    assert config.direct_config is not direct
    with pytest.raises(FrozenInstanceError):
        config.expected_steps = 2
    with pytest.raises(TypeError, match="begin_ready_b0_run"):
        B0RunSession(_manifest(), config, object())
    with pytest.raises(TypeError, match="subclassed"):
        class Child(B0RunSession):
            pass
    session, _, receipt = _execute(
        [_command(torch.float64, (0.0, 0.0, 0.0))], [_root(torch.float64)]
    )
    with pytest.raises(AttributeError):
        session.next_step = 9
    with pytest.raises(FrozenInstanceError):
        receipt.step_count = 2
    assert tuple(field.name for field in fields(B0RunReceipt)) == (
        "schema_version", "ready_manifest", "executed_manifest", "step_count",
        "trajectory_sha256",
    )
    assert not any(
        token in field.name
        for field in fields(B0RunReceipt)
        for token in ("root", "pose", "velocity", "tensor", "command", "history")
    )
    assert _state_fingerprint(session)[0] == "FINALIZED"


def test_receipt_constructor_validates_exact_owned_fields() -> None:
    ready = _manifest()
    _, _, receipt = _execute(
        [_command(torch.float64, (0.0, 0.0, 0.0))], [_root(torch.float64)],
        manifest=ready,
    )
    reconstructed = B0RunReceipt(
        receipt.schema_version,
        receipt.ready_manifest,
        receipt.executed_manifest,
        receipt.step_count,
        receipt.trajectory_sha256,
    )
    assert reconstructed.ready_manifest is not receipt.ready_manifest
    assert reconstructed.executed_manifest is not receipt.executed_manifest
    invalid = (
        ("m7.bad", receipt.ready_manifest, receipt.executed_manifest, 1, receipt.trajectory_sha256),
        (receipt.schema_version, receipt.executed_manifest, receipt.executed_manifest, 1, receipt.trajectory_sha256),
        (receipt.schema_version, receipt.ready_manifest, receipt.ready_manifest, 1, receipt.trajectory_sha256),
        (receipt.schema_version, receipt.ready_manifest, receipt.executed_manifest, 0, receipt.trajectory_sha256),
        (receipt.schema_version, receipt.ready_manifest, receipt.executed_manifest, 1, "A" * 64),
    )
    for candidate in invalid:
        with pytest.raises((TypeError, ValueError)):
            B0RunReceipt(*candidate)
    with pytest.raises(TypeError):
        B0RunReceipt(
            receipt.schema_version,
            receipt.ready_manifest,
            receipt.executed_manifest,
            True,
            receipt.trajectory_sha256,
        )


def test_no_tensor_retention_input_mutation_or_alias_and_receipt_privacy() -> None:
    ready = _manifest()
    session = begin_ready_b0_run(ready, B0RunConfig(1, B0DirectConfig()))
    command = _command(torch.float64, (0.4, -0.1, 0.2))
    root = _root(torch.float64, yaw_degrees=20.0)
    command_before, root_before = command.clone(), root.clone()
    command_ref, root_ref = weakref.ref(command), weakref.ref(root)
    envelope = session.step(command, root)
    assert torch.equal(command, command_before) and torch.equal(root, root_before)
    input_pointers = {command.untyped_storage().data_ptr(), root.untyped_storage().data_ptr()}
    output_pointers = {
        envelope.b0_execution.result.issued_body_velocity_m_s.untyped_storage().data_ptr(),
        envelope.b0_execution.result.issued_world_velocity_m_s.untyped_storage().data_ptr(),
        *(value.untyped_storage().data_ptr() for value in vars(envelope.lee_result).values()),
    }
    assert input_pointers.isdisjoint(output_pointers)
    assert all(not isinstance(getattr(session, slot), torch.Tensor) for slot in session.__slots__)
    del command, root, envelope
    gc.collect()
    assert command_ref() is None and root_ref() is None
    receipt = session.finalize()
    assert receipt.ready_manifest is not ready
    serialized = json.dumps({
        "schema_version": receipt.schema_version,
        "ready_manifest_sha256": receipt.ready_manifest.manifest_sha256,
        "executed_manifest_sha256": receipt.executed_manifest.manifest_sha256,
        "step_count": receipt.step_count,
        "trajectory_sha256": receipt.trajectory_sha256,
    }, sort_keys=True)
    assert not any(token in serialized for token in ("root_state", "position_w", "velocity_m_s", "tensor("))


def test_independent_hash_oracle_determinism_and_sensitivity() -> None:
    dtype = torch.float64
    commands = [_command(dtype, (0.25, -0.5, 0.75)), _command(dtype, (3.0, 4.0, 0.0), batch=2)]
    roots = [_root(dtype, yaw_degrees=10.0), _root(dtype, batch=2, yaw_degrees=90.0)]
    ready = _manifest()
    session, envelopes, receipt = _execute(commands, roots, manifest=ready)
    config = B0RunConfig(2, B0DirectConfig(2.0))
    trajectory = list(zip(commands, roots, envelopes, strict=True))
    oracle = _independent_hash(ready, config, trajectory)
    assert receipt.trajectory_sha256 == oracle
    _, _, repeated = _execute([item.clone() for item in commands], [item.clone() for item in roots], manifest=_manifest())
    assert repeated.trajectory_sha256 == receipt.trajectory_sha256
    assert _independent_hash(ready, config, list(reversed(trajectory))) != oracle
    assert _independent_hash(ready, config, trajectory, command_field_name="actor.command_alias") != oracle
    changed_command = [item.clone() for item in commands]; changed_command[0][0, 0] = torch.nextafter(changed_command[0][0, 0], torch.tensor(float("inf"), dtype=dtype))
    _, changed_envelopes, changed_receipt = _execute(changed_command, roots, manifest=ready)
    assert changed_receipt.trajectory_sha256 != oracle
    assert _independent_hash(ready, config, list(zip(changed_command, roots, changed_envelopes, strict=True))) == changed_receipt.trajectory_sha256
    float32_commands = [item.float() for item in commands]; float32_roots = [item.float() for item in roots]
    _, _, float32_receipt = _execute(float32_commands, float32_roots, manifest=ready)
    assert float32_receipt.trajectory_sha256 != oracle
    batch_command = [_command(dtype, (0.25, -0.5, 0.75), batch=2), commands[1]]
    batch_root = [_root(dtype, batch=2, yaw_degrees=10.0), roots[1]]
    _, _, batch_receipt = _execute(batch_command, batch_root, manifest=ready)
    assert batch_receipt.trajectory_sha256 != oracle
    _, _, manifest_receipt = _execute(commands, roots, manifest=_manifest(seed=32))
    assert manifest_receipt.trajectory_sha256 != oracle
    _, _, one_step_receipt = _execute(commands[:1], roots[:1], manifest=ready)
    assert one_step_receipt.trajectory_sha256 != oracle
    assert session.finalized


def test_exact_api_exports_imports_framing_and_actor_boundary() -> None:
    assert tuple(inspect.signature(B0RunConfig).parameters) == ("expected_steps", "direct_config")
    assert tuple(inspect.signature(begin_ready_b0_run).parameters) == ("manifest", "config")
    assert tuple(inspect.signature(B0RunSession.step).parameters) == (
        "self", "commanded_body_velocity_m_s", "controller_private_root_state_w",
    )
    assert tuple(inspect.signature(B0RunSession.finalize).parameters) == ("self",)
    assert all(
        parameter.default is inspect.Parameter.empty
        for callable_object in (B0RunConfig, begin_ready_b0_run, B0RunSession.step)
        for parameter in inspect.signature(callable_object).parameters.values()
        if parameter.name != "self"
    )
    assert evaluation.B0RunConfig is B0RunConfig
    assert evaluation.B0RunReceipt is B0RunReceipt
    assert evaluation.B0RunSession is B0RunSession
    assert evaluation.begin_ready_b0_run is begin_ready_b0_run
    assert evaluation.B0_RUN_RECEIPT_SCHEMA_VERSION == B0_RUN_RECEIPT_SCHEMA_VERSION
    path = ROOT / "instinctlab/evaluation/baseline_run.py"
    source = path.read_text(); tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom): imports.add(("." * node.level) + (node.module or ""))
    assert imports == {
        "__future__", "dataclasses", "hashlib", "re", "struct", "torch",
        "instinctlab.control.baseline_b0", ".baseline_controller_execution", ".baselines",
    }
    assert "instinctlab.control" not in imports
    assert "struct.pack(\"<Q\"" in source
    assert 'b"float32-le"' in source and 'b"float64-le"' in source
    assert '"actor.commanded_body_velocity_m_s"' in source
    assert '"controller_private.root_state_w"' in source
    assert resolve_baseline("B0").actor_input_fields == ("commanded_body_velocity_m_s",)
    public = {name for name, _ in inspect.getmembers(B0RunSession) if not name.startswith("_")}
    assert public == {"expected_steps", "finalize", "finalized", "next_step", "state_token", "step"}


def test_resigned_stale_five_manifest_rejects_before_session_creation() -> None:
    mapping = OrderedDict(_manifest().to_mapping())
    mapping["capability_bindings"] = mapping["capability_bindings"][:5]
    payload = OrderedDict((name, mapping[name]) for name in MANIFEST_PAYLOAD_FIELD_ORDER)
    mapping["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    forged = object.__new__(BaselineRunManifest)
    object.__setattr__(forged, "canonical_json", json.dumps(mapping, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    with pytest.raises(ValueError):
        begin_ready_b0_run(forged, B0RunConfig(1, B0DirectConfig()))
