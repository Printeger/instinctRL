"""Focused exact B0 manifest-to-TASLAB-controller handoff tests."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import inspect
import math
from pathlib import Path

import pytest
import torch

import instinctlab.evaluation as evaluation
import instinctlab.evaluation.baseline_controller_execution as bridge_module
from instinctlab.control.baseline_b0 import B0DirectConfig
from instinctlab.control.lee_velocity_controller import taslab_velocity_control
from instinctlab.evaluation.baseline_controller_execution import (
    B0ControllerExecutionEnvelope,
    execute_ready_b0_controller,
)
from instinctlab.evaluation.baseline_execution import execute_ready_b0
from instinctlab.evaluation.baselines import (
    ActorAuditStatus,
    BaselineRunManifest,
    CapabilityBinding,
    ExecutionStatus,
    build_run_manifest,
    resolve_baseline,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_COMMIT = "1e6886c44b5893f4ba6490f7a08ba9b21b7a539d"


def _manifest(status: ExecutionStatus = ExecutionStatus.READY) -> BaselineRunManifest:
    row = resolve_baseline("B0")
    bindings = tuple(
        CapabilityBinding(item.name, item.accepted_provider, item.accepted_evidence_sha256)
        for item in row.capability_requirements
    ) if status in (ExecutionStatus.READY, ExecutionStatus.EXECUTED) else ()
    return build_run_manifest(
        product_commit=PRODUCT_COMMIT,
        baseline_id="B0",
        config_switch=row.config_switch,
        capability_bindings=bindings,
        scenario_id="S1.corner",
        seed=23,
        actor_audit_status=(ActorAuditStatus.PASS if bindings else ActorAuditStatus.NOT_RUN),
        execution_status=status,
        skip_reason=None,
    )


def _non_b0_manifest() -> BaselineRunManifest:
    row = resolve_baseline("B1")
    return build_run_manifest(
        product_commit=PRODUCT_COMMIT,
        baseline_id="B1",
        config_switch=row.config_switch,
        capability_bindings=(),
        scenario_id="S1.corner",
        seed=23,
        actor_audit_status=ActorAuditStatus.NOT_RUN,
        execution_status=ExecutionStatus.NOT_READY,
        skip_reason=None,
    )


def _malformed_manifest() -> BaselineRunManifest:
    manifest = object.__new__(BaselineRunManifest)
    object.__setattr__(manifest, "canonical_json", "{}")
    return manifest


def _root(dtype: torch.dtype, yaw_degrees: float = 0.0, batch: int = 3) -> torch.Tensor:
    root = torch.zeros((batch, 13), dtype=dtype)
    half = math.radians(yaw_degrees) * 0.5
    root[:, 3] = math.cos(half)
    root[:, 6] = math.sin(half)
    root[:, 7:10] = torch.tensor((0.1, -0.2, 0.05), dtype=dtype)
    return root


class _TensorSubclass(torch.Tensor):
    pass


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("yaw", (0.0, 90.0))
def test_exact_direct_chain_parity_batch_and_oracles(dtype: torch.dtype, yaw: float) -> None:
    command = torch.tensor(((0.0, 0.0, 0.0), (0.5, -0.25, 0.75), (3.0, 4.0, 0.0)), dtype=dtype)
    root = _root(dtype, yaw)
    result = execute_ready_b0_controller(_manifest(), command, root, B0DirectConfig(2.0))
    direct_b0 = execute_ready_b0(_manifest(), command, root[:, 3:7].clone(), B0DirectConfig(2.0))
    direct_lee = taslab_velocity_control(root.clone(), direct_b0.result.issued_world_velocity_m_s)
    assert type(result) is B0ControllerExecutionEnvelope
    assert result.b0_execution.executed_manifest.canonical_json == direct_b0.executed_manifest.canonical_json
    assert torch.equal(result.b0_execution.result.issued_body_velocity_m_s, direct_b0.result.issued_body_velocity_m_s)
    assert torch.equal(result.b0_execution.result.issued_world_velocity_m_s, direct_b0.result.issued_world_velocity_m_s)
    for name in vars(direct_lee):
        assert torch.equal(getattr(result.lee_result, name), getattr(direct_lee, name))


def test_exact_call_order_target_identity_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    accepted_b0 = bridge_module.execute_ready_b0
    accepted_lee = bridge_module.taslab_velocity_control
    world_target = []

    def b0_spy(*args):
        calls.append("b0")
        result = accepted_b0(*args)
        world_target.append(result.result.issued_world_velocity_m_s)
        return result

    def lee_spy(root, target):
        calls.append("lee")
        assert target is world_target[0]
        return accepted_lee(root, target)

    monkeypatch.setattr(bridge_module, "execute_ready_b0", b0_spy)
    monkeypatch.setattr(bridge_module, "taslab_velocity_control", lee_spy)
    execute_ready_b0_controller(_manifest(), torch.zeros((1, 3)), _root(torch.float32, batch=1), B0DirectConfig())
    assert calls == ["b0", "lee"]


@pytest.mark.parametrize(
    "mutation",
    ("manifest_type", "command_type", "root_type", "config_type", "command_subclass",
     "root_subclass", "command_rank", "command_width",
     "empty", "root_width", "batch", "dtype", "grad_command", "grad_root",
     "nan_command", "inf_root", "nonunit", "alias"),
)
def test_bridge_owned_rejections_call_neither_accepted_function(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest: object = _manifest()
    command: object = torch.zeros((2, 3))
    root: object = _root(torch.float32, batch=2)
    config: object = B0DirectConfig()
    if mutation == "manifest_type": manifest = object()
    elif mutation == "command_type": command = [[0.0, 0.0, 0.0]]
    elif mutation == "root_type": root = [[0.0] * 13]
    elif mutation == "config_type": config = object()
    elif mutation == "command_subclass": command = command.as_subclass(_TensorSubclass)
    elif mutation == "root_subclass": root = root.as_subclass(_TensorSubclass)
    elif mutation == "command_rank": command = torch.zeros(3)
    elif mutation == "command_width": command = torch.zeros((2, 4))
    elif mutation == "empty": command, root = torch.zeros((0, 3)), torch.zeros((0, 13))
    elif mutation == "root_width": root = torch.zeros((2, 12))
    elif mutation == "batch": root = _root(torch.float32, batch=1)
    elif mutation == "dtype": root = _root(torch.float64, batch=2)
    elif mutation == "grad_command": command = torch.zeros((2, 3), requires_grad=True)
    elif mutation == "grad_root": root = _root(torch.float32, batch=2).requires_grad_()
    elif mutation == "nan_command": command[0, 0] = float("nan")
    elif mutation == "inf_root": root[0, 0] = float("inf")
    elif mutation == "nonunit": root[:, 3] = 2.0
    elif mutation == "alias":
        backing = torch.zeros((2, 13)); backing[:, 3] = 1.0
        command, root = backing[:, :3], backing
    calls = []
    monkeypatch.setattr(bridge_module, "execute_ready_b0", lambda *args: calls.append("b0"))
    monkeypatch.setattr(bridge_module, "taslab_velocity_control", lambda *args: calls.append("lee"))
    with pytest.raises((TypeError, ValueError)):
        execute_ready_b0_controller(manifest, command, root, config)  # type: ignore[arg-type]
    assert calls == []


@pytest.mark.parametrize("case", ("not_ready", "non_b0", "malformed"))
def test_b0_rejection_calls_b0_once_and_never_lee(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    accepted = bridge_module.execute_ready_b0
    def b0_spy(*args):
        calls.append("b0")
        return accepted(*args)
    monkeypatch.setattr(bridge_module, "execute_ready_b0", b0_spy)
    monkeypatch.setattr(bridge_module, "taslab_velocity_control", lambda *args: calls.append("lee"))
    manifests = {
        "not_ready": _manifest(ExecutionStatus.NOT_READY),
        "non_b0": _non_b0_manifest(),
        "malformed": _malformed_manifest(),
    }
    with pytest.raises(ValueError):
        execute_ready_b0_controller(
            manifests[case], torch.zeros((1, 3)),
            _root(torch.float32, batch=1), B0DirectConfig(),
        )
    assert calls == ["b0"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device mismatch requires CUDA")
def test_device_mismatch_rejects_before_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(bridge_module, "execute_ready_b0", lambda *args: calls.append("b0"))
    monkeypatch.setattr(bridge_module, "taslab_velocity_control", lambda *args: calls.append("lee"))
    with pytest.raises(ValueError):
        execute_ready_b0_controller(
            _manifest(), torch.zeros((1, 3)), _root(torch.float32, batch=1).cuda(),
            B0DirectConfig(),
        )
    assert calls == []


def test_inputs_immutable_outputs_owned_nonaliased_and_deterministic() -> None:
    manifest = _manifest()
    command = torch.tensor(((0.4, -0.1, 0.2),), dtype=torch.float64)
    root = _root(torch.float64, yaw_degrees=35.0, batch=1)
    command_before, root_before = command.clone(), root.clone()
    first = execute_ready_b0_controller(manifest, command, root, B0DirectConfig(2.0))
    second = execute_ready_b0_controller(manifest, command, root, B0DirectConfig(2.0))
    assert torch.equal(command, command_before) and torch.equal(root, root_before)
    assert first.b0_execution.executed_manifest.canonical_json == second.b0_execution.executed_manifest.canonical_json
    for name in vars(first.lee_result):
        left, right = getattr(first.lee_result, name), getattr(second.lee_result, name)
        assert torch.equal(left, right) and torch.isfinite(left).all() and not left.requires_grad
    pointers = {command.untyped_storage().data_ptr(), root.untyped_storage().data_ptr()}
    pointers.update(value.untyped_storage().data_ptr() for value in vars(first.lee_result).values())
    pointers.update((first.b0_execution.result.issued_body_velocity_m_s.untyped_storage().data_ptr(),
                     first.b0_execution.result.issued_world_velocity_m_s.untyped_storage().data_ptr()))
    assert len(pointers) == 9
    with pytest.raises(FrozenInstanceError):
        first.lee_result = second.lee_result


def test_signature_exports_imports_and_command_only_actor_boundary() -> None:
    signature = inspect.signature(execute_ready_b0_controller)
    assert tuple(signature.parameters) == (
        "manifest", "commanded_body_velocity_m_s", "controller_private_root_state_w", "config",
    )
    assert all(item.default is inspect.Parameter.empty for item in signature.parameters.values())
    assert evaluation.execute_ready_b0_controller is execute_ready_b0_controller
    assert evaluation.B0ControllerExecutionEnvelope is B0ControllerExecutionEnvelope
    path = ROOT / "instinctlab/evaluation/baseline_controller_execution.py"
    tree = ast.parse(path.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom): imports.add(("." * node.level) + (node.module or ""))
    assert imports == {"__future__", "dataclasses", "torch", "instinctlab.control.baseline_b0",
                       "instinctlab.control.lee_velocity_controller", ".baseline_execution", ".baselines"}
    assert "instinctlab.control" not in imports
    row = resolve_baseline("B0")
    assert row.actor_input_fields == ("commanded_body_velocity_m_s",)
    assert "root_state_w" not in row.actor_input_fields
    assert "root_state" not in _manifest().canonical_json


def test_bounded_smoke_has_exact_value_oracles_and_nonactuating_two_sided_gates() -> None:
    path = ROOT / "tests/run_m7_b0_controller_smoke.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    literal_keys = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"clip_vector_norm", "rotate_body_to_world", "taslab_velocity_control"} <= calls
    assert {
        "body_clip_value_exact", "body_clip_norm_bound",
        "world_rotation_value_exact", "lee_direct_chain_exact",
        "physical_thrust_nonnegative", "physical_thrust_maximum",
        "body_clip_max_abs_error", "body_clip_mismatch_count",
        "world_rotation_max_abs_error", "world_rotation_mismatch_count",
        "lee_direct_chain_max_abs_error", "lee_direct_chain_mismatch_count",
    } <= literal_keys
    assert 'oracle_errors["body_clip_mismatch_count"] == 0' in source
    assert 'oracle_errors["body_clip_max_abs_error"] == 0.0' in source
    assert 'oracle_errors["world_rotation_mismatch_count"] == 0' in source
    assert 'oracle_errors["world_rotation_max_abs_error"] == 0.0' in source
    assert 'oracle_errors["lee_direct_chain_mismatch_count"] == 0' in source
    assert 'oracle_errors["lee_direct_chain_max_abs_error"] == 0.0' in source
    assert 'trace["records"]["rotor_thrusts_n"] >= 0.0' in source
    assert 'trace["records"]["rotor_thrusts_n"] <= max_thrust' in source
    adapter_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply"
    ]
    assert len(adapter_calls) == 1
    assert ast.unparse(adapter_calls[0].args[1]) == "envelope.lee_result.normalized_rotor_commands"
    assert "expected_lee" not in ast.unparse(adapter_calls[0])
