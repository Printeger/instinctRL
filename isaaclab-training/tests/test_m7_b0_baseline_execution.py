"""Focused fail-closed B0 READY-manifest execution boundary tests."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import FrozenInstanceError
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest
import torch

import instinctlab.evaluation as evaluation
import instinctlab.evaluation.baseline_execution as execution_module
from instinctlab.control.baseline_b0 import (
    B0DirectConfig,
    B0DirectResult,
    execute_b0_direct_velocity,
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
from instinctlab.evaluation.baseline_execution import B0ExecutionEnvelope, execute_ready_b0


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_COMMIT = "fb9b9c4af7b0004c0b0647b80acef2e9e85e361b"


def _bindings() -> tuple[CapabilityBinding, ...]:
    return tuple(
        CapabilityBinding(item.name, item.accepted_provider, item.accepted_evidence_sha256)
        for item in resolve_baseline("B0").capability_requirements
    )


def _manifest(
    baseline_id: str = "B0", status: ExecutionStatus = ExecutionStatus.READY
) -> BaselineRunManifest:
    row = resolve_baseline(baseline_id)
    ready = status in (ExecutionStatus.READY, ExecutionStatus.EXECUTED)
    return build_run_manifest(
        product_commit=PRODUCT_COMMIT,
        baseline_id=baseline_id,
        config_switch=row.config_switch,
        capability_bindings=_bindings() if ready else (),
        scenario_id="S1.corner",
        seed=17,
        actor_audit_status=ActorAuditStatus.PASS if ready else ActorAuditStatus.NOT_RUN,
        execution_status=status,
        skip_reason="capability_unavailable" if status is ExecutionStatus.SKIPPED else None,
    )


def _resign(mapping: OrderedDict[str, object]) -> str:
    payload = OrderedDict((name, mapping[name]) for name in MANIFEST_PAYLOAD_FIELD_ORDER)
    mapping["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return json.dumps(mapping, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _forged(canonical_json: str) -> BaselineRunManifest:
    manifest = object.__new__(BaselineRunManifest)
    object.__setattr__(manifest, "canonical_json", canonical_json)
    return manifest


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_ready_to_executed_hand_oracle_parity_and_batch(dtype: torch.dtype) -> None:
    command = torch.tensor(
        ((0.0, 0.0, 0.0), (0.5, -0.25, 0.75), (3.0, 4.0, 0.0)), dtype=dtype
    )
    quaternion = torch.tensor(
        ((1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5),
         (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))), dtype=dtype
    )
    ready = _manifest()
    envelope = execute_ready_b0(ready, command, quaternion, B0DirectConfig(2.0))
    direct = execute_b0_direct_velocity(command, quaternion, B0DirectConfig(2.0))
    assert type(envelope) is B0ExecutionEnvelope
    assert torch.equal(envelope.result.issued_body_velocity_m_s, direct.issued_body_velocity_m_s)
    assert torch.equal(envelope.result.issued_world_velocity_m_s, direct.issued_world_velocity_m_s)
    assert envelope.ready_manifest is not ready
    assert envelope.ready_manifest.canonical_json == ready.canonical_json
    assert envelope.executed_manifest.to_mapping()["execution_status"] == "EXECUTED"


def test_successor_changes_only_status_and_hash() -> None:
    ready = _manifest()
    envelope = execute_ready_b0(
        ready,
        torch.tensor(((1.0, 0.0, 0.0),), dtype=torch.float64),
        torch.tensor(((1.0, 0.0, 0.0, 0.0),), dtype=torch.float64),
        B0DirectConfig(),
    )
    before, after = ready.to_mapping(), envelope.executed_manifest.to_mapping()
    changed = {name for name in before if before[name] != after[name]}
    assert changed == {"execution_status", "manifest_sha256"}
    assert before["manifest_sha256"] != after["manifest_sha256"]
    assert len(before["capability_bindings"]) == 8


def test_adapter_is_called_exactly_once_after_manifest_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    accepted = execution_module.execute_b0_direct_velocity

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return accepted(*args, **kwargs)

    monkeypatch.setattr(execution_module, "execute_b0_direct_velocity", spy)
    execute_ready_b0(
        _manifest(), torch.zeros((1, 3)), torch.tensor(((1.0, 0.0, 0.0, 0.0),)),
        B0DirectConfig(),
    )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "manifest",
    (
        None,
        {},
        *(_manifest(f"B{index}", ExecutionStatus.NOT_READY) for index in range(1, 8)),
        _manifest("B8", ExecutionStatus.SKIPPED),
        _manifest("B0", ExecutionStatus.NOT_READY),
        _manifest("B0", ExecutionStatus.EXECUTED),
        _forged(_manifest().canonical_json + "\n"),
        _forged(_manifest().canonical_json.replace('"manifest_sha256":"', '"manifest_sha256":"0')),
    ),
)
def test_unsupported_or_malformed_manifest_rejects_before_adapter(
    manifest: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(execution_module, "execute_b0_direct_velocity", lambda *args: calls.append(args))
    with pytest.raises((TypeError, ValueError)):
        execute_ready_b0(  # type: ignore[arg-type]
            manifest, torch.zeros((1, 3)), torch.tensor(((1.0, 0.0, 0.0, 0.0),)),
            B0DirectConfig(),
        )
    assert calls == []


@pytest.mark.parametrize(
    "mutation",
    (
        "alias", "switch", "actor_schema", "toggle", "audit", "skip",
        "stale_three", "stale_four", "stale_five", "stale_six", "stale_seven", "missing", "extra", "reordered", "duplicate_name",
        "duplicate_provider", "wrong_provider", "wrong_hash",
    ),
)
def test_resigned_semantic_mutations_reject_before_adapter(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = OrderedDict(_manifest().to_mapping())
    bindings = mapping["capability_bindings"]
    if mutation == "alias": mapping["baseline_id"] = "direct_velocity"
    elif mutation == "switch": mapping["config_switch"] = "no_history"
    elif mutation == "actor_schema": mapping["actor_input_fields"] = ["root_state"]
    elif mutation == "toggle": mapping["component_toggles"]["ics"] = True
    elif mutation == "audit": mapping["actor_audit_status"] = "NOT_RUN"
    elif mutation == "skip": mapping["skip_reason"] = "fallback"
    elif mutation == "stale_three": mapping["capability_bindings"] = bindings[:3]
    elif mutation == "stale_four": mapping["capability_bindings"] = bindings[:4]
    elif mutation == "stale_five": mapping["capability_bindings"] = bindings[:5]
    elif mutation == "stale_six": mapping["capability_bindings"] = bindings[:6]
    elif mutation == "stale_seven": mapping["capability_bindings"] = bindings[:7]
    elif mutation == "missing": mapping["capability_bindings"] = bindings[:-1]
    elif mutation == "extra": bindings.append(OrderedDict((
        ("name", "m7.extra"), ("provider", "instinctlab.extra"), ("evidence_sha256", "1" * 64))))
    elif mutation == "reordered": mapping["capability_bindings"] = [bindings[1], bindings[0], *bindings[2:]]
    elif mutation == "duplicate_name": bindings[1]["name"] = bindings[0]["name"]
    elif mutation == "duplicate_provider": bindings[1]["provider"] = bindings[0]["provider"]
    elif mutation == "wrong_provider": bindings[5]["provider"] = "instinctlab.evaluation.alias"
    elif mutation == "wrong_hash": bindings[6]["evidence_sha256"] = "0" * 64
    calls = []
    monkeypatch.setattr(execution_module, "execute_b0_direct_velocity", lambda *args: calls.append(args))
    with pytest.raises((KeyError, TypeError, ValueError)):
        execute_ready_b0(
            _forged(_resign(mapping)), torch.zeros((1, 3)),
            torch.tensor(((1.0, 0.0, 0.0, 0.0),)), B0DirectConfig(),
        )
    assert calls == []


def test_nonexact_config_and_tensor_types_reject_without_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(execution_module, "execute_b0_direct_velocity", lambda *args: calls.append(args))
    ready = _manifest()
    with pytest.raises(TypeError):
        execute_ready_b0(ready, [[0.0, 0.0, 0.0]], torch.zeros((1, 4)), B0DirectConfig())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        execute_ready_b0(ready, torch.zeros((1, 3)), [[1.0, 0.0, 0.0, 0.0]], B0DirectConfig())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        execute_ready_b0(ready, torch.zeros((1, 3)), torch.zeros((1, 4)), object())  # type: ignore[arg-type]
    assert calls == []


def test_inputs_unchanged_outputs_owned_and_repeated_calls_deterministic() -> None:
    ready = _manifest()
    command = torch.tensor(((3.0, 4.0, 0.0),), dtype=torch.float64)
    quaternion = torch.tensor(((1.0, 0.0, 0.0, 0.0),), dtype=torch.float64)
    before_command, before_quaternion = command.clone(), quaternion.clone()
    first = execute_ready_b0(ready, command, quaternion, B0DirectConfig(2.0))
    second = execute_ready_b0(ready, command, quaternion, B0DirectConfig(2.0))
    assert ready.canonical_json == first.ready_manifest.canonical_json
    assert torch.equal(command, before_command) and torch.equal(quaternion, before_quaternion)
    assert torch.equal(first.result.issued_body_velocity_m_s, second.result.issued_body_velocity_m_s)
    assert first.executed_manifest.canonical_json == second.executed_manifest.canonical_json
    pointers = {command.untyped_storage().data_ptr(), quaternion.untyped_storage().data_ptr(),
        first.result.issued_body_velocity_m_s.untyped_storage().data_ptr(),
        first.result.issued_world_velocity_m_s.untyped_storage().data_ptr()}
    assert len(pointers) == 4
    with pytest.raises(FrozenInstanceError):
        first.ready_manifest = ready


def test_signature_exports_import_boundary_and_no_reverse_import() -> None:
    signature = inspect.signature(execute_ready_b0)
    assert tuple(signature.parameters) == (
        "manifest", "commanded_body_velocity_m_s",
        "controller_private_quaternion_wxyz", "config",
    )
    assert all(item.default is inspect.Parameter.empty for item in signature.parameters.values())
    assert evaluation.execute_ready_b0 is execute_ready_b0
    assert evaluation.B0ExecutionEnvelope is B0ExecutionEnvelope
    path = ROOT / "instinctlab/evaluation/baseline_execution.py"
    tree = ast.parse(path.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom): imports.add(("." * node.level) + (node.module or ""))
    assert imports == {
        "__future__", "dataclasses", "torch",
        "instinctlab.control.baseline_b0", ".baselines",
    }
    assert "instinctlab.control" not in imports
    for root in (ROOT / "instinctlab/envs", ROOT / "instinctlab/observations", ROOT / "instinctlab/training"):
        assert all("baseline_execution" not in source.read_text() for source in root.rglob("*.py"))


def test_adapter_source_remains_exact_and_tensor_payload_not_serialized() -> None:
    assert hashlib.sha256((ROOT / "instinctlab/control/baseline_b0.py").read_bytes()).hexdigest() == (
        "5fa2b8d746eb231a1bb12f331fe1a76e8f988005e9a315e1dbf622e0bf1ed3d1"
    )
    envelope = execute_ready_b0(
        _manifest(), torch.zeros((1, 3)),
        torch.tensor(((1.0, 0.0, 0.0, 0.0),)), B0DirectConfig(),
    )
    assert "issued_body" not in envelope.executed_manifest.canonical_json
    assert type(envelope.result) is B0DirectResult
