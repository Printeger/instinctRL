"""Fail-closed canonical-manifest boundary for pure B0 execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from instinctlab.control import (
    B0DirectConfig,
    B0DirectResult,
    execute_b0_direct_velocity,
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
    build_run_manifest,
    resolve_baseline,
    validate_capability_bindings,
)


@dataclass(frozen=True)
class B0ExecutionEnvelope:
    """Independent canonical manifests and the owned direct-adapter result."""

    ready_manifest: BaselineRunManifest
    executed_manifest: BaselineRunManifest
    result: B0DirectResult


def _require_ready_b0(manifest: BaselineRunManifest) -> tuple[
    BaselineRunManifest, dict[str, object], tuple[CapabilityBinding, ...]
]:
    if type(manifest) is not BaselineRunManifest:
        raise TypeError("manifest must be an exact BaselineRunManifest")
    canonical = BaselineRunManifest.from_json(manifest.canonical_json)
    mapping = canonical.to_mapping()
    if mapping["baseline_id"] != BaselineId.B0.value:
        raise ValueError("manifest must identify exact B0")
    baseline = resolve_baseline(BaselineId.B0)
    if mapping["config_switch"] != baseline.config_switch:
        raise ValueError("B0 manifest must use the exact direct_velocity switch")
    if not baseline.deployable or baseline.privilege_class is not PrivilegeClass.ACTOR_CLEAN:
        raise ValueError("B0 registry row must remain deployable and actor-clean")
    if baseline.actor_input_fields != ("commanded_body_velocity_m_s",):
        raise ValueError("B0 actor schema must remain command-only")
    if baseline.component_toggles != ComponentToggles(False, False, False, False):
        raise ValueError("B0 component toggles must all remain false")
    raw_bindings = mapping["capability_bindings"]
    if type(raw_bindings) is not list:
        raise TypeError("B0 capability bindings must be a canonical JSON array")
    bindings = tuple(
        CapabilityBinding(item["name"], item["provider"], item["evidence_sha256"])
        for item in raw_bindings
    )
    validate_capability_bindings(baseline, bindings, ExecutionStatus.READY)
    if mapping["actor_audit_status"] != ActorAuditStatus.PASS.value:
        raise ValueError("B0 READY execution requires actor audit PASS")
    if mapping["execution_status"] != ExecutionStatus.READY.value:
        raise ValueError("B0 execution boundary accepts only READY manifests")
    if mapping["skip_reason"] is not None:
        raise ValueError("B0 READY execution cannot contain a skip reason")
    return canonical, mapping, bindings


def execute_ready_b0(
    manifest: BaselineRunManifest,
    commanded_body_velocity_m_s: torch.Tensor,
    controller_private_quaternion_wxyz: torch.Tensor,
    config: B0DirectConfig,
) -> B0ExecutionEnvelope:
    """Validate exact B0 READY provenance, execute once, and bind EXECUTED."""

    ready_manifest, ready_mapping, bindings = _require_ready_b0(manifest)
    if type(commanded_body_velocity_m_s) is not torch.Tensor:
        raise TypeError("commanded_body_velocity_m_s must be an exact torch.Tensor")
    if type(controller_private_quaternion_wxyz) is not torch.Tensor:
        raise TypeError(
            "controller_private_quaternion_wxyz must be an exact torch.Tensor"
        )
    if type(config) is not B0DirectConfig:
        raise TypeError("config must be an exact B0DirectConfig")
    result = execute_b0_direct_velocity(
        commanded_body_velocity_m_s,
        controller_private_quaternion_wxyz,
        config,
    )
    executed_manifest = build_run_manifest(
        product_commit=ready_mapping["product_commit"],
        baseline_id=BaselineId.B0,
        config_switch=ready_mapping["config_switch"],
        capability_bindings=bindings,
        scenario_id=ready_mapping["scenario_id"],
        seed=ready_mapping["seed"],
        actor_audit_status=ActorAuditStatus.PASS,
        execution_status=ExecutionStatus.EXECUTED,
        skip_reason=None,
    )
    executed_mapping = executed_manifest.to_mapping()
    for name in MANIFEST_PAYLOAD_FIELD_ORDER:
        if name != "execution_status" and executed_mapping[name] != ready_mapping[name]:
            raise RuntimeError(f"EXECUTED successor changed immutable payload field {name}")
    if executed_mapping["execution_status"] != ExecutionStatus.EXECUTED.value:
        raise RuntimeError("EXECUTED successor has the wrong status")
    if executed_manifest.manifest_sha256 == ready_manifest.manifest_sha256:
        raise RuntimeError("READY and EXECUTED manifests must have different hashes")
    return B0ExecutionEnvelope(ready_manifest, executed_manifest, result)


__all__ = ["B0ExecutionEnvelope", "execute_ready_b0"]
