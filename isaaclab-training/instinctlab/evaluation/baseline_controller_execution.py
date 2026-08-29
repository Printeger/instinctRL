"""Exact manifest-gated B0 handoff to the conventional TASLAB controller."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from instinctlab.control.baseline_b0 import B0DirectConfig
from instinctlab.control.lee_velocity_controller import (
    LeeControllerResult,
    QUATERNION_NORM_TOLERANCE,
    taslab_velocity_control,
)

from .baseline_execution import B0ExecutionEnvelope, execute_ready_b0
from .baselines import BaselineRunManifest


@dataclass(frozen=True)
class B0ControllerExecutionEnvelope:
    """Owned B0 execution and conventional Lee-controller result."""

    b0_execution: B0ExecutionEnvelope
    lee_result: LeeControllerResult


def _owned_controller_inputs(
    manifest: BaselineRunManifest,
    commanded_body_velocity_m_s: torch.Tensor,
    controller_private_root_state_w: torch.Tensor,
    config: B0DirectConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if type(manifest) is not BaselineRunManifest:
        raise TypeError("manifest must be an exact BaselineRunManifest")
    if type(commanded_body_velocity_m_s) is not torch.Tensor:
        raise TypeError("commanded_body_velocity_m_s must be an exact torch.Tensor")
    if type(controller_private_root_state_w) is not torch.Tensor:
        raise TypeError("controller_private_root_state_w must be an exact torch.Tensor")
    if type(config) is not B0DirectConfig:
        raise TypeError("config must be an exact B0DirectConfig")
    if commanded_body_velocity_m_s.ndim != 2 or commanded_body_velocity_m_s.shape[1] != 3:
        raise ValueError("commanded_body_velocity_m_s must have shape (N, 3)")
    if commanded_body_velocity_m_s.shape[0] <= 0:
        raise ValueError("controller handoff batch must be nonempty")
    if controller_private_root_state_w.shape != (commanded_body_velocity_m_s.shape[0], 13):
        raise ValueError("controller_private_root_state_w must have matching shape (N, 13)")
    if commanded_body_velocity_m_s.dtype not in (torch.float32, torch.float64):
        raise TypeError("controller handoff tensors must use float32 or float64")
    if controller_private_root_state_w.dtype != commanded_body_velocity_m_s.dtype:
        raise TypeError("controller handoff tensors must share one dtype")
    if controller_private_root_state_w.device != commanded_body_velocity_m_s.device:
        raise ValueError("controller handoff tensors must share one device")
    if (
        commanded_body_velocity_m_s.untyped_storage().data_ptr()
        == controller_private_root_state_w.untyped_storage().data_ptr()
    ):
        raise ValueError("command and controller-private root state must not alias")
    if commanded_body_velocity_m_s.requires_grad or controller_private_root_state_w.requires_grad:
        raise ValueError("controller handoff tensors must not require gradients")
    if not bool(torch.isfinite(commanded_body_velocity_m_s).all().item()):
        raise ValueError("commanded_body_velocity_m_s must be finite")
    if not bool(torch.isfinite(controller_private_root_state_w).all().item()):
        raise ValueError("controller_private_root_state_w must be finite")
    quaternion = controller_private_root_state_w[:, 3:7]
    quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1)
    if bool((torch.abs(quaternion_norm - 1.0) > QUATERNION_NORM_TOLERANCE).any().item()):
        raise ValueError("controller-private root quaternion must be unit length")
    command_snapshot = commanded_body_velocity_m_s.detach().clone()
    root_snapshot = controller_private_root_state_w.detach().clone()
    quaternion_snapshot = root_snapshot[:, 3:7].clone()
    return command_snapshot, root_snapshot, quaternion_snapshot


def execute_ready_b0_controller(
    manifest: BaselineRunManifest,
    commanded_body_velocity_m_s: torch.Tensor,
    controller_private_root_state_w: torch.Tensor,
    config: B0DirectConfig,
) -> B0ControllerExecutionEnvelope:
    """Execute exact B0 once, then hand its exact world target to Lee once."""

    command, root, quaternion = _owned_controller_inputs(
        manifest,
        commanded_body_velocity_m_s,
        controller_private_root_state_w,
        config,
    )
    b0_execution = execute_ready_b0(manifest, command, quaternion, config)
    lee_result = taslab_velocity_control(
        root,
        b0_execution.result.issued_world_velocity_m_s,
    )
    return B0ControllerExecutionEnvelope(b0_execution, lee_result)


__all__ = ["B0ControllerExecutionEnvelope", "execute_ready_b0_controller"]
