"""Pure command-only B0 direct-velocity execution primitive."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .command_adapter import QUATERNION_NORM_TOLERANCE, rotate_body_to_world
from .governor import clip_vector_norm


@dataclass(frozen=True)
class B0DirectConfig:
    """Immutable positive body-command norm limit in metres per second."""

    velocity_limit_m_s: float = 2.0

    def __post_init__(self) -> None:
        value = self.velocity_limit_m_s
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("velocity_limit_m_s must be a real scalar")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("velocity_limit_m_s must be positive and finite")
        object.__setattr__(self, "velocity_limit_m_s", float(value))


@dataclass(frozen=True)
class B0DirectResult:
    """Owned issued commands on the body and world frame boundaries."""

    issued_body_velocity_m_s: torch.Tensor
    issued_world_velocity_m_s: torch.Tensor


def _validate_inputs(
    commanded_body_velocity_m_s: torch.Tensor,
    controller_private_quaternion_wxyz: torch.Tensor,
    config: B0DirectConfig,
) -> None:
    if type(config) is not B0DirectConfig:
        raise TypeError("config must be B0DirectConfig")
    if type(commanded_body_velocity_m_s) is not torch.Tensor:
        raise TypeError("commanded_body_velocity_m_s must be an exact torch.Tensor")
    if type(controller_private_quaternion_wxyz) is not torch.Tensor:
        raise TypeError(
            "controller_private_quaternion_wxyz must be an exact torch.Tensor"
        )
    if (
        commanded_body_velocity_m_s.ndim != 2
        or commanded_body_velocity_m_s.shape[0] <= 0
        or commanded_body_velocity_m_s.shape[1] != 3
    ):
        raise ValueError("commanded_body_velocity_m_s must have shape (N, 3), N > 0")
    if controller_private_quaternion_wxyz.shape != (
        commanded_body_velocity_m_s.shape[0],
        4,
    ):
        raise ValueError(
            "controller_private_quaternion_wxyz must have matching shape (N, 4)"
        )
    if commanded_body_velocity_m_s.dtype not in (torch.float32, torch.float64):
        raise TypeError("commanded_body_velocity_m_s must use float32 or float64")
    if controller_private_quaternion_wxyz.dtype != commanded_body_velocity_m_s.dtype:
        raise TypeError("B0 command and controller-private quaternion must share dtype")
    if controller_private_quaternion_wxyz.device != commanded_body_velocity_m_s.device:
        raise ValueError("B0 command and controller-private quaternion must share device")
    if commanded_body_velocity_m_s.requires_grad or (
        commanded_body_velocity_m_s.grad_fn is not None
    ):
        raise ValueError("commanded_body_velocity_m_s must be detached and gradient-free")
    if controller_private_quaternion_wxyz.requires_grad or (
        controller_private_quaternion_wxyz.grad_fn is not None
    ):
        raise ValueError(
            "controller_private_quaternion_wxyz must be detached and gradient-free"
        )
    if not bool(torch.isfinite(commanded_body_velocity_m_s).all().item()):
        raise ValueError("commanded_body_velocity_m_s must be finite")
    if not bool(torch.isfinite(controller_private_quaternion_wxyz).all().item()):
        raise ValueError("controller_private_quaternion_wxyz must be finite")
    norm_error = torch.abs(
        torch.linalg.vector_norm(controller_private_quaternion_wxyz, dim=-1) - 1.0
    )
    if bool((norm_error > QUATERNION_NORM_TOLERANCE).any().item()):
        raise ValueError(
            "controller_private_quaternion_wxyz must be unit length within tolerance"
        )


def execute_b0_direct_velocity(
    commanded_body_velocity_m_s: torch.Tensor,
    controller_private_quaternion_wxyz: torch.Tensor,
    config: B0DirectConfig,
) -> B0DirectResult:
    """Issue an exact norm-limited command across the named frame boundary."""

    _validate_inputs(
        commanded_body_velocity_m_s,
        controller_private_quaternion_wxyz,
        config,
    )
    issued_body_velocity_m_s = clip_vector_norm(
        commanded_body_velocity_m_s, config.velocity_limit_m_s
    ).detach().clone()
    issued_world_velocity_m_s = rotate_body_to_world(
        issued_body_velocity_m_s, controller_private_quaternion_wxyz
    ).detach().clone()
    if not bool(torch.isfinite(issued_body_velocity_m_s).all().item()) or not bool(
        torch.isfinite(issued_world_velocity_m_s).all().item()
    ):
        raise ValueError("B0 direct adapter produced a nonfinite issued command")
    return B0DirectResult(
        issued_body_velocity_m_s=issued_body_velocity_m_s,
        issued_world_velocity_m_s=issued_world_velocity_m_s,
    )


__all__ = ["B0DirectConfig", "B0DirectResult", "execute_b0_direct_velocity"]
