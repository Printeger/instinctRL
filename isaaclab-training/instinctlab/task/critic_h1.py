"""The frozen 21-field ``critic-h1-v2`` privileged-state contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .critic import QUATERNION_NORM_TOLERANCE

CRITIC_H1_VERSION = "critic-h1-v2"
CRITIC_H1_FIELD_ORDER = (
    "local_position_x_div_10", "local_position_y_div_10", "local_position_z_div_10",
    "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z",
    "linear_velocity_w_x_m_s", "linear_velocity_w_y_m_s", "linear_velocity_w_z_m_s",
    "angular_velocity_w_x_rad_s", "angular_velocity_w_y_rad_s", "angular_velocity_w_z_rad_s",
    "actual_linear_velocity_b_x_m_s", "actual_linear_velocity_b_y_m_s", "actual_linear_velocity_b_z_m_s",
    "exact_clearance_m", "collision_label",
    "command_b_x_m_s", "command_b_y_m_s", "command_b_z_m_s",
)
CRITIC_H1_SCALES = (0.1, 0.1, 0.1) + (1.0,) * 18
CRITIC_H1_DIMENSION = len(CRITIC_H1_FIELD_ORDER)


@dataclass(frozen=True)
class CriticH1State:
    privileged: torch.Tensor
    field_order: tuple[str, ...] = CRITIC_H1_FIELD_ORDER
    scales: tuple[float, ...] = CRITIC_H1_SCALES


def _finite_float(name: str, value: torch.Tensor, shape: tuple[int, ...], reference: torch.Tensor | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(value).all().item()):
        raise TypeError(f"{name} must be finite float32 or float64")
    if reference is not None and (value.dtype != reference.dtype or value.device != reference.device):
        raise TypeError(f"{name} must share root-state dtype and device")
    return value


def build_critic_h1_state(
    *, root_state_w: torch.Tensor, env_origin_w: torch.Tensor, actual_velocity_b: torch.Tensor,
    exact_clearance_m: torch.Tensor, collision: torch.Tensor, command_b: torch.Tensor,
) -> CriticH1State:
    """Build local-position/command-conditioned critic state without an actor container."""
    if not isinstance(root_state_w, torch.Tensor) or root_state_w.ndim != 2 or root_state_w.shape[0] <= 0 or root_state_w.shape[1] != 13:
        raise ValueError("root_state_w must have shape (N, 13)")
    _finite_float("root_state_w", root_state_w, tuple(root_state_w.shape))
    batch = root_state_w.shape[0]
    origin = _finite_float("env_origin_w", env_origin_w, (batch, 3), root_state_w)
    actual = _finite_float("actual_velocity_b", actual_velocity_b, (batch, 3), root_state_w)
    clearance = _finite_float("exact_clearance_m", exact_clearance_m, (batch, 1), root_state_w)
    command = _finite_float("command_b", command_b, (batch, 3), root_state_w)
    if bool((clearance < 0.0).any().item()):
        raise ValueError("exact_clearance_m must be nonnegative")
    if not isinstance(collision, torch.Tensor) or collision.shape != (batch, 1) or collision.dtype != torch.bool:
        raise TypeError("collision must be boolean with shape (N, 1)")
    if collision.device != root_state_w.device:
        raise ValueError("collision must share root-state device")
    quaternion = root_state_w[:, 3:7]
    if bool((torch.abs(torch.linalg.vector_norm(quaternion, dim=-1) - 1.0) > QUATERNION_NORM_TOLERANCE).any().item()):
        raise ValueError("root quaternion must be unit length within tolerance")
    privileged = torch.cat(
        ((root_state_w[:, :3] - origin) / 10.0, root_state_w[:, 3:13], actual,
         clearance, collision.to(root_state_w.dtype), command), dim=-1,
    )
    if privileged.shape != (batch, CRITIC_H1_DIMENSION) or not bool(torch.isfinite(privileged).all().item()):
        raise RuntimeError("internal critic-h1 state contract violation")
    return CriticH1State(privileged)


__all__ = [
    "CRITIC_H1_DIMENSION", "CRITIC_H1_FIELD_ORDER", "CRITIC_H1_SCALES", "CRITIC_H1_VERSION",
    "CriticH1State", "build_critic_h1_state",
]
