"""Fixed-order critic-only privileged tensor builder."""

from __future__ import annotations

from dataclasses import dataclass

import torch

CRITIC_FIELD_ORDER = (
    "position_w_x_m", "position_w_y_m", "position_w_z_m",
    "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z",
    "linear_velocity_w_x_m_s", "linear_velocity_w_y_m_s", "linear_velocity_w_z_m_s",
    "angular_velocity_w_x_rad_s", "angular_velocity_w_y_rad_s", "angular_velocity_w_z_rad_s",
    "actual_linear_velocity_b_x_m_s", "actual_linear_velocity_b_y_m_s", "actual_linear_velocity_b_z_m_s",
    "exact_clearance_m", "collision_label",
)
CRITIC_DIMENSION = len(CRITIC_FIELD_ORDER)
QUATERNION_NORM_TOLERANCE = 1.0e-5


@dataclass(frozen=True)
class CriticState:
    privileged: torch.Tensor
    field_order: tuple[str, ...] = CRITIC_FIELD_ORDER


def build_critic_state(
    *,
    root_state_w: torch.Tensor,
    actual_velocity_b: torch.Tensor,
    exact_clearance_m: torch.Tensor,
    collision: torch.Tensor,
) -> CriticState:
    """Build `[N,18]` critic-only state without accepting actor/control containers."""
    if not isinstance(root_state_w, torch.Tensor) or root_state_w.ndim != 2 or root_state_w.shape[0] <= 0 or root_state_w.shape[1] != 13:
        raise ValueError("root_state_w must have shape (N, 13)")
    if root_state_w.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(root_state_w).all().item()):
        raise TypeError("root_state_w must be finite float32 or float64")
    batch = root_state_w.shape[0]
    if not isinstance(actual_velocity_b, torch.Tensor) or actual_velocity_b.shape != (batch, 3):
        raise ValueError("actual_velocity_b must have shape (N, 3)")
    if actual_velocity_b.dtype != root_state_w.dtype or actual_velocity_b.device != root_state_w.device:
        raise TypeError("actual_velocity_b must share root-state dtype and device")
    if not bool(torch.isfinite(actual_velocity_b).all().item()):
        raise ValueError("actual_velocity_b must be finite")
    if not isinstance(exact_clearance_m, torch.Tensor) or exact_clearance_m.shape != (batch, 1):
        raise ValueError("exact_clearance_m must have shape (N, 1)")
    if exact_clearance_m.dtype != root_state_w.dtype or exact_clearance_m.device != root_state_w.device:
        raise TypeError("exact_clearance_m must share root-state dtype and device")
    if not bool(torch.isfinite(exact_clearance_m).all().item()) or bool((exact_clearance_m < 0.0).any().item()):
        raise ValueError("exact_clearance_m must be finite and nonnegative")
    if not isinstance(collision, torch.Tensor) or collision.shape != (batch, 1) or collision.dtype != torch.bool:
        raise TypeError("collision must be boolean with shape (N, 1)")
    if collision.device != root_state_w.device:
        raise ValueError("collision must share root-state device")
    quaternion = root_state_w[:, 3:7]
    norm_error = torch.abs(torch.linalg.vector_norm(quaternion, dim=-1) - 1.0)
    if bool((norm_error > QUATERNION_NORM_TOLERANCE).any().item()):
        raise ValueError("root quaternion must be unit length within tolerance")
    privileged = torch.cat(
        (root_state_w, actual_velocity_b, exact_clearance_m, collision.to(root_state_w.dtype)), dim=-1
    )
    if privileged.shape != (batch, CRITIC_DIMENSION) or not bool(torch.isfinite(privileged).all().item()):
        raise RuntimeError("internal critic state contract violation")
    return CriticState(privileged)


__all__ = [
    "CRITIC_DIMENSION", "CRITIC_FIELD_ORDER", "CriticState",
    "QUATERNION_NORM_TOLERANCE", "build_critic_state",
]
