"""Controller-private scalar-first body-to-world command adapter."""

from __future__ import annotations

import torch

QUATERNION_NORM_TOLERANCE = 1.0e-5


def validate_adapter_inputs(v_final_b: torch.Tensor, quaternion_wxyz: torch.Tensor) -> None:
    if not isinstance(v_final_b, torch.Tensor) or not isinstance(quaternion_wxyz, torch.Tensor):
        raise TypeError("adapter inputs must be torch.Tensor instances")
    if v_final_b.ndim != 2 or v_final_b.shape[0] <= 0 or v_final_b.shape[1] != 3:
        raise ValueError("v_final_b must have shape (N, 3)")
    if quaternion_wxyz.shape != (v_final_b.shape[0], 4):
        raise ValueError("quaternion_wxyz must have shape (N, 4)")
    if v_final_b.dtype not in (torch.float32, torch.float64) or quaternion_wxyz.dtype != v_final_b.dtype:
        raise TypeError("adapter inputs must share float32 or float64 dtype")
    if quaternion_wxyz.device != v_final_b.device:
        raise ValueError("adapter inputs must share one device")
    if not bool(torch.isfinite(v_final_b).all().item()) or not bool(torch.isfinite(quaternion_wxyz).all().item()):
        raise ValueError("adapter inputs must be finite")
    norm_error = torch.abs(torch.linalg.vector_norm(quaternion_wxyz, dim=-1) - 1.0)
    if bool((norm_error > QUATERNION_NORM_TOLERANCE).any().item()):
        raise ValueError("quaternion_wxyz must be unit length within tolerance")


def rotate_body_to_world(v_final_b: torch.Tensor, quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Normalize only tiny drift, then apply q*v*q-conjugate in tensor form."""
    validate_adapter_inputs(v_final_b, quaternion_wxyz)
    quaternion = quaternion_wxyz / torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True)
    scalar = quaternion[:, 0:1]
    imaginary = quaternion[:, 1:4]
    return (
        v_final_b
        + 2.0 * scalar * torch.cross(imaginary, v_final_b, dim=-1)
        + 2.0 * torch.cross(imaginary, torch.cross(imaginary, v_final_b, dim=-1), dim=-1)
    )


__all__ = ["QUATERNION_NORM_TOLERANCE", "rotate_body_to_world", "validate_adapter_inputs"]
