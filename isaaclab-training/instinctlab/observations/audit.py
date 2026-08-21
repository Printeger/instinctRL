"""Fail-closed audit for the exact M4 actor-visible observation."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .actor import (
    AZIMUTH_SAMPLES,
    DEFAULT_GRAVITY_NORM_TOLERANCE,
    ELEVATION_SAMPLES,
    HISTORY_LENGTH,
    M3_MAX_RANGE_M,
    M3_MIN_RANGE_M,
)

EXACT_ACTOR_KEYS = frozenset(("lidar_grid", "state_vec"))
APPROVED_COMMAND_FIELD_NAMES = frozenset(
    ("commanded_body_velocity_m_s", "previous_issued_body_velocity_m_s")
)
FORBIDDEN_FIELD_TOKENS = (
    "pose",
    "position",
    "odometry",
    "actual_velocity",
    "estimated_velocity",
    "reconstructed_velocity",
    "translational_velocity",
    "root_state",
    "map",
    "slam",
    "obstacle_truth",
    "privileged",
    "simulator_state",
    "future",
    "critic",
    "reward",
    "logging",
    "controller",
)


class ActorObservationAuditError(ValueError):
    """Raised whenever an actor observation fails closed."""


def _forbidden_reason(key: object) -> str | None:
    normalized = str(key).strip().lower()
    if normalized in APPROVED_COMMAND_FIELD_NAMES:
        return None
    for token in FORBIDDEN_FIELD_TOKENS:
        if token in normalized:
            return token
    if normalized in {"velocity", "vel", "vel_g", "vel_w"}:
        return normalized
    return None


def _scan_keys(value: object, path: str = "actor") -> None:
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        child_path = f"{path}.{key}"
        reason = _forbidden_reason(key)
        if reason is not None:
            raise ActorObservationAuditError(
                f"forbidden actor field at {child_path}: matched {reason}"
            )
        _scan_keys(child, child_path)


def audit_actor_observation(observation: Mapping[str, torch.Tensor]) -> None:
    """Require exact keys, tensors, shape, dtype, device, finiteness, and domains."""
    if not isinstance(observation, Mapping):
        raise ActorObservationAuditError("actor observation must be a mapping")
    _scan_keys(observation)
    if set(observation.keys()) != EXACT_ACTOR_KEYS:
        raise ActorObservationAuditError(
            f"actor observation keys must be exactly {sorted(EXACT_ACTOR_KEYS)}"
        )
    lidar_grid = observation["lidar_grid"]
    state_vec = observation["state_vec"]
    if not isinstance(lidar_grid, torch.Tensor) or not isinstance(state_vec, torch.Tensor):
        raise ActorObservationAuditError("both actor values must be torch.Tensor instances")
    if lidar_grid.ndim != 4:
        raise ActorObservationAuditError("lidar_grid must be rank four")
    batch_size = lidar_grid.shape[0]
    expected_lidar_shape = (batch_size, HISTORY_LENGTH * 3, ELEVATION_SAMPLES, AZIMUTH_SAMPLES)
    expected_state_shape = (batch_size, HISTORY_LENGTH * 13)
    if tuple(lidar_grid.shape) != expected_lidar_shape:
        raise ActorObservationAuditError(f"lidar_grid must have shape {expected_lidar_shape}")
    if state_vec.ndim != 2 or tuple(state_vec.shape) != expected_state_shape:
        raise ActorObservationAuditError(f"state_vec must have shape {expected_state_shape}")
    if lidar_grid.shape[0] <= 0:
        raise ActorObservationAuditError("actor batch must be non-empty")
    if lidar_grid.dtype not in (torch.float32, torch.float64) or state_vec.dtype != lidar_grid.dtype:
        raise ActorObservationAuditError("actor tensors must share torch.float32 or torch.float64 dtype")
    if state_vec.device != lidar_grid.device:
        raise ActorObservationAuditError("actor tensors must share one device")
    if not bool(torch.isfinite(lidar_grid).all().item()) or not bool(torch.isfinite(state_vec).all().item()):
        raise ActorObservationAuditError("actor tensors must be finite")

    ranges = lidar_grid[:, 0::3]
    validity = lidar_grid[:, 1::3]
    reliability = lidar_grid[:, 2::3]
    state_frames = state_vec.reshape(batch_size, HISTORY_LENGTH, 13)
    lidar_zero = (
        (ranges == 0.0).flatten(start_dim=2).all(dim=-1)
        & (validity == 0.0).flatten(start_dim=2).all(dim=-1)
        & (reliability == 0.0).flatten(start_dim=2).all(dim=-1)
    )
    state_zero = (state_frames == 0.0).all(dim=-1)
    zero_frames = lidar_zero & state_zero
    if bool((lidar_zero != state_zero).any().item()):
        raise ActorObservationAuditError("zero-filled LiDAR and state history frames must be coherent")
    if not bool(((validity == 0.0) | (validity == 1.0)).all().item()):
        raise ActorObservationAuditError("validity history channels must be exactly binary")
    if bool(((reliability < 0.0) | (reliability > 1.0)).any().item()):
        raise ActorObservationAuditError("reliability history channels must be in [0, 1]")
    if bool((reliability > validity).any().item()):
        raise ActorObservationAuditError("reliability must be zero for invalid beams")
    if bool((state_vec[:, 12::13] < 0.0).any().item()):
        raise ActorObservationAuditError("sensor age history must be nonnegative")

    populated = ~zero_frames
    populated_expanded = populated.unsqueeze(-1).unsqueeze(-1)
    bad_populated_range = populated_expanded & (
        (ranges < M3_MIN_RANGE_M) | (ranges > M3_MAX_RANGE_M)
    )
    if bool(bad_populated_range.any().item()):
        raise ActorObservationAuditError(
            "populated range frames must preserve the M3 closed [0.1, 40.0] metre domain"
        )
    invalid_nonfill = populated_expanded & (validity == 0.0) & (ranges != M3_MAX_RANGE_M)
    if bool(invalid_nonfill.any().item()):
        raise ActorObservationAuditError(
            "invalid beams in populated frames must use the exact 40.0 metre M3 fill"
        )
    gravity = state_frames[:, :, 3:6]
    gravity_norm = torch.linalg.vector_norm(gravity, dim=-1)
    bad_gravity = populated & (
        torch.abs(gravity_norm - 1.0) > DEFAULT_GRAVITY_NORM_TOLERANCE
    )
    if bool(bad_gravity.any().item()):
        raise ActorObservationAuditError("populated gravity history slices must be unit length")


__all__ = [
    "APPROVED_COMMAND_FIELD_NAMES",
    "ActorObservationAuditError",
    "EXACT_ACTOR_KEYS",
    "FORBIDDEN_FIELD_TOKENS",
    "audit_actor_observation",
]
