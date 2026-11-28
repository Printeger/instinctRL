"""Actor-clean deterministic velocity-governor decoding."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GovernorConfig:
    """Body-frame governor bounds and optional previous-command smoothing."""

    v_corr_limit: float = 0.5
    velocity_limit: float = 2.0
    smoothing: float = 0.0

    def __post_init__(self) -> None:
        for name in ("v_corr_limit", "velocity_limit", "smoothing"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.v_corr_limit < 0.0:
            raise ValueError("v_corr_limit must be nonnegative")
        if self.velocity_limit <= 0.0:
            raise ValueError("velocity_limit must be positive")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")


@dataclass(frozen=True)
class GovernorResult:
    alpha: torch.Tensor
    v_corr_b: torch.Tensor
    v_gov_b: torch.Tensor


def clip_vector_norm(vector: torch.Tensor, limit: float) -> torch.Tensor:
    """Clip the last-axis Euclidean norm without changing direction."""
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    scale = torch.clamp(float(limit) / norm.clamp_min(torch.finfo(vector.dtype).tiny), max=1.0)
    return vector * scale


def extract_latest_commands(state_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract current command and previous issued command from exact M4 state."""
    if not isinstance(state_vec, torch.Tensor):
        raise TypeError("state_vec must be a torch.Tensor")
    if state_vec.ndim != 2 or state_vec.shape[0] <= 0 or state_vec.shape[1] != 52:
        raise ValueError("state_vec must have shape (N, 52)")
    return state_vec[:, -7:-4], state_vec[:, -4:-1]


def _validate_inputs(
    action: torch.Tensor, state_vec: torch.Tensor, config: GovernorConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(config, GovernorConfig):
        raise TypeError("config must be GovernorConfig")
    if not isinstance(action, torch.Tensor):
        raise TypeError("action must be a torch.Tensor")
    if action.ndim != 2 or action.shape[0] <= 0 or action.shape[1] != 4:
        raise ValueError("action must have shape (N, 4)")
    v_cmd_b, previous_b = extract_latest_commands(state_vec)
    if action.shape[0] != state_vec.shape[0]:
        raise ValueError("action and state_vec batch dimensions must match")
    if action.dtype not in (torch.float32, torch.float64) or state_vec.dtype != action.dtype:
        raise TypeError("action and state_vec must share float32 or float64 dtype")
    if action.device != state_vec.device:
        raise ValueError("action and state_vec must share one device")
    if not bool(torch.isfinite(action).all().item()) or not bool(torch.isfinite(state_vec).all().item()):
        raise ValueError("action and state_vec must be finite")
    if bool(((action < 0.0) | (action > 1.0)).any().item()):
        raise ValueError("action must be in the closed [0, 1] domain")
    return v_cmd_b, previous_b


def decode_governor(
    action: torch.Tensor, state_vec: torch.Tensor, config: GovernorConfig = GovernorConfig()
) -> GovernorResult:
    """Decode normalized action into a bounded body-frame command."""
    v_cmd_b, previous_b = _validate_inputs(action, state_vec, config)
    alpha = action[:, 0:1]
    v_corr_b = (2.0 * action[:, 1:4] - 1.0) * config.v_corr_limit
    candidate = alpha * v_cmd_b + v_corr_b
    if config.smoothing > 0.0:
        candidate = (1.0 - config.smoothing) * candidate + config.smoothing * previous_b
    return GovernorResult(alpha, v_corr_b, clip_vector_norm(candidate, config.velocity_limit))


def b0_governor(
    state_vec: torch.Tensor, velocity_limit: float = 2.0
) -> GovernorResult:
    """Return alpha one, zero correction, and norm-limited current command."""
    config = GovernorConfig(v_corr_limit=0.0, velocity_limit=velocity_limit)
    v_cmd_b, _ = extract_latest_commands(state_vec)
    action = torch.full(
        (state_vec.shape[0], 4), 0.5, dtype=state_vec.dtype, device=state_vec.device
    )
    action[:, 0] = 1.0
    _validate_inputs(action, state_vec, config)
    return GovernorResult(
        torch.ones_like(action[:, 0:1]),
        torch.zeros_like(v_cmd_b),
        clip_vector_norm(v_cmd_b, config.velocity_limit),
    )


__all__ = [
    "GovernorConfig",
    "GovernorResult",
    "b0_governor",
    "clip_vector_norm",
    "decode_governor",
    "extract_latest_commands",
]
