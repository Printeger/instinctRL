"""Strict collision, height-bound, and timeout task termination semantics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

TERMINATION_NONE = 0
TERMINATION_TIMEOUT = 1
TERMINATION_ABOVE_HEIGHT = 2
TERMINATION_BELOW_HEIGHT = 3
TERMINATION_COLLISION = 4
TERMINATION_MASK_KEYS = (
    "collision",
    "below_height",
    "above_height",
    "timeout",
)


@dataclass(frozen=True)
class TerminationConfig:
    minimum_height_w_m: float = 0.5
    maximum_height_w_m: float = 4.0
    maximum_episode_steps: int = 1000

    def __post_init__(self) -> None:
        for name in ("minimum_height_w_m", "maximum_height_w_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise TypeError(f"{name} must be a finite real scalar")
            object.__setattr__(self, name, float(value))
        if self.minimum_height_w_m < 0.0 or self.maximum_height_w_m <= self.minimum_height_w_m:
            raise ValueError("height bounds must satisfy 0 <= minimum < maximum")
        if isinstance(self.maximum_episode_steps, bool) or not isinstance(self.maximum_episode_steps, int):
            raise TypeError("maximum_episode_steps must be an integer")
        if self.maximum_episode_steps <= 0:
            raise ValueError("maximum_episode_steps must be positive")


@dataclass(frozen=True)
class TerminationResult:
    terminated: torch.Tensor
    truncated: torch.Tensor
    done: torch.Tensor
    reason_code: torch.Tensor
    masks: dict[str, torch.Tensor]


def compute_termination(
    *,
    height_w_m: torch.Tensor,
    collision: torch.Tensor,
    elapsed_steps: torch.Tensor,
    config: TerminationConfig = TerminationConfig(),
) -> TerminationResult:
    """Collision > below > above > timeout reason precedence; masks remain independent."""
    if not isinstance(config, TerminationConfig):
        raise TypeError("config must be TerminationConfig")
    if not isinstance(height_w_m, torch.Tensor) or height_w_m.ndim != 2 or height_w_m.shape[0] <= 0 or height_w_m.shape[1] != 1:
        raise ValueError("height_w_m must have shape (N, 1)")
    if height_w_m.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(height_w_m).all().item()):
        raise TypeError("height_w_m must be finite float32 or float64")
    expected = (height_w_m.shape[0], 1)
    if not isinstance(collision, torch.Tensor) or collision.shape != expected or collision.dtype != torch.bool:
        raise TypeError("collision must be boolean with shape (N, 1)")
    if collision.device != height_w_m.device:
        raise ValueError("collision must share height device")
    if not isinstance(elapsed_steps, torch.Tensor) or elapsed_steps.shape != expected or elapsed_steps.dtype != torch.long:
        raise TypeError("elapsed_steps must be torch.long with shape (N, 1)")
    if elapsed_steps.device != height_w_m.device or bool((elapsed_steps < 0).any().item()):
        raise ValueError("elapsed_steps must be nonnegative and share height device")

    below = height_w_m < config.minimum_height_w_m
    above = height_w_m > config.maximum_height_w_m
    timeout = elapsed_steps >= config.maximum_episode_steps
    terminated = collision | below | above
    truncated = timeout
    done = terminated | truncated
    reason = torch.full(expected, TERMINATION_NONE, dtype=torch.long, device=height_w_m.device)
    reason = torch.where(timeout, torch.full_like(reason, TERMINATION_TIMEOUT), reason)
    reason = torch.where(above, torch.full_like(reason, TERMINATION_ABOVE_HEIGHT), reason)
    reason = torch.where(below, torch.full_like(reason, TERMINATION_BELOW_HEIGHT), reason)
    reason = torch.where(collision, torch.full_like(reason, TERMINATION_COLLISION), reason)
    masks = {"collision": collision, "below_height": below, "above_height": above, "timeout": timeout}
    if tuple(masks) != TERMINATION_MASK_KEYS:
        raise RuntimeError("internal termination mask contract violation")
    return TerminationResult(terminated, truncated, done, reason, masks)


__all__ = [
    "TERMINATION_ABOVE_HEIGHT", "TERMINATION_BELOW_HEIGHT", "TERMINATION_COLLISION",
    "TERMINATION_MASK_KEYS", "TERMINATION_NONE", "TERMINATION_TIMEOUT",
    "TerminationConfig", "TerminationResult", "compute_termination",
]
