"""Actor-clean measurement-space anchor lifecycle and robust error."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from instinctlab.observations.actor import M3_MAX_RANGE_M, M3_MIN_RANGE_M

RESET_NONE = 0
RESET_EPISODE = 1
RESET_EXPLICIT = 2
RESET_COMMAND_EXIT = 3
RESET_MINIMUM_VALID = 4

ANCHOR_METRIC_KEYS = frozenset(
    ("active", "reset_code", "usable_fraction", "mean_absolute_error_m", "huber_loss")
)
ANCHOR_CACHE_KEYS = frozenset(("range_star_m", "validity_star", "reliability_star"))


@dataclass(frozen=True)
class AnchorConfig:
    eps_enter: float = 0.05
    eps_exit: float = 0.15
    min_valid_fraction: float = 0.1
    min_reliability: float = 0.1
    huber_delta_m: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "eps_enter", "eps_exit", "min_valid_fraction", "min_reliability", "huber_delta_m"
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.eps_enter < self.eps_exit:
            raise ValueError("anchor thresholds require 0 <= eps_enter < eps_exit")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        if not 0.0 <= self.min_reliability <= 1.0:
            raise ValueError("min_reliability must be in [0, 1]")
        if self.huber_delta_m <= 0.0:
            raise ValueError("huber_delta_m must be positive")


@dataclass(frozen=True)
class AnchorResult:
    metrics: dict[str, torch.Tensor]


def huber_loss(error: torch.Tensor, delta_m: float) -> torch.Tensor:
    absolute = error.abs()
    return torch.where(
        absolute <= delta_m,
        0.5 * error.square(),
        delta_m * (absolute - 0.5 * delta_m),
    )


class MeasurementAnchor:
    """Per-environment anchor whose dense reference remains internal."""

    def __init__(self, config: AnchorConfig = AnchorConfig()) -> None:
        if not isinstance(config, AnchorConfig):
            raise TypeError("config must be AnchorConfig")
        self.config = config
        self._active: torch.Tensor | None = None
        self._range_star_m: torch.Tensor | None = None
        self._validity_star: torch.Tensor | None = None
        self._reliability_star: torch.Tensor | None = None

    def validate_update(
        self,
        ranges_m: torch.Tensor,
        validity: torch.Tensor,
        reliability: torch.Tensor,
        v_cmd_b: torch.Tensor,
        episode_reset: torch.Tensor,
        explicit_reset: torch.Tensor,
    ) -> None:
        tensors = (ranges_m, validity, reliability, v_cmd_b, episode_reset, explicit_reset)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("all anchor inputs must be torch.Tensor instances")
        if ranges_m.ndim != 3 or tuple(ranges_m.shape[1:]) != (59, 360) or ranges_m.shape[0] <= 0:
            raise ValueError("ranges_m must have shape (N, 59, 360)")
        if validity.shape != ranges_m.shape or reliability.shape != ranges_m.shape:
            raise ValueError("validity and reliability must match ranges_m")
        batch = ranges_m.shape[0]
        if v_cmd_b.shape != (batch, 3):
            raise ValueError("v_cmd_b must have shape (N, 3)")
        if episode_reset.shape != (batch,) or explicit_reset.shape != (batch,):
            raise ValueError("reset masks must have shape (N,)")
        if validity.dtype != torch.bool or episode_reset.dtype != torch.bool or explicit_reset.dtype != torch.bool:
            raise TypeError("validity and reset masks must have boolean dtype")
        if ranges_m.dtype not in (torch.float32, torch.float64):
            raise TypeError("anchor floating inputs must use float32 or float64")
        if reliability.dtype != ranges_m.dtype or v_cmd_b.dtype != ranges_m.dtype:
            raise TypeError("anchor floating inputs must share one dtype")
        if len({value.device for value in tensors}) != 1:
            raise ValueError("all anchor inputs must share one device")
        if not all(bool(torch.isfinite(value).all().item()) for value in (ranges_m, reliability, v_cmd_b)):
            raise ValueError("anchor floating inputs must be finite")
        if bool(((ranges_m < M3_MIN_RANGE_M) | (ranges_m > M3_MAX_RANGE_M)).any().item()):
            raise ValueError("ranges_m must preserve the M3 range domain")
        if bool(((~validity) & (ranges_m != M3_MAX_RANGE_M)).any().item()):
            raise ValueError("invalid ranges must use the exact M3 fill")
        if bool(((reliability < 0.0) | (reliability > 1.0)).any().item()):
            raise ValueError("reliability must be in [0, 1]")
        if bool((reliability[~validity] != 0.0).any().item()):
            raise ValueError("invalid ranges must have zero reliability")
        if self._active is not None and (
            self._active.shape[0] != batch
            or self._active.device != ranges_m.device
            or self._range_star_m is None
            or self._range_star_m.dtype != ranges_m.dtype
        ):
            raise ValueError("anchor batch, device, and dtype are fixed after first update")

    def update(
        self,
        ranges_m: torch.Tensor,
        validity: torch.Tensor,
        reliability: torch.Tensor,
        v_cmd_b: torch.Tensor,
        episode_reset: torch.Tensor,
        explicit_reset: torch.Tensor,
    ) -> AnchorResult:
        """Atomically validate, then update lifecycle and scalar metrics."""
        self.validate_update(ranges_m, validity, reliability, v_cmd_b, episode_reset, explicit_reset)
        if self._active is None:
            self._active = torch.zeros(ranges_m.shape[0], dtype=torch.bool, device=ranges_m.device)
            self._range_star_m = torch.zeros_like(ranges_m)
            self._validity_star = torch.zeros_like(validity)
            self._reliability_star = torch.zeros_like(reliability)
        assert self._range_star_m is not None
        assert self._validity_star is not None
        assert self._reliability_star is not None
        reset_code = torch.zeros_like(self._active, dtype=torch.long)
        command_exit = self._active & (torch.linalg.vector_norm(v_cmd_b, dim=-1) >= self.config.eps_exit)
        usable_current = validity & (reliability >= self.config.min_reliability)
        current_fraction = usable_current.flatten(1).to(ranges_m.dtype).mean(dim=1)
        insufficient = current_fraction < self.config.min_valid_fraction
        reset_code = torch.where(command_exit, RESET_COMMAND_EXIT, reset_code)
        reset_code = torch.where(insufficient, RESET_MINIMUM_VALID, reset_code)
        reset_code = torch.where(explicit_reset, RESET_EXPLICIT, reset_code)
        reset_code = torch.where(episode_reset, RESET_EPISODE, reset_code)
        reset = reset_code != RESET_NONE
        self._active[reset] = False
        self._range_star_m[reset] = 0.0
        self._validity_star[reset] = False
        self._reliability_star[reset] = 0.0

        capture = (
            (~self._active)
            & (~reset)
            & (~insufficient)
            & (torch.linalg.vector_norm(v_cmd_b, dim=-1) <= self.config.eps_enter)
        )
        self._range_star_m[capture] = ranges_m[capture]
        self._validity_star[capture] = validity[capture]
        self._reliability_star[capture] = reliability[capture]
        self._active[capture] = True

        usable = (
            self._active[:, None, None]
            & usable_current
            & self._validity_star
            & (self._reliability_star >= self.config.min_reliability)
        )
        weight = torch.minimum(reliability, self._reliability_star) * usable.to(ranges_m.dtype)
        error = ranges_m - self._range_star_m
        denominator = usable.flatten(1).sum(dim=1, keepdim=True).clamp_min(1)
        usable_fraction = usable.flatten(1).to(ranges_m.dtype).mean(dim=1, keepdim=True)
        mean_absolute = (error.abs() * weight).flatten(1).sum(dim=1, keepdim=True) / denominator
        robust = (huber_loss(error, self.config.huber_delta_m) * weight).flatten(1).sum(dim=1, keepdim=True) / denominator
        metrics = {
            "active": self._active[:, None].to(ranges_m.dtype).clone(),
            "reset_code": reset_code[:, None].clone(),
            "usable_fraction": usable_fraction,
            "mean_absolute_error_m": mean_absolute,
            "huber_loss": robust,
        }
        return AnchorResult(metrics)

    def cache_snapshot(self) -> dict[str, torch.Tensor]:
        """Return detached diagnostic copies, never an actor/public pipeline output."""
        if self._active is None:
            return {}
        assert self._range_star_m is not None
        assert self._validity_star is not None
        assert self._reliability_star is not None
        return {
            "range_star_m": self._range_star_m.detach().clone(),
            "validity_star": self._validity_star.detach().clone(),
            "reliability_star": self._reliability_star.detach().clone(),
        }


__all__ = [
    "ANCHOR_CACHE_KEYS", "ANCHOR_METRIC_KEYS", "AnchorConfig", "AnchorResult",
    "MeasurementAnchor", "RESET_COMMAND_EXIT", "RESET_EPISODE", "RESET_EXPLICIT",
    "RESET_MINIMUM_VALID", "RESET_NONE", "huber_loss",
]
