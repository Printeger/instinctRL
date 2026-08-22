"""Strict pure-tensor reward contract for the command-governor task."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

REWARD_COMPONENT_KEYS = (
    "reward_actual_tracking",
    "reward_null_actual_speed",
    "reward_anchor",
    "reward_clearance",
    "reward_collision",
    "reward_intervention_usage",
    "reward_issued_smoothness",
    "reward_governor_proxy",
    "reward_final_proxy",
    "reward_preservation",
    "reward_total",
)
REWARD_DIAGNOSTIC_KEYS = (
    "command_active",
    "ordinary_tracking_gate",
    "actual_tracking_error_m_s",
    "null_actual_speed_m_s",
    "anchor_gate",
    "clearance_violation_m",
    "intervention_command_delta_m_s",
    "issued_command_delta_m_s",
    "preservation_ratio",
    "preservation_violation",
)


@dataclass(frozen=True)
class RewardConfig:
    """Fixed migration defaults; weights are nonnegative and tracking is mandatory."""

    actual_tracking_weight: float = 1.0
    null_actual_speed_weight: float = 2.0
    anchor_weight: float = 4.0
    clearance_weight: float = 0.5
    collision_weight: float = 10.0
    intervention_usage_weight: float = 0.05
    issued_smoothness_weight: float = 0.05
    governor_proxy_weight: float = 0.1
    final_proxy_weight: float = 0.25
    preservation_weight: float = 0.5
    clearance_safe_m: float = 0.8
    clearance_margin_m: float = 0.2
    preservation_lower: float = 0.75
    preservation_upper: float = 1.05
    command_epsilon_m_s: float = 1.0e-3
    full_compliance_beta: float = 0.999

    def __post_init__(self) -> None:
        numeric = (
            "actual_tracking_weight", "null_actual_speed_weight", "anchor_weight",
            "clearance_weight", "collision_weight", "intervention_usage_weight",
            "issued_smoothness_weight", "governor_proxy_weight", "final_proxy_weight",
            "preservation_weight", "clearance_safe_m", "clearance_margin_m",
            "preservation_lower", "preservation_upper", "command_epsilon_m_s",
            "full_compliance_beta",
        )
        for name in numeric:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise TypeError(f"{name} must be a finite real scalar")
            object.__setattr__(self, name, float(value))
        weights = numeric[:10]
        if any(getattr(self, name) < 0.0 for name in weights):
            raise ValueError("reward weights must be nonnegative")
        if self.actual_tracking_weight <= 0.0:
            raise ValueError("actual_tracking_weight must remain positive; proxy-only reward is forbidden")
        if self.clearance_safe_m <= 0.0 or self.clearance_margin_m < 0.0:
            raise ValueError("clearance_safe_m must be positive and clearance_margin_m nonnegative")
        if self.preservation_lower < 0.0 or self.preservation_upper < self.preservation_lower:
            raise ValueError("preservation bounds must satisfy 0 <= lower <= upper")
        if self.command_epsilon_m_s <= 0.0:
            raise ValueError("command_epsilon_m_s must be positive")
        if not 0.0 < self.full_compliance_beta <= 1.0:
            raise ValueError("full_compliance_beta must be in (0, 1]")


@dataclass(frozen=True)
class RewardResult:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]


def _vector(name: str, value: torch.Tensor, batch: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if batch is not None and value.shape[0] != batch:
        raise ValueError(f"{name} batch size must match")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")
    return value


def _scalar(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != (reference.shape[0], 1):
        raise ValueError(f"{name} must have shape (N, 1)")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise TypeError(f"{name} must share command dtype and device")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")
    return value


def _label(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != (reference.shape[0], 1) or value.dtype != torch.bool:
        raise TypeError(f"{name} must be boolean with shape (N, 1)")
    if value.device != reference.device:
        raise ValueError(f"{name} must share command device")
    return value


def compute_reward(
    *,
    v_cmd_b: torch.Tensor,
    v_gov_b: torch.Tensor,
    v_final_b: torch.Tensor,
    previous_issued_v_final_b: torch.Tensor,
    actual_velocity_b: torch.Tensor,
    anchor_loss: torch.Tensor,
    anchor_active: torch.Tensor,
    ics_beta: torch.Tensor,
    ics_emergency: torch.Tensor,
    exact_clearance_m: torch.Tensor,
    collision: torch.Tensor,
    config: RewardConfig = RewardConfig(),
) -> RewardResult:
    """Compute the exact exposed composition without clipping or hidden repair."""
    if not isinstance(config, RewardConfig):
        raise TypeError("config must be RewardConfig")
    command = _vector("v_cmd_b", v_cmd_b)
    batch = command.shape[0]
    governor = _vector("v_gov_b", v_gov_b, batch)
    final = _vector("v_final_b", v_final_b, batch)
    previous = _vector("previous_issued_v_final_b", previous_issued_v_final_b, batch)
    actual = _vector("actual_velocity_b", actual_velocity_b, batch)
    for name, value in (("v_gov_b", governor), ("v_final_b", final),
                        ("previous_issued_v_final_b", previous), ("actual_velocity_b", actual)):
        if value.dtype != command.dtype or value.device != command.device:
            raise TypeError(f"{name} must share command dtype and device")
    anchor_error = _scalar("anchor_loss", anchor_loss, command)
    beta = _scalar("ics_beta", ics_beta, command)
    clearance = _scalar("exact_clearance_m", exact_clearance_m, command)
    anchor_on = _label("anchor_active", anchor_active, command)
    emergency = _label("ics_emergency", ics_emergency, command)
    collided = _label("collision", collision, command)
    if bool((anchor_error < 0.0).any().item()):
        raise ValueError("anchor_loss must be nonnegative")
    if bool(((beta < 0.0) | (beta > 1.0)).any().item()):
        raise ValueError("ics_beta must be in [0, 1]")
    if bool((clearance < 0.0).any().item()):
        raise ValueError("exact_clearance_m must be nonnegative")
    if bool((emergency & (beta != 0.0)).any().item()):
        raise ValueError("ICS emergency requires exact zero beta")

    command_norm = torch.linalg.vector_norm(command, dim=-1, keepdim=True)
    command_active_bool = command_norm > config.command_epsilon_m_s
    command_active = command_active_bool.to(command.dtype)
    null_command = 1.0 - command_active
    ordinary_gate_bool = command_active_bool & (beta >= config.full_compliance_beta) & ~emergency
    ordinary_gate = ordinary_gate_bool.to(command.dtype)
    actual_error = torch.linalg.vector_norm(actual - command, dim=-1, keepdim=True)
    null_actual_speed = torch.linalg.vector_norm(actual, dim=-1, keepdim=True)
    anchor_gate = null_command * anchor_on.to(command.dtype)
    clearance_threshold = config.clearance_safe_m + config.clearance_margin_m
    clearance_violation = torch.relu(clearance_threshold - clearance)
    intervention_delta = torch.linalg.vector_norm(governor - final, dim=-1, keepdim=True)
    issued_delta = torch.linalg.vector_norm(final - previous, dim=-1, keepdim=True)
    preservation_ratio = torch.where(
        command_active_bool,
        torch.linalg.vector_norm(final, dim=-1, keepdim=True) / command_norm.clamp_min(config.command_epsilon_m_s),
        torch.zeros_like(command_norm),
    )
    preservation_violation = (
        torch.relu(config.preservation_lower - preservation_ratio)
        + torch.relu(preservation_ratio - config.preservation_upper)
    ) * ordinary_gate

    components: dict[str, torch.Tensor] = {
        "reward_actual_tracking": -config.actual_tracking_weight * ordinary_gate * actual_error,
        "reward_null_actual_speed": -config.null_actual_speed_weight * null_command * null_actual_speed,
        "reward_anchor": -config.anchor_weight * anchor_gate * anchor_error,
        "reward_clearance": -config.clearance_weight * clearance_violation,
        "reward_collision": -config.collision_weight * collided.to(command.dtype),
        "reward_intervention_usage": -config.intervention_usage_weight * ((1.0 - beta) + intervention_delta),
        "reward_issued_smoothness": -config.issued_smoothness_weight * issued_delta,
        "reward_governor_proxy": -config.governor_proxy_weight * ordinary_gate
        * torch.linalg.vector_norm(governor - command, dim=-1, keepdim=True),
        "reward_final_proxy": -config.final_proxy_weight * ordinary_gate
        * torch.linalg.vector_norm(final - command, dim=-1, keepdim=True),
        "reward_preservation": -config.preservation_weight * preservation_violation,
    }
    total = sum(components.values())
    components["reward_total"] = total
    if tuple(components) != REWARD_COMPONENT_KEYS or any(value.shape != (batch, 1) for value in components.values()):
        raise RuntimeError("internal reward component contract violation")
    if any(not bool(torch.isfinite(value).all().item()) for value in components.values()):
        raise FloatingPointError("reward computation produced a non-finite value")
    diagnostics = {
        "command_active": command_active,
        "ordinary_tracking_gate": ordinary_gate,
        "actual_tracking_error_m_s": actual_error,
        "null_actual_speed_m_s": null_actual_speed * null_command,
        "anchor_gate": anchor_gate,
        "clearance_violation_m": clearance_violation,
        "intervention_command_delta_m_s": intervention_delta,
        "issued_command_delta_m_s": issued_delta,
        "preservation_ratio": preservation_ratio,
        "preservation_violation": preservation_violation,
    }
    if tuple(diagnostics) != REWARD_DIAGNOSTIC_KEYS:
        raise RuntimeError("internal reward diagnostic contract violation")
    return RewardResult(total, components, diagnostics)


__all__ = [
    "REWARD_COMPONENT_KEYS", "REWARD_DIAGNOSTIC_KEYS", "RewardConfig",
    "RewardResult", "compute_reward",
]
