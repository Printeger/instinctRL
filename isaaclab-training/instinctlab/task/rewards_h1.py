"""Frozen ``reward-h1-v2`` and action-independent command-risk calculation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from instinctlab.control.ics import ICSConfig, attenuate_command, extract_lidar_history, mounted_mid360_rays_body

REWARD_H1_VERSION = "reward-h1-v2"
RESIDUAL_NEUTRALITY_VERSION = "reward-h1-zero-neutrality-v1"
REWARD_H1_COMPONENT_KEYS = (
    "reward_velocity", "reward_zero_extra", "reward_anchor", "reward_clearance",
    "reward_intervention", "reward_smooth", "reward_governor_proxy", "reward_final_proxy",
    "reward_preservation", "reward_failure", "reward_total",
)
REWARD_RISK_CONFIG = ICSConfig(
    d_safe_m=1.0, emergency_clearance_m=0.25, a_max_m_s2=2.0,
    active_horizon_margin_m=0.5, min_reliability=0.1, velocity_limit_m_s=2.0,
    latency_s=0.0, clearance_margin_m=0.0, range_rate_filter=False,
    empty_active_set_beta=1.0,
)


class MeasurementCoverageUnavailable(RuntimeError):
    """Raised before learning when the latest reliable-beam fraction is below 0.1."""


@dataclass(frozen=True)
class CommandRisk:
    beta_cmd: torch.Tensor
    emergency_cmd: torch.Tensor
    zero_command: torch.Tensor
    dangerous_command: torch.Tensor
    safe_nonzero_command: torch.Tensor
    reliable_fraction: torch.Tensor


@dataclass(frozen=True)
class RewardH1Result:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]
    risk: CommandRisk


def _vector(name: str, value: torch.Tensor, batch: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 3 or value.shape[0] <= 0:
        raise ValueError(f"{name} must have shape (N, 3)")
    if batch is not None and value.shape[0] != batch:
        raise ValueError(f"{name} batch mismatch")
    if value.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(value).all().item()):
        raise TypeError(f"{name} must be finite float32 or float64")
    return value


def _column(name: str, value: torch.Tensor, reference: torch.Tensor, *, boolean: bool = False) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != (reference.shape[0], 1):
        raise ValueError(f"{name} must have shape (N, 1)")
    if boolean:
        if value.dtype != torch.bool or value.device != reference.device:
            raise TypeError(f"{name} must be boolean on the command device")
    elif value.dtype != reference.dtype or value.device != reference.device or not bool(torch.isfinite(value).all().item()):
        raise TypeError(f"{name} must be finite and share command dtype/device")
    return value


def compute_command_risk(
    lidar_grid: torch.Tensor, command_b: torch.Tensor, *, rays_body: torch.Tensor | None = None,
) -> CommandRisk:
    """Compute q only from the pre-action measurement and raw command."""
    command = _vector("command_b", command_b)
    if lidar_grid.shape[0] != command.shape[0] or lidar_grid.dtype != command.dtype or lidar_grid.device != command.device:
        raise TypeError("lidar_grid and command_b must share batch, dtype and device")
    _, validity, reliability = extract_lidar_history(lidar_grid)
    latest_reliable = validity[:, -1] & (reliability[:, -1] >= REWARD_RISK_CONFIG.min_reliability)
    reliable_fraction = latest_reliable.flatten(1).to(command.dtype).mean(dim=1, keepdim=True)
    if bool((reliable_fraction < 0.1).any().item()):
        raise MeasurementCoverageUnavailable("MEASUREMENT_COVERAGE_UNAVAILABLE: reliable fraction below 0.1")
    rays = rays_body if rays_body is not None else mounted_mid360_rays_body(device=command.device, dtype=command.dtype)
    result = attenuate_command(lidar_grid, rays, command, 0.1, REWARD_RISK_CONFIG)
    zero = torch.linalg.vector_norm(command, dim=-1, keepdim=True) <= 1.0e-3
    emergency = result.metrics["emergency"] == 1.0
    dangerous = (~zero) & ((result.metrics["beta"] < 0.999) | emergency)
    safe = (~zero) & ~dangerous
    return CommandRisk(result.metrics["beta"], emergency, zero, dangerous, safe, reliable_fraction)


def compute_reward_h1(
    *, lidar_grid: torch.Tensor, command_b: torch.Tensor, v_gov_b: torch.Tensor,
    v_final_b: torch.Tensor, previous_issued_v_final_b: torch.Tensor,
    actual_velocity_b: torch.Tensor, anchor_loss: torch.Tensor, anchor_active: torch.Tensor,
    anchor_usable_fraction: torch.Tensor, ics_beta: torch.Tensor, exact_clearance_m: torch.Tensor,
    collision: torch.Tensor, below_height: torch.Tensor, above_height: torch.Tensor,
    rays_body: torch.Tensor | None = None,
    v_corr_b: torch.Tensor | None = None, residual_neutrality_weight: float = 0.0,
) -> RewardH1Result:
    if (isinstance(residual_neutrality_weight, bool) or not isinstance(residual_neutrality_weight, (int, float))
            or not math.isfinite(residual_neutrality_weight) or residual_neutrality_weight < 0):
        raise ValueError("residual neutrality weight must be finite and nonnegative")
    command = _vector("command_b", command_b)
    batch = command.shape[0]
    correction = None
    if v_corr_b is not None:
        correction = _vector("v_corr_b", v_corr_b, batch)
        if correction.dtype != command.dtype or correction.device != command.device:
            raise TypeError("v_corr_b must share command dtype/device")
    if residual_neutrality_weight > 0 and correction is None:
        raise ValueError("positive residual neutrality requires decoded v_corr_b")
    vectors = {
        name: _vector(name, value, batch) for name, value in (
            ("v_gov_b", v_gov_b), ("v_final_b", v_final_b),
            ("previous_issued_v_final_b", previous_issued_v_final_b),
            ("actual_velocity_b", actual_velocity_b),
        )
    }
    if any(value.dtype != command.dtype or value.device != command.device for value in vectors.values()):
        raise TypeError("all reward vectors must share command dtype/device")
    anchor_loss = _column("anchor_loss", anchor_loss, command)
    anchor_active = _column("anchor_active", anchor_active, command)
    anchor_usable_fraction = _column("anchor_usable_fraction", anchor_usable_fraction, command)
    beta = _column("ics_beta", ics_beta, command)
    clearance = _column("exact_clearance_m", exact_clearance_m, command)
    collision = _column("collision", collision, command, boolean=True)
    below = _column("below_height", below_height, command, boolean=True)
    above = _column("above_height", above_height, command, boolean=True)
    if bool((anchor_loss < 0).any().item()) or bool((clearance < 0).any().item()):
        raise ValueError("anchor loss and clearance must be nonnegative")
    if bool(((anchor_usable_fraction < 0) | (anchor_usable_fraction > 1)).any().item()):
        raise ValueError("anchor usable fraction must be in [0,1]")
    if bool(((beta < 0) | (beta > 1)).any().item()):
        raise ValueError("ICS beta must be in [0,1]")
    risk = compute_command_risk(lidar_grid, command, rays_body=rays_body)
    dtype = command.dtype
    h = risk.zero_command.to(dtype)
    q = risk.dangerous_command.to(dtype)
    g = risk.safe_nonzero_command.to(dtype)
    actual = vectors["actual_velocity_b"]
    governor = vectors["v_gov_b"]
    final = vectors["v_final_b"]
    previous = vectors["previous_issued_v_final_b"]
    command_norm = torch.linalg.vector_norm(command, dim=-1, keepdim=True)
    ratio = torch.where(~risk.zero_command, torch.linalg.vector_norm(final, dim=-1, keepdim=True) / command_norm.clamp_min(1.0e-3), torch.zeros_like(command_norm))
    failure = collision | below | above
    anchor_gate = h * (anchor_active == 1.0).to(dtype) * (anchor_usable_fraction >= 0.1).to(dtype)
    components = {
        "reward_velocity": -(1.0 - q) * torch.linalg.vector_norm(actual - command, dim=-1, keepdim=True) - q * torch.linalg.vector_norm(actual, dim=-1, keepdim=True),
        "reward_zero_extra": (torch.zeros_like(command_norm) if residual_neutrality_weight == 0 else
                              -residual_neutrality_weight * h * (correction / 0.1).square().sum(dim=-1, keepdim=True)),
        "reward_anchor": -4.0 * anchor_gate * anchor_loss,
        "reward_clearance": -0.5 * torch.relu(1.0 - clearance),
        "reward_intervention": -0.05 * ((1.0 - beta) + torch.linalg.vector_norm(governor - final, dim=-1, keepdim=True)),
        "reward_smooth": -0.05 * torch.linalg.vector_norm(final - previous, dim=-1, keepdim=True),
        "reward_governor_proxy": -0.1 * g * torch.linalg.vector_norm(governor - command, dim=-1, keepdim=True),
        "reward_final_proxy": -0.25 * g * torch.linalg.vector_norm(final - command, dim=-1, keepdim=True),
        "reward_preservation": -0.5 * g * (torch.relu(0.75 - ratio) + torch.relu(ratio - 1.05)),
        "reward_failure": -300.0 * failure.to(dtype),
    }
    total = sum(components.values())
    components["reward_total"] = total
    if tuple(components) != REWARD_H1_COMPONENT_KEYS or any(not bool(torch.isfinite(v).all().item()) for v in components.values()):
        raise RuntimeError("reward-h1 component contract violation")
    diagnostics = {
        "h": h, "q": q, "g": g, "beta_cmd": risk.beta_cmd,
        "emergency_cmd": risk.emergency_cmd.to(dtype), "reliable_fraction": risk.reliable_fraction,
        "preservation_ratio": ratio, "failure_union": failure.to(dtype),
    }
    return RewardH1Result(total, components, diagnostics, risk)


__all__ = [
    "CommandRisk", "MeasurementCoverageUnavailable", "REWARD_H1_COMPONENT_KEYS", "REWARD_H1_VERSION",
    "REWARD_RISK_CONFIG", "RewardH1Result", "compute_command_risk", "compute_reward_h1",
]
