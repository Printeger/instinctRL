"""Strict pure-tensor M7 evaluation metrics with explicit aggregation state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
import math

import torch

from instinctlab.task.termination import (
    TERMINATION_ABOVE_HEIGHT,
    TERMINATION_BELOW_HEIGHT,
    TERMINATION_COLLISION,
    TERMINATION_NONE,
    TERMINATION_TIMEOUT,
)

EVALUATION_INPUT_KEYS = (
    "commanded_velocity_b_m_s",
    "actual_velocity_b_m_s",
    "issued_velocity_b_m_s",
    "position_w_m",
    "station_reference_position_w_m",
    "anchor_error_m",
    "anchor_valid",
    "minimum_clearance_m",
    "clearance_valid",
    "collision",
    "ics_beta",
    "ics_emergency",
    "ics_worst_margin_m_s",
    "terminated",
    "truncated",
    "termination_reason",
    "termination_collision",
    "termination_below_height",
    "termination_above_height",
    "termination_timeout",
    "observability_value",
    "observability_valid",
    "observability_source",
)
METRIC_KEYS = (
    "tracking_rmse_m_s",
    "null_station_drift_mean_m",
    "null_station_speed_mean_m_s",
    "anchor_error_mean_m",
    "minimum_clearance_m",
    "collision_rate",
    "ics_violation_rate",
    "command_preservation_ratio",
    "intervention_frequency",
    "observability_mean",
)
STEP_VALUE_KEYS = (
    "tracking_squared_error_m2_s2",
    "null_station_drift_m",
    "null_station_speed_m_s",
    "anchor_error_m",
    "minimum_clearance_m",
    "collision",
    "ics_violation",
    "command_preservation_ratio",
    "intervention",
    "observability_value",
)
STEP_MASK_KEYS = (
    "all_samples",
    "null_command",
    "anchor_valid",
    "clearance_valid",
    "preservation_valid",
    "observability_valid",
)


class ObservabilitySource(IntEnum):
    """Per-sample evaluation-only observability source classification."""

    UNAVAILABLE = 0
    PROXY = 1
    EXACT = 2


@dataclass(frozen=True)
class EvaluationMetricConfig:
    schema_version: str = "m7.evaluation.v1"
    null_command_threshold_m_s: float = 1.0e-3
    full_compliance_beta: float = 0.999

    def __post_init__(self) -> None:
        if self.schema_version != "m7.evaluation.v1":
            raise ValueError("schema_version must be exactly m7.evaluation.v1")
        for name in ("null_command_threshold_m_s", "full_compliance_beta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if self.null_command_threshold_m_s <= 0.0:
            raise ValueError("null_command_threshold_m_s must be positive")
        if not 0.0 < self.full_compliance_beta <= 1.0:
            raise ValueError("full_compliance_beta must be in (0, 1]")


@dataclass(frozen=True)
class StepMetricResult:
    """Detached ordered step values and validity masks, all shaped [T,N]."""

    values: tuple[tuple[str, torch.Tensor], ...]
    masks: tuple[tuple[str, torch.Tensor], ...]
    time_steps: int
    environments: int

    def value(self, name: str) -> torch.Tensor:
        return dict(self.values)[name]

    def mask(self, name: str) -> torch.Tensor:
        return dict(self.masks)[name]


@dataclass(frozen=True)
class MetricResult:
    """A finite scalar or an explicit unavailable result."""

    name: str
    reduction: str
    unit: str
    available: bool
    value: float | None
    numerator: float | None
    denominator: int
    valid_count: int
    total_count: int


@dataclass(frozen=True)
class AggregateMetricResult:
    metrics: tuple[MetricResult, ...]
    time_steps: int
    environments: int

    def metric(self, name: str) -> MetricResult:
        return {metric.name: metric for metric in self.metrics}[name]


def _check_no_aliases(inputs: Mapping[str, torch.Tensor]) -> None:
    storage_owners: dict[tuple[str, int], str] = {}
    for name, value in inputs.items():
        key = (value.device.type + str(value.device.index), value.untyped_storage().data_ptr())
        if key in storage_owners:
            raise ValueError(f"evaluation tensors {storage_owners[key]} and {name} alias storage")
        storage_owners[key] = name


def validate_evaluation_inputs(inputs: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    """Validate the exact ordered evaluation-only [T,N] mapping, failing closed."""
    if not isinstance(inputs, Mapping):
        raise TypeError("evaluation branch must be a mapping")
    if tuple(inputs.keys()) != EVALUATION_INPUT_KEYS:
        raise ValueError(f"evaluation keys must be exactly ordered as {EVALUATION_INPUT_KEYS}")
    if not all(isinstance(value, torch.Tensor) for value in inputs.values()):
        raise TypeError("every evaluation value must be a torch.Tensor")

    command = inputs["commanded_velocity_b_m_s"]
    if command.ndim != 3 or command.shape[0] <= 0 or command.shape[1] <= 0 or command.shape[2] != 3:
        raise ValueError("commanded_velocity_b_m_s must have shape [T,N,3] with T,N positive")
    if command.dtype not in (torch.float32, torch.float64):
        raise TypeError("evaluation floating tensors must use float32 or float64")
    time_steps, environments = command.shape[:2]
    vector_keys = EVALUATION_INPUT_KEYS[:5]
    scalar_float_keys = (
        "anchor_error_m", "minimum_clearance_m", "ics_beta", "ics_worst_margin_m_s",
        "observability_value",
    )
    bool_keys = (
        "anchor_valid", "clearance_valid", "collision", "ics_emergency", "terminated",
        "truncated", "termination_collision", "termination_below_height",
        "termination_above_height", "termination_timeout", "observability_valid",
    )
    long_keys = ("termination_reason", "observability_source")
    for name in vector_keys:
        value = inputs[name]
        if value.shape != (time_steps, environments, 3):
            raise ValueError(f"{name} must have shape [T,N,3]")
        if value.dtype != command.dtype or value.device != command.device:
            raise TypeError(f"{name} must share command dtype and device")
    for name in scalar_float_keys:
        value = inputs[name]
        if value.shape != (time_steps, environments):
            raise ValueError(f"{name} must have shape [T,N]")
        if value.dtype != command.dtype or value.device != command.device:
            raise TypeError(f"{name} must share command dtype and device")
    for name in bool_keys:
        value = inputs[name]
        if value.shape != (time_steps, environments) or value.dtype != torch.bool:
            raise TypeError(f"{name} must be bool with shape [T,N]")
        if value.device != command.device:
            raise ValueError(f"{name} must share command device")
    for name in long_keys:
        value = inputs[name]
        if value.shape != (time_steps, environments) or value.dtype != torch.long:
            raise TypeError(f"{name} must be torch.long with shape [T,N]")
        if value.device != command.device:
            raise ValueError(f"{name} must share command device")
    if any(not bool(torch.isfinite(inputs[name]).all().item()) for name in (*vector_keys, *scalar_float_keys)):
        raise ValueError("evaluation floating tensors must be finite")
    _check_no_aliases(inputs)

    if bool((inputs["anchor_error_m"] < 0.0).any().item()):
        raise ValueError("anchor_error_m must be nonnegative")
    if bool((inputs["minimum_clearance_m"] < 0.0).any().item()):
        raise ValueError("minimum_clearance_m must be nonnegative")
    beta = inputs["ics_beta"]
    if bool(((beta < 0.0) | (beta > 1.0)).any().item()):
        raise ValueError("ics_beta must be in [0,1]")
    if bool((inputs["ics_emergency"] & (beta != 0.0)).any().item()):
        raise ValueError("ICS emergency requires exact zero beta")

    collision = inputs["collision"]
    collision_mask = inputs["termination_collision"]
    below = inputs["termination_below_height"]
    above = inputs["termination_above_height"]
    timeout = inputs["termination_timeout"]
    if not torch.equal(collision, collision_mask):
        raise ValueError("collision and termination_collision must be identical")
    if not torch.equal(inputs["terminated"], collision_mask | below | above):
        raise ValueError("terminated must preserve independent M6 masks")
    if not torch.equal(inputs["truncated"], timeout):
        raise ValueError("truncated must equal the independent timeout mask")
    reason = torch.full_like(inputs["termination_reason"], TERMINATION_NONE)
    reason = torch.where(timeout, torch.full_like(reason, TERMINATION_TIMEOUT), reason)
    reason = torch.where(above, torch.full_like(reason, TERMINATION_ABOVE_HEIGHT), reason)
    reason = torch.where(below, torch.full_like(reason, TERMINATION_BELOW_HEIGHT), reason)
    reason = torch.where(collision_mask, torch.full_like(reason, TERMINATION_COLLISION), reason)
    if not torch.equal(inputs["termination_reason"], reason):
        raise ValueError("termination_reason violates accepted M6 precedence")

    source = inputs["observability_source"]
    allowed = (source == int(ObservabilitySource.UNAVAILABLE)) | (source == int(ObservabilitySource.PROXY)) | (source == int(ObservabilitySource.EXACT))
    if not bool(allowed.all().item()):
        raise ValueError("observability_source must be UNAVAILABLE, PROXY, or EXACT")
    if not torch.equal(inputs["observability_valid"], source != int(ObservabilitySource.UNAVAILABLE)):
        raise ValueError("observability validity must agree exactly with source classification")
    return time_steps, environments


def compute_step_metrics(
    inputs: Mapping[str, torch.Tensor],
    config: EvaluationMetricConfig = EvaluationMetricConfig(),
) -> StepMetricResult:
    """Compute detached step values without reducing, repairing, or padding samples."""
    if not isinstance(config, EvaluationMetricConfig):
        raise TypeError("config must be EvaluationMetricConfig")
    time_steps, environments = validate_evaluation_inputs(inputs)
    command = inputs["commanded_velocity_b_m_s"]
    actual = inputs["actual_velocity_b_m_s"]
    issued = inputs["issued_velocity_b_m_s"]
    command_norm = torch.linalg.vector_norm(command, dim=-1)
    null_command = command_norm <= config.null_command_threshold_m_s
    intervention = (inputs["ics_beta"] < config.full_compliance_beta) | inputs["ics_emergency"]
    preservation_valid = (~null_command) & (~intervention)
    preservation = torch.zeros_like(command_norm)
    preservation[preservation_valid] = (
        torch.linalg.vector_norm(issued[preservation_valid], dim=-1)
        / command_norm[preservation_valid]
    )
    values = {
        "tracking_squared_error_m2_s2": (actual - command).square().sum(dim=-1),
        "null_station_drift_m": torch.linalg.vector_norm(
            inputs["position_w_m"] - inputs["station_reference_position_w_m"], dim=-1
        ),
        "null_station_speed_m_s": torch.linalg.vector_norm(actual, dim=-1),
        "anchor_error_m": inputs["anchor_error_m"],
        "minimum_clearance_m": inputs["minimum_clearance_m"],
        "collision": inputs["collision"].to(command.dtype),
        "ics_violation": (inputs["ics_worst_margin_m_s"] < 0.0).to(command.dtype),
        "command_preservation_ratio": preservation,
        "intervention": intervention.to(command.dtype),
        "observability_value": inputs["observability_value"],
    }
    masks = {
        "all_samples": torch.ones((time_steps, environments), dtype=torch.bool, device=command.device),
        "null_command": null_command,
        "anchor_valid": inputs["anchor_valid"],
        "clearance_valid": inputs["clearance_valid"],
        "preservation_valid": preservation_valid,
        "observability_valid": inputs["observability_valid"],
    }
    if tuple(values) != STEP_VALUE_KEYS or tuple(masks) != STEP_MASK_KEYS:
        raise RuntimeError("internal step metric ordering violation")
    if any(not bool(torch.isfinite(value).all().item()) for value in values.values()):
        raise FloatingPointError("step metric computation produced a non-finite value")
    return StepMetricResult(
        tuple((name, value.detach().clone()) for name, value in values.items()),
        tuple((name, value.detach().clone()) for name, value in masks.items()),
        time_steps,
        environments,
    )


def _reduce(
    name: str,
    values: torch.Tensor,
    mask: torch.Tensor,
    reduction: str,
    unit: str,
    total_count: int,
) -> MetricResult:
    selected = values[mask]
    count = int(selected.numel())
    if count == 0:
        return MetricResult(name, reduction, unit, False, None, None, 0, 0, total_count)
    if reduction == "minimum":
        numerator = float(selected.min().item())
        denominator = 1
        value = numerator
    else:
        numerator = float(selected.sum().item())
        denominator = count
        mean = numerator / denominator
        value = math.sqrt(mean) if reduction == "root_mean_square" else mean
    if not math.isfinite(numerator) or not math.isfinite(value):
        raise FloatingPointError(f"aggregate {name} is non-finite")
    return MetricResult(name, reduction, unit, True, value, numerator, denominator, count, total_count)


def aggregate_metrics(step: StepMetricResult) -> AggregateMetricResult:
    """Reduce every [T,N] sample exactly once with explicit conditional counts."""
    if not isinstance(step, StepMetricResult):
        raise TypeError("step must be StepMetricResult")
    if tuple(name for name, _ in step.values) != STEP_VALUE_KEYS or tuple(name for name, _ in step.masks) != STEP_MASK_KEYS:
        raise ValueError("step metric result has invalid ordering")
    total = step.time_steps * step.environments
    values, masks = dict(step.values), dict(step.masks)
    specifications = (
        ("tracking_rmse_m_s", "tracking_squared_error_m2_s2", "all_samples", "root_mean_square", "m/s"),
        ("null_station_drift_mean_m", "null_station_drift_m", "null_command", "mean", "m"),
        ("null_station_speed_mean_m_s", "null_station_speed_m_s", "null_command", "mean", "m/s"),
        ("anchor_error_mean_m", "anchor_error_m", "anchor_valid", "mean", "m"),
        ("minimum_clearance_m", "minimum_clearance_m", "clearance_valid", "minimum", "m"),
        ("collision_rate", "collision", "all_samples", "mean", "fraction"),
        ("ics_violation_rate", "ics_violation", "all_samples", "mean", "fraction"),
        ("command_preservation_ratio", "command_preservation_ratio", "preservation_valid", "mean", "ratio"),
        ("intervention_frequency", "intervention", "all_samples", "mean", "fraction"),
        ("observability_mean", "observability_value", "observability_valid", "mean", "source-defined"),
    )
    metrics = tuple(
        _reduce(name, values[value_name], masks[mask_name], reduction, unit, total)
        for name, value_name, mask_name, reduction, unit in specifications
    )
    if tuple(metric.name for metric in metrics) != METRIC_KEYS:
        raise RuntimeError("internal aggregate metric ordering violation")
    return AggregateMetricResult(metrics, step.time_steps, step.environments)


__all__ = [
    "AggregateMetricResult", "EVALUATION_INPUT_KEYS", "EvaluationMetricConfig", "METRIC_KEYS",
    "MetricResult", "ObservabilitySource", "STEP_MASK_KEYS", "STEP_VALUE_KEYS", "StepMetricResult",
    "aggregate_metrics", "compute_step_metrics", "validate_evaluation_inputs",
]
