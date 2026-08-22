"""M7 two-branch actor audit and deterministic evaluation record."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

import torch

from instinctlab.observations.audit import audit_actor_observation
from instinctlab.task.termination import (
    TERMINATION_ABOVE_HEIGHT,
    TERMINATION_BELOW_HEIGHT,
    TERMINATION_COLLISION,
    TERMINATION_NONE,
    TERMINATION_TIMEOUT,
)
from .metrics import (
    EVALUATION_INPUT_KEYS,
    AggregateMetricResult,
    EvaluationMetricConfig,
    ObservabilitySource,
    aggregate_metrics,
    compute_step_metrics,
)

BRANCH_KEYS = ("policy", "evaluation")
PROVENANCE_KEYS = ("product_commit", "fixture_id", "generator_id", "seed")
PLATFORM_ID = "TASLAB_UAV"
SENSOR_ID = "Livox MID360"


@dataclass(frozen=True)
class EvaluationRecord:
    """Owned canonical JSON; parsing returns a fresh mutable copy."""

    canonical_json: str

    def to_mapping(self) -> dict[str, object]:
        return json.loads(self.canonical_json)


def _tensor_sha256(value: torch.Tensor) -> str:
    canonical = value.detach().contiguous().cpu().numpy().tobytes(order="C")
    return hashlib.sha256(canonical).hexdigest()


def _validate_provenance(provenance: Mapping[str, object]) -> None:
    if not isinstance(provenance, Mapping) or tuple(provenance.keys()) != PROVENANCE_KEYS:
        raise ValueError(f"provenance keys must be exactly ordered as {PROVENANCE_KEYS}")
    commit = provenance["product_commit"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("product_commit must be a lowercase full 40-character Git SHA")
    for name in ("fixture_id", "generator_id"):
        value = provenance[name]
        if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
            raise ValueError(f"{name} must be a nonempty printable string")
    seed = provenance["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _reject_cross_branch_storage_aliases(
    policy: Mapping[str, torch.Tensor], evaluation: object
) -> None:
    """Reject evaluation tensors backed by either audited policy storage."""
    if not isinstance(evaluation, Mapping):
        return
    policy_storage = {
        (value.device, value.untyped_storage().data_ptr()): name
        for name, value in policy.items()
    }
    for name, value in evaluation.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = (value.device, value.untyped_storage().data_ptr())
        if key in policy_storage:
            raise ValueError(
                f"policy tensor {policy_storage[key]} and evaluation tensor {name} alias storage"
            )


def _metric_record(aggregate: AggregateMetricResult) -> OrderedDict[str, object]:
    output: OrderedDict[str, object] = OrderedDict()
    for metric in aggregate.metrics:
        output[metric.name] = OrderedDict(
            (
                ("reduction", metric.reduction),
                ("unit", metric.unit),
                ("available", metric.available),
                ("value", metric.value),
                ("numerator", metric.numerator),
                ("denominator", metric.denominator),
                ("valid_count", metric.valid_count),
                ("total_count", metric.total_count),
            )
        )
    return output


def _termination_counts(evaluation: Mapping[str, torch.Tensor]) -> OrderedDict[str, int]:
    terminated = evaluation["terminated"]
    truncated = evaluation["truncated"]
    done = terminated | truncated
    reason = evaluation["termination_reason"]
    return OrderedDict(
        (
            ("collision", int(evaluation["termination_collision"].sum().item())),
            ("below_height", int(evaluation["termination_below_height"].sum().item())),
            ("above_height", int(evaluation["termination_above_height"].sum().item())),
            ("timeout", int(evaluation["termination_timeout"].sum().item())),
            ("terminated", int(terminated.sum().item())),
            ("truncated", int(truncated.sum().item())),
            ("done", int(done.sum().item())),
            ("simultaneous_terminated_truncated", int((terminated & truncated).sum().item())),
            ("reason_none", int((reason == TERMINATION_NONE).sum().item())),
            ("reason_timeout", int((reason == TERMINATION_TIMEOUT).sum().item())),
            ("reason_above_height", int((reason == TERMINATION_ABOVE_HEIGHT).sum().item())),
            ("reason_below_height", int((reason == TERMINATION_BELOW_HEIGHT).sum().item())),
            ("reason_collision", int((reason == TERMINATION_COLLISION).sum().item())),
        )
    )


def build_evaluation_record(
    branches: Mapping[str, Mapping[str, torch.Tensor]],
    provenance: Mapping[str, object],
    config: EvaluationMetricConfig = EvaluationMetricConfig(),
) -> EvaluationRecord:
    """Audit policy first, then consume isolated privilege into canonical JSON."""
    if not isinstance(config, EvaluationMetricConfig):
        raise TypeError("config must be EvaluationMetricConfig")
    if not isinstance(branches, Mapping) or tuple(branches.keys()) != BRANCH_KEYS:
        raise ValueError(f"public branches must be exactly ordered as {BRANCH_KEYS}")
    policy = branches["policy"]
    evaluation = branches["evaluation"]
    audit_actor_observation(policy)
    _reject_cross_branch_storage_aliases(policy, evaluation)
    _validate_provenance(provenance)
    step = compute_step_metrics(evaluation, config)
    if policy["lidar_grid"].shape[0] != step.environments:
        raise ValueError("policy and evaluation environment dimensions must match")
    aggregate = aggregate_metrics(step)

    actor_manifest = OrderedDict(
        (
            ("status", "PASS"),
            ("keys", list(policy.keys())),
            ("lidar_grid_shape", list(policy["lidar_grid"].shape)),
            ("state_vec_shape", list(policy["state_vec"].shape)),
            ("dtype", str(policy["lidar_grid"].dtype)),
            ("device", str(policy["lidar_grid"].device)),
            ("lidar_grid_sha256", _tensor_sha256(policy["lidar_grid"])),
            ("state_vec_sha256", _tensor_sha256(policy["state_vec"])),
        )
    )
    source = evaluation["observability_source"]
    source_counts = OrderedDict(
        (label.name, int((source == int(label)).sum().item()))
        for label in (ObservabilitySource.UNAVAILABLE, ObservabilitySource.PROXY, ObservabilitySource.EXACT)
    )
    config_manifest = OrderedDict(asdict(config).items())
    record = OrderedDict(
        (
            ("schema_version", config.schema_version),
            ("platform", PLATFORM_ID),
            ("sensor", SENSOR_ID),
            ("claim", "pure-tensor contract record; no simulator evaluation or policy-performance claim"),
            ("actor_audit", actor_manifest),
            ("input_dimensions", OrderedDict((("time_steps", step.time_steps), ("environments", step.environments), ("samples", step.time_steps * step.environments)))),
            ("evaluation_input_order", list(EVALUATION_INPUT_KEYS)),
            ("metric_config", config_manifest),
            ("observability_source_counts", source_counts),
            ("metrics", _metric_record(aggregate)),
            ("termination_counts", _termination_counts(evaluation)),
            ("provenance", OrderedDict(provenance.items())),
        )
    )
    serialized = json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    decoded = json.loads(serialized)
    if not isinstance(decoded, dict) or json.dumps(decoded, ensure_ascii=True, allow_nan=False, separators=(",", ":")) != serialized:
        raise RuntimeError("evaluation record failed exact canonical JSON round trip")
    for metric in aggregate.metrics:
        for value in (metric.value, metric.numerator):
            if value is not None and not math.isfinite(value):
                raise FloatingPointError("evaluation record contains a non-finite metric")
    return EvaluationRecord(serialized)


__all__ = [
    "BRANCH_KEYS", "EvaluationRecord", "PLATFORM_ID", "PROVENANCE_KEYS", "SENSOR_ID",
    "build_evaluation_record",
]
