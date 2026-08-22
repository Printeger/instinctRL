"""Hand-computed, negative, ownership, serialization, and isolation M7 tests."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import FrozenInstanceError
import json
import math

import pytest
import torch

from instinctlab.evaluation import (
    BRANCH_KEYS,
    EVALUATION_INPUT_KEYS,
    METRIC_KEYS,
    EvaluationMetricConfig,
    ObservabilitySource,
    aggregate_metrics,
    build_evaluation_record,
    compute_step_metrics,
    validate_evaluation_inputs,
)
from instinctlab.observations.audit import ActorObservationAuditError
from instinctlab.task.termination import (
    TERMINATION_ABOVE_HEIGHT,
    TERMINATION_BELOW_HEIGHT,
    TERMINATION_COLLISION,
    TERMINATION_NONE,
)


def _actor(batch: int = 2, dtype: torch.dtype = torch.float64, device: str = "cpu"):
    lidar = torch.empty(batch, 12, 59, 360, dtype=dtype, device=device)
    lidar[:, 0::3] = 40.0
    lidar[:, 1::3] = 1.0
    lidar[:, 2::3] = 1.0
    state = torch.zeros(batch, 52, dtype=dtype, device=device)
    for offset in range(0, 52, 13):
        state[:, offset + 5] = 1.0
    state[:, -7] = torch.arange(batch, dtype=dtype, device=device)
    return OrderedDict((("lidar_grid", lidar), ("state_vec", state)))


def _evaluation(dtype: torch.dtype = torch.float64, device: str = "cpu"):
    def f(value):
        return torch.tensor(value, dtype=dtype, device=device)

    def b(value):
        return torch.tensor(value, dtype=torch.bool, device=device)

    def i(value):
        return torch.tensor(value, dtype=torch.long, device=device)

    values = OrderedDict(
        (
            ("commanded_velocity_b_m_s", f([[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [2, 0, 0]]])),
            ("actual_velocity_b_m_s", f([[[1, 0, 0], [2, 0, 0]], [[0, 0, 0], [0, 0, 0]]])),
            ("issued_velocity_b_m_s", f([[[0, 0, 0], [0.8, 0, 0]], [[0, 0, 0], [2.2, 0, 0]]])),
            ("position_w_m", f([[[1, 0, 0], [8, 0, 0]], [[3, 0, 0], [9, 0, 0]]])),
            ("station_reference_position_w_m", f([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])),
            ("anchor_error_m", f([[0.1, 0.2], [0.3, 0.4]])),
            ("anchor_valid", b([[True, False], [True, False]])),
            ("minimum_clearance_m", f([[2.0, 1.0], [0.5, 3.0]])),
            ("clearance_valid", b([[True, True], [True, False]])),
            ("collision", b([[False, True], [False, False]])),
            ("ics_beta", f([[1.0, 1.0], [0.5, 1.0]])),
            ("ics_emergency", b([[False, False], [False, False]])),
            ("ics_worst_margin_m_s", f([[1.0, -1.0], [0.0, -2.0]])),
            ("terminated", b([[False, True], [True, True]])),
            ("truncated", b([[False, True], [True, False]])),
            ("termination_reason", i([[TERMINATION_NONE, TERMINATION_COLLISION], [TERMINATION_BELOW_HEIGHT, TERMINATION_ABOVE_HEIGHT]])),
            ("termination_collision", b([[False, True], [False, False]])),
            ("termination_below_height", b([[False, False], [True, False]])),
            ("termination_above_height", b([[False, False], [False, True]])),
            ("termination_timeout", b([[False, True], [True, False]])),
            ("observability_value", f([[0.0, 2.0], [4.0, 6.0]])),
            ("observability_valid", b([[False, True], [True, True]])),
            ("observability_source", i([[ObservabilitySource.UNAVAILABLE, ObservabilitySource.PROXY], [ObservabilitySource.EXACT, ObservabilitySource.EXACT]])),
        )
    )
    assert tuple(values) == EVALUATION_INPUT_KEYS
    return values


def _provenance():
    return OrderedDict(
        (("product_commit", "0" * 40), ("fixture_id", "hand-table-v1"), ("generator_id", "pytest"), ("seed", 7))
    )


def _branches(evaluation=None, actor=None):
    return OrderedDict((("policy", _actor() if actor is None else actor), ("evaluation", _evaluation() if evaluation is None else evaluation)))


def test_exact_hand_computed_multistep_metrics_counts_and_units():
    step = compute_step_metrics(_evaluation())
    aggregate = aggregate_metrics(step)
    assert (step.time_steps, step.environments) == (2, 2)
    assert tuple(metric.name for metric in aggregate.metrics) == METRIC_KEYS
    expected = {
        "tracking_rmse_m_s": math.sqrt(1.5),
        "null_station_drift_mean_m": 2.0,
        "null_station_speed_mean_m_s": 0.5,
        "anchor_error_mean_m": 0.2,
        "minimum_clearance_m": 0.5,
        "collision_rate": 0.25,
        "ics_violation_rate": 0.5,
        "command_preservation_ratio": 0.95,
        "intervention_frequency": 0.25,
        "observability_mean": 4.0,
    }
    for name, value in expected.items():
        metric = aggregate.metric(name)
        assert metric.available and math.isclose(metric.value, value, rel_tol=1e-12)
        assert metric.total_count == 4 and metric.valid_count > 0
    assert aggregate.metric("tracking_rmse_m_s").numerator == 6.0
    assert aggregate.metric("tracking_rmse_m_s").denominator == 4
    assert aggregate.metric("minimum_clearance_m").numerator == 0.5
    assert aggregate.metric("minimum_clearance_m").denominator == 1
    assert aggregate.metric("minimum_clearance_m").valid_count == 3
    assert aggregate.metric("command_preservation_ratio").valid_count == 2


def test_single_sample_and_zero_one_mixed_conditional_denominators():
    values = _evaluation()
    values["anchor_valid"][:] = False
    values["clearance_valid"][:] = False
    values["observability_source"][:] = int(ObservabilitySource.UNAVAILABLE)
    values["observability_valid"][:] = False
    values["ics_beta"][:, 1] = 0.5
    aggregate = aggregate_metrics(compute_step_metrics(values))
    for name in ("anchor_error_mean_m", "minimum_clearance_m", "command_preservation_ratio", "observability_mean"):
        metric = aggregate.metric(name)
        assert not metric.available and metric.value is None and metric.numerator is None
        assert metric.denominator == 0 and metric.valid_count == 0
    one = _evaluation()
    one["anchor_valid"][:] = False
    one["anchor_valid"][0, 0] = True
    metric = aggregate_metrics(compute_step_metrics(one)).metric("anchor_error_mean_m")
    assert metric.available and metric.value == 0.1 and metric.denominator == metric.valid_count == 1
    mixed = aggregate_metrics(compute_step_metrics(_evaluation())).metric("anchor_error_mean_m")
    assert mixed.denominator == mixed.valid_count == 2


def test_termination_counts_preserve_independent_simultaneous_masks_and_precedence():
    record = build_evaluation_record(_branches(), _provenance()).to_mapping()
    counts = record["termination_counts"]
    assert counts == {
        "collision": 1, "below_height": 1, "above_height": 1, "timeout": 2,
        "terminated": 3, "truncated": 2, "done": 3,
        "simultaneous_terminated_truncated": 2, "reason_none": 1, "reason_timeout": 0,
        "reason_above_height": 1, "reason_below_height": 1, "reason_collision": 1,
    }


def test_record_order_identity_source_labels_and_exact_json_round_trip():
    record = build_evaluation_record(_branches(), _provenance())
    decoded = record.to_mapping()
    assert tuple(decoded) == (
        "schema_version", "platform", "sensor", "claim", "actor_audit", "input_dimensions",
        "evaluation_input_order", "metric_config", "observability_source_counts", "metrics",
        "termination_counts", "provenance",
    )
    assert decoded["platform"] == "TASLAB_UAV" and decoded["sensor"] == "Livox MID360"
    assert decoded["actor_audit"]["status"] == "PASS"
    assert decoded["actor_audit"]["keys"] == ["lidar_grid", "state_vec"]
    assert decoded["observability_source_counts"] == {"UNAVAILABLE": 1, "PROXY": 1, "EXACT": 2}
    assert json.dumps(decoded, ensure_ascii=True, allow_nan=False, separators=(",", ":")) == record.canonical_json
    assert build_evaluation_record(_branches(), _provenance()).canonical_json == record.canonical_json


def test_every_source_is_detached_owned_and_input_mutation_cannot_change_results():
    actor, evaluation = _actor(), _evaluation()
    step = compute_step_metrics(evaluation)
    record = build_evaluation_record(_branches(evaluation, actor), _provenance())
    step_snapshot = tuple((name, value.clone()) for name, value in step.values)
    serialized = record.canonical_json
    for value in actor.values():
        value.add_(1.0)
    for value in evaluation.values():
        if value.dtype == torch.bool:
            value.logical_not_()
        else:
            value.add_(1)
    assert record.canonical_json == serialized
    for (name, before), (actual_name, actual) in zip(step_snapshot, step.values, strict=True):
        assert name == actual_name and torch.equal(before, actual)
    config = EvaluationMetricConfig()
    with pytest.raises(FrozenInstanceError):
        config.full_compliance_beta = 0.5
    with pytest.raises(FrozenInstanceError):
        record.canonical_json = "changed"


def test_evaluation_privilege_cannot_change_actor_output_but_actor_input_discriminates():
    actor = _actor()
    actor_output = actor["state_vec"][:, -7:-4].square().sum(dim=-1)
    first = build_evaluation_record(_branches(_evaluation(), actor), _provenance())
    changed = _evaluation()
    changed["actual_velocity_b_m_s"].add_(3.0)
    changed["position_w_m"].add_(4.0)
    changed["anchor_error_m"].add_(2.0)
    changed["minimum_clearance_m"].add_(1.0)
    changed["ics_worst_margin_m_s"].mul_(-1.0)
    changed["observability_value"].add_(8.0)
    second = build_evaluation_record(_branches(changed, actor), _provenance())
    second_actor_output = actor["state_vec"][:, -7:-4].square().sum(dim=-1)
    assert torch.equal(actor_output, second_actor_output)
    assert first.canonical_json != second.canonical_json
    actor_changed = _actor()
    actor_changed["state_vec"][0, -7] = 0.5
    changed_actor_output = actor_changed["state_vec"][:, -7:-4].square().sum(dim=-1)
    assert not torch.equal(actor_output, changed_actor_output)


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "alias", "shape", "dtype", "nonfinite", "domain", "inconsistent"])
def test_evaluation_mapping_fail_closed(mutation):
    values = _evaluation()
    if mutation == "missing":
        values.pop("anchor_error_m")
    elif mutation == "extra":
        values["extra"] = torch.zeros(2, 2)
    elif mutation == "reordered":
        values.move_to_end("commanded_velocity_b_m_s")
    elif mutation == "alias":
        values["actual_velocity_b_m_s"] = values["commanded_velocity_b_m_s"]
    elif mutation == "shape":
        values["anchor_error_m"] = torch.zeros(2, 2, 1, dtype=torch.float64)
    elif mutation == "dtype":
        values["anchor_error_m"] = values["anchor_error_m"].float()
    elif mutation == "nonfinite":
        values["anchor_error_m"][0, 0] = float("nan")
    elif mutation == "domain":
        values["ics_beta"][0, 0] = 1.1
    else:
        values["truncated"][0, 0] = True
    with pytest.raises((TypeError, ValueError)):
        validate_evaluation_inputs(values)


@pytest.mark.parametrize("mutation", ["branch_missing", "branch_extra", "branch_reordered", "batch", "actor_extra", "actor_forbidden"])
def test_public_boundary_and_m4_actor_negatives_remain_fail_closed(mutation):
    actor, evaluation = _actor(), _evaluation()
    branches = _branches(evaluation, actor)
    if mutation == "branch_missing":
        branches.pop("evaluation")
    elif mutation == "branch_extra":
        branches["logging"] = OrderedDict()
    elif mutation == "branch_reordered":
        branches.move_to_end("policy")
    elif mutation == "batch":
        branches["policy"] = _actor(batch=1)
    elif mutation == "actor_extra":
        actor["age"] = torch.zeros(2, 1)
    else:
        actor["privileged_actual_velocity"] = torch.zeros(2, 3)
    with pytest.raises((ActorObservationAuditError, TypeError, ValueError)):
        build_evaluation_record(branches, _provenance())


@pytest.mark.parametrize("mutation", ["source_domain", "source_validity", "emergency_beta", "collision_mismatch", "reason_precedence"])
def test_source_and_cross_field_consistency_fail_closed(mutation):
    values = _evaluation()
    if mutation == "source_domain":
        values["observability_source"][0, 0] = 3
    elif mutation == "source_validity":
        values["observability_valid"][0, 0] = True
    elif mutation == "emergency_beta":
        values["ics_emergency"][0, 0] = True
    elif mutation == "collision_mismatch":
        values["termination_collision"][0, 0] = True
    else:
        values["termination_reason"][0, 1] = TERMINATION_NONE
    with pytest.raises(ValueError):
        validate_evaluation_inputs(values)


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "commit", "fixture", "generator", "seed"])
def test_malformed_provenance_fails_closed(mutation):
    provenance = _provenance()
    if mutation == "missing": provenance.pop("fixture_id")
    elif mutation == "extra": provenance["extra"] = "x"
    elif mutation == "reordered": provenance.move_to_end("product_commit")
    elif mutation == "commit": provenance["product_commit"] = "ABC"
    elif mutation == "fixture": provenance["fixture_id"] = ""
    elif mutation == "generator": provenance["generator_id"] = "\n"
    else: provenance["seed"] = True
    with pytest.raises((TypeError, ValueError)):
        build_evaluation_record(_branches(), provenance)


def test_wrong_device_is_rejected_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    values = _evaluation()
    values["anchor_error_m"] = values["anchor_error_m"].cuda()
    with pytest.raises(TypeError):
        validate_evaluation_inputs(values)
