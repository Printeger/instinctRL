from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch

from training_workloads.unified_h1.evaluation import (
    continuation_decision,
    evaluate_checkpoint,
    select_checkpoint,
)
from test_unified_h1_rework import _trace


CASES = ("Z", "T", "S", "X", "W")


def _traces(*, baseline: bool = False, one_behavior_failure: bool = False):
    traces = []
    for case in CASES:
        for lane in range(32):
            trace = _trace(case, lane)
            if baseline and case == "Z":
                trace.velocity_b[:, 0] = .2
            if baseline and case == "T":
                trace.velocity_b.zero_()
            if one_behavior_failure and case == "Z" and lane == 0:
                trace.velocity_b[:, 0] = .09
            traces.append(trace)
    return traces


def test_actual_parser_missing_and_invalid_timing_fail_engineering_gate():
    missing = _traces()
    mask = torch.zeros_like(missing[0].missing); mask[7] = True
    missing[0] = replace(missing[0], missing=mask)
    missing_summary = evaluate_checkpoint(missing, update=256)
    assert missing_summary["episodes"][0]["reason"] == "MISSING_REQUIRED_SAMPLE"
    assert missing_summary["episodes"][0]["engineering"]["evidence_valid"] is False
    assert missing_summary["invalid_evidence_episodes"] == 1
    assert missing_summary["engineering_gates_pass"] is False

    invalid_timing = _traces()
    invalid_timing[32] = replace(invalid_timing[32], sample_policy_steps=(99,) + invalid_timing[32].sample_policy_steps[1:])
    timing_summary = evaluate_checkpoint(invalid_timing, update=256)
    assert timing_summary["episodes"][32]["reason"] == "CASE_TIMING_MISMATCH"
    assert timing_summary["invalid_evidence_episodes"] == 1
    assert timing_summary["engineering_gates_pass"] is False


def test_continuation_blocks_invalid_evidence_but_allows_valid_behavior_failure():
    baseline = evaluate_checkpoint(_traces(baseline=True), update=0)
    u64 = evaluate_checkpoint(_traces(baseline=True), update=64)
    current_traces = _traces(one_behavior_failure=True)
    previous = evaluate_checkpoint(current_traces, update=192)
    current = evaluate_checkpoint(current_traces, update=256)
    assert current["engineering_gates_pass"] is True
    assert current["all_gates_pass"] is False
    assert current["failures"]["Z"] == 1
    assert continuation_decision(baseline=baseline, current=current, previous=previous, u64=u64)["extend"] is True

    missing_traces = _traces()
    mask = torch.zeros_like(missing_traces[0].missing); mask[7] = True
    missing_traces[0] = replace(missing_traces[0], missing=mask)
    missing_current = evaluate_checkpoint(missing_traces, update=256)
    decision = continuation_decision(baseline=baseline, current=missing_current,
                                     previous=evaluate_checkpoint(_traces(), update=192), u64=u64)
    assert decision["extend"] is False

    r5_baseline = evaluate_checkpoint(_traces(baseline=True), update=256)
    r5_current = evaluate_checkpoint(_traces(), update=512)
    r5_previous = evaluate_checkpoint(_traces(), update=448)
    assert continuation_decision(baseline=r5_baseline, current=r5_current,
                                 previous=r5_previous, stage="R5")["extend"] is True
    missing_previous = evaluate_checkpoint(missing_traces, update=448)
    assert continuation_decision(baseline=r5_baseline, current=r5_current,
                                 previous=missing_previous, stage="R5")["extend"] is False


def test_checkpoint_selection_requires_consistent_complete_evidence_and_keeps_ordering():
    earlier = evaluate_checkpoint(_traces(), update=192)
    later = evaluate_checkpoint(_traces(), update=256)
    assert select_checkpoint([later, earlier]) is earlier

    minimal = {"schema_version": "metric-h1-v1.checkpoint.v1", "update": 64,
               "loss": 0., "all_gates_pass": True}
    inconsistent = deepcopy(later)
    inconsistent["invalid_evidence_episodes"] = 1
    assert select_checkpoint([minimal, inconsistent]) is None
