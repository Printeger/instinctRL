from __future__ import annotations

from dataclasses import replace

import torch

from training_workloads.unified_h1.evaluation import continuation_decision, evaluate_checkpoint
from test_unified_h1_a0005 import _traces


def _flagged(traces, index: int, field: str):
    output = list(traces)
    mask = getattr(output[index], field).clone()
    mask[-1] = True
    output[index] = replace(output[index], **{field: mask})
    return output


def _r4_nodes():
    baseline = evaluate_checkpoint(_traces(baseline=True), update=0)
    u64 = evaluate_checkpoint(_traces(baseline=True), update=64)
    latest = _traces(one_behavior_failure=True)
    previous = evaluate_checkpoint(latest, update=192)
    current = evaluate_checkpoint(latest, update=256)
    return baseline, u64, previous, current


def test_r4_early_safety_failures_are_retained_but_only_latest_nodes_gate_safety():
    baseline, u64, previous, current = _r4_nodes()
    early_collision = evaluate_checkpoint(_flagged(_traces(baseline=True), 0, "collision"), update=0)
    early_height = evaluate_checkpoint(_flagged(_traces(baseline=True), 32, "above_height"), update=64)

    assert early_collision["collisions"] == 1 and early_collision["invalid_evidence_episodes"] == 0
    assert early_height["height_failures"] == 1 and early_height["invalid_evidence_episodes"] == 0
    assert early_collision["failures"] == baseline["failures"]
    assert early_collision["error_components"] == baseline["error_components"]
    assert early_height["failures"] == u64["failures"]
    assert early_height["error_components"] == u64["error_components"]
    assert continuation_decision(baseline=early_collision, current=current,
                                 previous=previous, u64=u64)["extend"] is True
    assert continuation_decision(baseline=baseline, current=current,
                                 previous=previous, u64=early_height)["extend"] is True


def test_r4_latest_safety_failures_and_invalid_evidence_at_any_node_are_rejected():
    baseline, u64, previous, current = _r4_nodes()
    latest_collision = evaluate_checkpoint(
        _flagged(_traces(one_behavior_failure=True), 0, "collision"), update=192)
    latest_height = evaluate_checkpoint(
        _flagged(_traces(one_behavior_failure=True), 32, "above_height"), update=256)
    assert not continuation_decision(baseline=baseline, current=current,
                                     previous=latest_collision, u64=u64)["extend"]
    assert not continuation_decision(baseline=baseline, current=latest_height,
                                     previous=previous, u64=u64)["extend"]

    valid_nodes = [baseline, u64, previous, current]
    updates = (0, 64, 192, 256)
    source_kinds = (True, True, False, False)
    for invalid_index, (update, use_baseline) in enumerate(zip(updates, source_kinds, strict=True)):
        invalid = evaluate_checkpoint(_flagged(_traces(baseline=use_baseline,
                                                       one_behavior_failure=not use_baseline), 0, "missing"),
                                      update=update)
        assert invalid["invalid_evidence_episodes"] == 1
        nodes = list(valid_nodes); nodes[invalid_index] = invalid
        assert not continuation_decision(baseline=nodes[0], current=nodes[3],
                                         previous=nodes[2], u64=nodes[1])["extend"]


def test_r5_early_safety_failure_is_allowed_but_latest_safety_failure_is_not():
    baseline = evaluate_checkpoint(_flagged(_traces(baseline=True), 0, "collision"), update=256)
    previous = evaluate_checkpoint(_traces(), update=448)
    current = evaluate_checkpoint(_traces(), update=512)
    assert continuation_decision(baseline=baseline, current=current,
                                 previous=previous, stage="R5")["extend"] is True

    latest_height = evaluate_checkpoint(_flagged(_traces(), 32, "above_height"), update=448)
    assert continuation_decision(baseline=baseline, current=current,
                                 previous=latest_height, stage="R5")["extend"] is False
