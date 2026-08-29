#!/usr/bin/env python3
"""Exact complete comparison for two independent M7-011 environment bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _compare(left: Any, right: Any, path: str, report: dict[str, int]) -> None:
    if type(left) is not type(right):
        raise AssertionError(f"type mismatch at {path}")
    if isinstance(left, dict):
        if list(left) != list(right):
            raise AssertionError(f"key/order mismatch at {path}")
        for key in left:
            _compare(left[key], right[key], f"{path}.{key}", report)
    elif isinstance(left, torch.Tensor):
        if left.shape != right.shape:
            raise AssertionError(f"shape mismatch at {path}")
        if left.dtype != right.dtype:
            raise AssertionError(f"dtype mismatch at {path}")
        if not torch.equal(left, right):
            raise AssertionError(f"exact tensor mismatch at {path}")
        report["tensor_count"] += 1
        report["element_count"] += left.numel()
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f"length mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare(left_item, right_item, f"{path}[{index}]", report)
    elif left != right:
        raise AssertionError(f"value mismatch at {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = torch.load(args.first.with_suffix(".pt"), map_location="cpu", weights_only=True)
    second = torch.load(args.second.with_suffix(".pt"), map_location="cpu", weights_only=True)
    expected_keys = (
        "contract", "command_schedule", "physics", "policies", "receipt",
        "coverage", "call_counts", "gates",
    )
    if tuple(first) != expected_keys or tuple(second) != expected_keys:
        raise AssertionError("bundle root keys are missing, extra, or reordered")
    if not first["gates"]["all_gates_pass"] or not second["gates"]["all_gates_pass"]:
        raise AssertionError("both process gate sets must pass")
    report = {"status": "PASS", "tensor_count": 0, "element_count": 0}
    _compare(first, second, "root", report)
    report["trajectory_sha256"] = first["receipt"]["trajectory_sha256"]
    report["policy_steps"] = first["coverage"]["policy_steps"]
    report["physics_steps"] = first["coverage"]["physics_steps"]
    report["environment_samples"] = first["coverage"]["environment_samples"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
