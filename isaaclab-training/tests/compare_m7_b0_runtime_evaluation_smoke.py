#!/usr/bin/env python3
"""Exact complete comparator for two M7-012 B0 runtime evaluation bundles."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
from typing import Any

import torch


ROOT_KEYS = (
    "contract", "command_schedule", "environment_receipt", "evaluation_inputs",
    "observability_details", "command_identity_sha256", "source_identity_sha256",
    "result_identity_sha256", "metric_oracle", "canonical_record", "record_mapping",
    "call_counts", "retention", "gates",
)


def _compare(left: Any, right: Any, path: str, report: dict[str, Any]) -> None:
    if type(left) is not type(right):
        raise AssertionError(f"type mismatch at {path}")
    if isinstance(left, dict):
        if list(left) != list(right):
            raise AssertionError(f"missing, extra, or reordered keys at {path}")
        for key in left:
            _compare(left[key], right[key], f"{path}.{key}", report)
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f"length mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare(left_item, right_item, f"{path}[{index}]", report)
    elif isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"tensor metadata mismatch at {path}")
        if left.is_floating_point() and (
            not bool(torch.isfinite(left).all().item())
            or not bool(torch.isfinite(right).all().item())
        ):
            raise AssertionError(f"nonfinite tensor at {path}")
        if not torch.equal(left, right):
            raise AssertionError(f"tensor mismatch at {path}")
        report["tensor_count"] += 1
        report["element_count"] += left.numel()
    elif isinstance(left, float):
        if not math.isfinite(left) or not math.isfinite(right) or left != right:
            raise AssertionError(f"float mismatch at {path}")
    elif left != right:
        raise AssertionError(f"value mismatch at {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = torch.load(args.first, map_location="cpu", weights_only=True)
    second = torch.load(args.second, map_location="cpu", weights_only=True)
    if type(first) is not OrderedDict or tuple(first) != ROOT_KEYS:
        raise AssertionError("first bundle root keys are missing, extra, or reordered")
    if type(second) is not OrderedDict or tuple(second) != ROOT_KEYS:
        raise AssertionError("second bundle root keys are missing, extra, or reordered")
    report = {
        "status": "PASS", "tensor_count": 0, "element_count": 0,
        "max_absolute_error": 0.0,
    }
    _compare(first, second, "root", report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
