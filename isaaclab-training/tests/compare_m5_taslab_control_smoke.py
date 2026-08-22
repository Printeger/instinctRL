"""Complete ordered finite comparison for independent M5 runtime traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

TOLERANCE = 1.0e-5


def _compare(left: Any, right: Any, path: str, report: dict[str, Any]) -> None:
    if type(left) is not type(right):
        raise AssertionError(f"type mismatch at {path}")
    if isinstance(left, dict):
        if set(left) != set(right) or list(left) != list(right):
            raise AssertionError(f"missing, extra, or reordered keys at {path}")
        for key in left:
            _compare(left[key], right[key], f"{path}.{key}", report)
        return
    if isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"tensor metadata mismatch at {path}")
        if left.is_floating_point():
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise AssertionError(f"non-finite tensor at {path}")
            error = float(torch.abs(left - right).max().item()) if left.numel() else 0.0
            if error > TOLERANCE:
                raise AssertionError(f"tolerance exceeded at {path}: {error}")
            report["max_absolute_error"] = max(report["max_absolute_error"], error)
        elif not torch.equal(left, right):
            raise AssertionError(f"exact tensor mismatch at {path}")
        report["tensor_count"] += 1
        report["element_count"] += left.numel()
        return
    if isinstance(left, float):
        if not (math.isfinite(left) and math.isfinite(right)):
            raise AssertionError(f"non-finite scalar at {path}")
        error = abs(left - right)
        if error > TOLERANCE:
            raise AssertionError(f"scalar tolerance exceeded at {path}")
        report["max_absolute_error"] = max(report["max_absolute_error"], error)
        return
    if left != right:
        raise AssertionError(f"value mismatch at {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = torch.load(args.first.with_suffix(".pt"), map_location="cpu", weights_only=True)
    second = torch.load(args.second.with_suffix(".pt"), map_location="cpu", weights_only=True)
    report = {"status": "PASS", "tolerance": TOLERANCE, "tensor_count": 0, "element_count": 0, "max_absolute_error": 0.0}
    _compare(first, second, "root", report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
