#!/usr/bin/env python3
"""Compare two normalized M2 smoke results at the fixed absolute tolerance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TOLERANCE = 1.0e-5


def _compare(left: Any, right: Any, path: str, errors: list[str], maxima: list[float]) -> None:
    if isinstance(left, bool) or isinstance(right, bool):
        if left is not right:
            errors.append(f"{path}: boolean mismatch {left!r} != {right!r}")
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            errors.append(f"{path}: non-finite numeric value")
        else:
            error = abs(float(left) - float(right))
            maxima.append(error)
            if error > TOLERANCE:
                errors.append(f"{path}: absolute error {error} exceeds {TOLERANCE}")
    elif isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            errors.append(f"{path}: key mismatch")
            return
        for key in sorted(left):
            _compare(left[key], right[key], f"{path}.{key}", errors, maxima)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            errors.append(f"{path}: length mismatch {len(left)} != {len(right)}")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare(left_item, right_item, f"{path}[{index}]", errors, maxima)
    elif left != right:
        errors.append(f"{path}: discrete mismatch {left!r} != {right!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = json.loads(args.first.read_text(encoding="utf-8"))
    second = json.loads(args.second.read_text(encoding="utf-8"))
    errors: list[str] = []
    maxima: list[float] = []
    _compare(first, second, "$", errors, maxima)
    result = {"schema_version": "m2-smoke-comparison-1", "tolerance": TOLERANCE,
              "max_absolute_numeric_error": max(maxima, default=0.0),
              "numeric_values_compared": len(maxima), "errors": errors,
              "status": "PASS" if not errors else "FAIL"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
