#!/usr/bin/env python3
"""Compare every normalized M3 smoke value at the fixed absolute tolerance."""

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
    range_values_compared = sum(len(row) for row in first.get("ranges_m", []))
    boolean_values_compared = sum(1 for key in ("validity",) for row in first[key] for value in row if isinstance(value, bool))
    coverage_pass = (
        range_values_compared == 21240
        and boolean_values_compared == 21240
        and first["contract"]["ray_count"] == 21240
    )
    if not coverage_pass:
        errors.append("$: full 21,240-element validity coverage was not proven")
    result = {
        "schema_version": "m3-smoke-comparison-1",
        "tolerance": TOLERANCE,
        "max_absolute_numeric_error": max(maxima, default=0.0),
        "numeric_values_compared": len(maxima),
        "range_values_compared": range_values_compared,
        "validity_values_compared": boolean_values_compared,
        "full_vector_coverage_pass": coverage_pass,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
