#!/usr/bin/env python3
"""Exact complete-record comparison for two independent M7-009 smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

def _compare(left: Any, right: Any, path: str, report: dict[str, int]) -> None:
    if type(left) is not type(right): raise AssertionError(f"type mismatch at {path}")
    if isinstance(left, dict):
        if list(left) != list(right): raise AssertionError(f"key mismatch at {path}")
        for key in left: _compare(left[key], right[key], f"{path}.{key}", report)
    elif isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right): raise AssertionError(f"exact tensor mismatch at {path}")
        report["tensor_count"] += 1; report["element_count"] += left.numel()
    elif left != right: raise AssertionError(f"value mismatch at {path}")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--first", type=Path, required=True); parser.add_argument("--second", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    first = torch.load(args.first.with_suffix(".pt"), map_location="cpu", weights_only=True)
    second = torch.load(args.second.with_suffix(".pt"), map_location="cpu", weights_only=True)
    report = {"status": "PASS", "tensor_count": 0, "element_count": 0}
    _compare(first, second, "root", report)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); print(json.dumps(report, sort_keys=True))

if __name__ == "__main__": main()
