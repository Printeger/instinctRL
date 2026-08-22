#!/usr/bin/env python3
"""Fail-closed exact comparison of two complete M7 runtime trees."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def _compare(left: Any, right: Any, path: str, report: dict[str, Any]) -> None:
    if type(left) is not type(right):
        raise AssertionError(f'type mismatch at {path}')
    if isinstance(left, dict):
        if list(left) != list(right):
            raise AssertionError(f'missing, extra, or reordered keys at {path}')
        for key in left:
            _compare(left[key], right[key], f'{path}.{key}', report)
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f'length mismatch at {path}')
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _compare(a, b, f'{path}[{index}]', report)
    elif isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f'tensor metadata mismatch at {path}')
        if left.is_floating_point() and (not torch.isfinite(left).all() or not torch.isfinite(right).all()):
            raise AssertionError(f'nonfinite tensor at {path}')
        if not torch.equal(left, right):
            raise AssertionError(f'tensor mismatch at {path}')
        report['tensor_count'] += 1
        report['element_count'] += left.numel()
    elif isinstance(left, float):
        if not math.isfinite(left) or not math.isfinite(right) or left != right:
            raise AssertionError(f'float mismatch at {path}')
    elif left != right:
        raise AssertionError(f'value mismatch at {path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    first = torch.load(args.first, map_location='cpu', weights_only=True)
    second = torch.load(args.second, map_location='cpu', weights_only=True)
    report = {'status': 'PASS', 'tensor_count': 0, 'element_count': 0, 'max_absolute_error': 0.0}
    _compare(first, second, 'root', report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
