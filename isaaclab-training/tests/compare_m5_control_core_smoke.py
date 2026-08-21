"""Compare every persisted tensor from two independent M5 CUDA processes."""

from __future__ import annotations

import argparse

import torch


def compare(left, right, path: str, counts: dict[str, int]) -> None:
    if type(left) is not type(right):
        raise AssertionError(f"type mismatch at {path}")
    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(f"key mismatch at {path}")
        for key in sorted(left):
            compare(left[key], right[key], f"{path}.{key}", counts)
        return
    if isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"tensor metadata mismatch at {path}")
        if left.is_floating_point() and (not torch.isfinite(left).all() or not torch.isfinite(right).all()):
            raise AssertionError(f"non-finite tensor at {path}")
        if not torch.equal(left, right):
            raise AssertionError(f"tensor mismatch at {path}")
        counts["tensors"] += 1
        counts["elements"] += left.numel()
        return
    if left != right:
        raise AssertionError(f"value mismatch at {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    counts = {"tensors": 0, "elements": 0}
    compare(torch.load(args.left, map_location="cpu", weights_only=True), torch.load(args.right, map_location="cpu", weights_only=True), "root", counts)
    print(f"M5 complete comparison PASS: tensors={counts['tensors']} elements={counts['elements']} max_error=0")


if __name__ == "__main__":
    main()
