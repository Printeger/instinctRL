"""Compare every tensor element from two independent M4 CUDA smokes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("ascii"))
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def load_bundle(summary_path: Path) -> tuple[dict[str, object], list[dict[str, torch.Tensor]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bundle = torch.load(summary_path.with_suffix(".pt"), map_location="cpu", weights_only=True)
    if not isinstance(bundle, list) or len(bundle) != 6:
        raise ValueError("smoke tensor bundle must contain exactly six complete outputs")
    return summary, bundle


def compare(first_path: Path, second_path: Path) -> dict[str, object]:
    first_summary, first_bundle = load_bundle(first_path)
    second_summary, second_bundle = load_bundle(second_path)
    summary_equal = first_summary == second_summary
    records: list[dict[str, object]] = []
    all_equal = summary_equal
    elements_compared = 0
    for step, (first, second) in enumerate(zip(first_bundle, second_bundle, strict=True), start=1):
        if set(first) != {"lidar_grid", "state_vec"} or set(second) != set(first):
            raise ValueError("bundle output keys violate the exact actor schema")
        for key in ("lidar_grid", "state_vec"):
            left = first[key]
            right = second[key]
            same_shape_dtype = left.shape == right.shape and left.dtype == right.dtype
            exact_equal = bool(same_shape_dtype and torch.equal(left, right))
            max_abs_error = float(torch.max(torch.abs(left - right)).item()) if same_shape_dtype else None
            elements_compared += left.numel()
            all_equal = all_equal and exact_equal
            records.append(
                {
                    "step": step,
                    "key": key,
                    "shape": list(left.shape),
                    "dtype": str(left.dtype),
                    "first_sha256": tensor_hash(left),
                    "second_sha256": tensor_hash(right),
                    "exact_equal": exact_equal,
                    "max_abs_error": max_abs_error,
                }
            )
    return {
        "schema": "m4-observation-comparison-v1",
        "summary_equal": summary_equal,
        "complete_tensor_equal": all_equal,
        "elements_compared": elements_compared,
        "records": records,
        "status": "PASS" if all_equal else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.first, args.second)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
