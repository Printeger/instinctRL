"""Exact all-field comparator for the deterministic-mean pair."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
from typing import Any

import torch

from attempt_binding import AttemptBinding, binding_from_environment, bound_path, path_from_environment
from io_utils import atomic_canonical_replace, canonical_bytes, sha256_file
from semantic_validator import load_semantic, validate_semantic


V2_PROFILE = "EVAL_DETERMINISTIC_MEAN_PAIR_V2"
V2_LIVE_METADATA_PATHS = (
    "root.cuda_resources.before.free_memory_bytes",
    "root.cuda_resources.after.free_memory_bytes",
)
MINIMUM_FREE_MEMORY_BYTES = 1024 ** 3


def compare_exact(left: Any, right: Any, path: str = "root") -> tuple[int, float]:
    if type(left) is not type(right):
        raise ValueError(f"type mismatch at {path}")
    if isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            error = math.inf
            if left.shape == right.shape and left.numel() and left.dtype == right.dtype:
                error = float((left.double() - right.double()).abs().max().item())
            raise ValueError(f"tensor mismatch at {path}, max_error={error}")
        return left.numel(), 0.0
    if isinstance(left, dict):
        if tuple(left) != tuple(right):
            raise ValueError(f"mapping key/order mismatch at {path}")
        count = 0
        for key in left:
            child_count, _ = compare_exact(left[key], right[key], f"{path}.{key}")
            count += child_count
        return count, 0.0
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"sequence length mismatch at {path}")
        count = 0
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            child_count, _ = compare_exact(a, b, f"{path}[{index}]")
            count += child_count
        return count, 0.0
    if left != right:
        raise ValueError(f"scalar mismatch at {path}")
    return 1, 0.0


def _type_name(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"tensor[{value.dtype},{list(value.shape)}]"
    return type(value).__name__


def _collect_v2(
    left: Any,
    right: Any,
    path: str,
    differences: list[OrderedDict[str, Any]],
) -> int:
    if path in V2_LIVE_METADATA_PATHS:
        return 0
    if type(left) is not type(right):
        differences.append(OrderedDict((
            ("path", path), ("kind", "type"),
            ("left_type", _type_name(left)), ("right_type", _type_name(right)),
        )))
        return 0
    if isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            differences.append(OrderedDict((
                ("path", path), ("kind", "tensor_type_or_shape"),
                ("left", _type_name(left)), ("right", _type_name(right)),
            )))
            return 0
        unequal = torch.ne(left, right)
        if bool(unequal.any().item()):
            maximum_error = math.inf
            if left.numel() and not left.dtype.is_complex:
                maximum_error = float((left.double() - right.double()).abs().max().item())
            differences.append(OrderedDict((
                ("path", path), ("kind", "tensor_value"),
                ("unequal_elements", int(unequal.sum().item())),
                ("maximum_error", maximum_error),
            )))
        return left.numel()
    if isinstance(left, dict):
        left_keys = tuple(left)
        right_keys = tuple(right)
        if left_keys != right_keys:
            differences.append(OrderedDict((
                ("path", path), ("kind", "mapping_key_order"),
                ("left_keys", list(left_keys)), ("right_keys", list(right_keys)),
            )))
        count = 0
        for key in left_keys:
            if key in right:
                count += _collect_v2(left[key], right[key], f"{path}.{key}", differences)
        return count
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            differences.append(OrderedDict((
                ("path", path), ("kind", "sequence_length"),
                ("left_length", len(left)), ("right_length", len(right)),
            )))
        count = 0
        for index, (a, b) in enumerate(zip(left, right)):
            count += _collect_v2(a, b, f"{path}[{index}]", differences)
        return count
    if isinstance(left, float) and (not math.isfinite(left) or not math.isfinite(right)):
        differences.append(OrderedDict((
            ("path", path), ("kind", "nonfinite_scalar"),
            ("left", repr(left)), ("right", repr(right)),
        )))
    elif left != right:
        differences.append(OrderedDict((
            ("path", path), ("kind", "scalar_value"),
            ("left", left), ("right", right),
        )))
    return 1


def _path_value(value: Any, path: str) -> Any:
    current = value
    for component in path.removeprefix("root.").split("."):
        if not isinstance(current, dict) or component not in current:
            raise ValueError(f"missing declared live metadata path: {path}")
        current = current[component]
    return current


def compare_v2(left: Any, right: Any) -> OrderedDict[str, Any]:
    """Collect all V2 differences and gate exactly two live metadata values."""

    differences: list[OrderedDict[str, Any]] = []
    compared_count = _collect_v2(left, right, "root", differences)
    metadata: list[OrderedDict[str, Any]] = []
    metadata_failures: list[OrderedDict[str, Any]] = []
    all_difference_paths = [entry["path"] for entry in differences]
    for path in V2_LIVE_METADATA_PATHS:
        try:
            left_value = _path_value(left, path)
            right_value = _path_value(right, path)
            phase = path.split(".")[-2]
            left_total = _path_value(left, f"root.cuda_resources.{phase}.total_memory_bytes")
            right_total = _path_value(right, f"root.cuda_resources.{phase}.total_memory_bytes")
            valid = all(
                type(value) is int
                for value in (left_value, right_value, left_total, right_total)
            ) and all(
                MINIMUM_FREE_MEMORY_BYTES <= free <= total
                for free, total in ((left_value, left_total), (right_value, right_total))
            )
            if not valid:
                metadata_failures.append(OrderedDict((
                    ("path", path), ("kind", "unsafe_or_malformed_live_metadata"),
                )))
            delta = abs(left_value - right_value) if type(left_value) is int and type(right_value) is int else None
            metadata.append(OrderedDict((
                ("path", path), ("left", left_value), ("right", right_value),
                ("absolute_delta", delta), ("safety_gate", "PASS" if valid else "FAIL"),
            )))
            if left_value != right_value:
                all_difference_paths.append(path)
        except (KeyError, TypeError, ValueError) as error:
            metadata_failures.append(OrderedDict((
                ("path", path), ("kind", "missing_live_metadata"),
                ("message", str(error)),
            )))
            metadata.append(OrderedDict((
                ("path", path), ("left", None), ("right", None),
                ("absolute_delta", None), ("safety_gate", "FAIL"),
            )))
    status = "PASS" if not differences and not metadata_failures else "FAIL"
    failure_class = None
    if differences:
        failure_class = "PRODUCT_SEMANTIC_FAILURE"
    elif metadata_failures:
        failure_class = "INVALID_LIVE_RESOURCE_METADATA"
    return OrderedDict((
        ("schema_version", "eval-001-pair-comparison.v2"),
        ("profile", V2_PROFILE),
        ("status", status),
        ("failure_class", failure_class),
        ("declared_live_metadata_paths", list(V2_LIVE_METADATA_PATHS)),
        ("all_difference_paths", all_difference_paths),
        ("deterministic_differences", differences),
        ("live_metadata", metadata),
        ("metadata_failures", metadata_failures),
        ("compared_leaf_or_tensor_element_count", compared_count),
        ("deterministic_domain_maximum_error", 0.0 if not differences else None),
    ))


def _receipt(path: Path) -> OrderedDict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("receipt must be an absolute regular non-symlink file")
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"), object_pairs_hook=OrderedDict)
    if raw != canonical_bytes(value) or not isinstance(value, OrderedDict):
        raise ValueError("receipt must be canonical JSON object")
    return value


def compare_files(binding: AttemptBinding, left_path: Path, right_path: Path, left_receipt: Path, right_receipt: Path) -> OrderedDict[str, Any]:
    left_path = bound_path(binding, left_path, "results/mean_cert_a/semantic.pt", kind="file", must_exist=True)
    right_path = bound_path(binding, right_path, "results/mean_cert_b/semantic.pt", kind="file", must_exist=True)
    left_receipt = bound_path(binding, left_receipt, "results/mean_cert_a/receipt.json", kind="file", must_exist=True)
    right_receipt = bound_path(binding, right_receipt, "results/mean_cert_b/receipt.json", kind="file", must_exist=True)
    left = load_semantic(left_path)
    right = load_semantic(right_path)
    validate_semantic(left, run_id="mean_cert_a")
    validate_semantic(right, run_id="mean_cert_b")
    left_process = _receipt(left_receipt)
    right_process = _receipt(right_receipt)
    if left_process.get("run_id") != "mean_cert_a" or right_process.get("run_id") != "mean_cert_b":
        raise ValueError("pair receipt run identity mismatch")
    if not isinstance(left_process.get("pid"), int) or not isinstance(right_process.get("pid"), int) or left_process["pid"] == right_process["pid"]:
        raise ValueError("formal process PIDs are not distinct")
    count, maximum_error = compare_exact(left, right)
    return OrderedDict((
        ("schema_version", "eval-001-pair-comparison.v1"),
        ("status", "PASS"), ("action_mode", "deterministic_mean"),
        ("compared_leaf_or_tensor_element_count", count),
        ("maximum_error", maximum_error),
        ("left_sha256", sha256_file(left_path)),
        ("right_sha256", sha256_file(right_path)),
        ("left_receipt_sha256", sha256_file(left_receipt)),
        ("right_receipt_sha256", sha256_file(right_receipt)),
        ("formal_pids", [left_process["pid"], right_process["pid"]]),
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    binding = binding_from_environment()
    output = bound_path(
        binding, path_from_environment("EVAL_PAIR_OUTPUT"),
        "results/pair_comparison.json", must_exist=False,
    )
    try:
        result = compare_files(
            binding,
            path_from_environment("EVAL_LEFT_SEMANTIC"),
            path_from_environment("EVAL_RIGHT_SEMANTIC"),
            path_from_environment("EVAL_LEFT_RECEIPT"),
            path_from_environment("EVAL_RIGHT_RECEIPT"),
        )
        atomic_canonical_replace(output, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as error:
        atomic_canonical_replace(output, {
            "failure_class": "PRODUCT_SEMANTIC_FAILURE",
            "message": f"{type(error).__name__}: {error}",
            "semantic_exit": 1,
            "status": "FAIL",
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
