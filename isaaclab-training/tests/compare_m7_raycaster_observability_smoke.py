#!/usr/bin/env python3
"""Fail-closed exact comparison of complete live normal-acquisition trees."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


RAW_INF_SUFFIXES = (
    ".source.accepted_ray_hits_w",
    ".source.accepted_raw_ranges_m",
    ".acquisition.ray_hits_w",
    ".acquisition.ray_distances_m",
    ".acquisition.raw_surface_normals_w",
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
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _compare(a, b, f"{path}[{index}]", report)
    elif isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"tensor metadata mismatch at {path}")
        permits_inf = any(path.endswith(suffix) for suffix in RAW_INF_SUFFIXES)
        if left.is_floating_point() and not permits_inf and (
            not torch.isfinite(left).all() or not torch.isfinite(right).all()
        ):
            raise AssertionError(f"nonfinite tensor at {path}")
        if permits_inf and (torch.isnan(left).any() or torch.isnan(right).any()):
            raise AssertionError(f"NaN in audited raw tensor at {path}")
        if not torch.equal(left, right):
            raise AssertionError(f"tensor mismatch at {path}")
        report["tensor_count"] += 1
        report["element_count"] += left.numel()
    elif isinstance(left, float):
        if not math.isfinite(left) or not math.isfinite(right) or left != right:
            raise AssertionError(f"float mismatch at {path}")
    elif left != right:
        raise AssertionError(f"value mismatch at {path}")


def _validate(tree: dict[str, Any]) -> None:
    if list(tree) != ["schema_version", "manifest", "provenance", "scans", "gates"]:
        raise AssertionError("root schema missing, extra, or reordered")
    manifest = tree["manifest"]
    expected = {"num_envs": 2, "scans": 2, "rays_per_environment": 21240,
                "ray_samples": 84960, "public_actions": 1, "physics_steps": 20,
                "actor_inference_calls": 0, "sampling_calls": 0, "gradient_calls": 0,
                "optimizer_calls": 0, "training_performed": False, "runtime_record_wiring": False}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise AssertionError("manifest coverage or prohibited-operation mismatch")
    if len(tree["scans"]) != 2 or [scan["label"] for scan in tree["scans"]] != [
        "initial", "after_one_public_action"
    ]:
        raise AssertionError("scan order mismatch")
    if not all(tree["gates"].values()):
        raise AssertionError("top-level gate failure")
    for scan in tree["scans"]:
        if not all(scan["gates"].values()):
            raise AssertionError(f"scan gate failure: {scan['label']}")
        source, acquisition, exact = scan["source"], scan["acquisition"], scan["observability"]
        valid = source["accepted_beam_valid"]
        if valid.shape != (2, 21240) or valid.dtype != torch.bool:
            raise AssertionError("accepted validity metadata mismatch")
        if not torch.equal(valid, acquisition["beam_valid"][0]):
            raise AssertionError("handoff validity mismatch")
        if not torch.equal(source["accepted_ray_hits_w"], acquisition["ray_hits_w"]):
            raise AssertionError("accepted/acquired hit mismatch")
        if not torch.equal(valid, torch.isfinite(acquisition["ray_hits_w"]).all(dim=-1)):
            raise AssertionError("hit validity disagreement")
        invalid = ~valid
        if not bool((torch.isinf(acquisition["raw_surface_normals_w"][invalid]) &
                     (acquisition["raw_surface_normals_w"][invalid] > 0)).all().item()):
            raise AssertionError("raw normal miss convention mismatch")
        if not bool((acquisition["surface_normals_w"][0][invalid] == 0).all().item()):
            raise AssertionError("qualified miss placeholder mismatch")
        if not bool((exact["numerical_rank"] == 1).all().item()):
            raise AssertionError("planar rank mismatch")
        if bool(exact["condition_valid"].any().item()) or bool(exact["weak_direction_valid"].any().item()):
            raise AssertionError("planar degeneracy availability mismatch")
        if acquisition["hit_max_error_m"].item() != 0.0:
            raise AssertionError("nonzero hit error")
    frozen = tree["provenance"]["frozen"]
    if frozen["isaac_lab_head"] != "37ddf626871758333d6ed89cf64ad702aef127d0" or frozen["isaac_lab_version"] != "2.3.2":
        raise AssertionError("frozen provenance mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = torch.load(args.first, map_location="cpu", weights_only=True)
    second = torch.load(args.second, map_location="cpu", weights_only=True)
    _validate(first)
    _validate(second)
    report = {"status": "PASS", "tensor_count": 0, "element_count": 0,
              "ray_samples_per_process": 84960, "max_absolute_error": 0.0}
    _compare(first, second, "root", report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
