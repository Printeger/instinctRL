#!/usr/bin/env python3
"""Complete exact comparator for two M7-018 cluttered_indoor CUDA bundles."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import json
from pathlib import Path

import torch


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--first", type=Path, required=True)
PARSER.add_argument("--second", type=Path, required=True)
PARSER.add_argument("--output", type=Path, required=True)
PARSER.add_argument("--mutation-self-test", action="store_true")
ARGS = PARSER.parse_args()


def _compare(first: object, second: object, path: str = "root") -> None:
    if type(first) is not type(second):
        raise AssertionError(f"{path}: exact types differ")
    if isinstance(first, torch.Tensor):
        if first.shape != second.shape or first.dtype != second.dtype or not torch.equal(first, second):
            raise AssertionError(f"{path}: tensor bits differ")
        return
    if isinstance(first, dict):
        if tuple(first) != tuple(second):
            raise AssertionError(f"{path}: mapping keys/order differ")
        for key in first:
            _compare(first[key], second[key], f"{path}.{key}")
        return
    if isinstance(first, (tuple, list)):
        if len(first) != len(second):
            raise AssertionError(f"{path}: sequence length differs")
        for index, (left, right) in enumerate(zip(first, second, strict=True)):
            _compare(left, right, f"{path}[{index}]")
        return
    if first != second:
        raise AssertionError(f"{path}: values differ")


def _must_reject(reference: object, mutated: object, name: str) -> str:
    try:
        _compare(reference, mutated)
    except AssertionError:
        return name
    raise AssertionError(f"mutation was not rejected: {name}")


def main() -> None:
    if ARGS.output.exists():
        raise FileExistsError(ARGS.output)
    first = torch.load(ARGS.first, map_location="cpu", weights_only=False)
    second = torch.load(ARGS.second, map_location="cpu", weights_only=False)
    _compare(first, second)
    mutations = []
    if ARGS.mutation_self_test:
        targets = (
            ("missing", lambda value: value.pop("geometry")),
            ("extra", lambda value: value.update(extra=None)),
            ("reordered", lambda value: value.move_to_end("fixture")),
            ("tensor_bit", lambda value: value["evaluation_inputs"]["commanded_velocity_b_m_s"].view(torch.uint8).__setitem__(0, value["evaluation_inputs"]["commanded_velocity_b_m_s"].view(torch.uint8)[0] ^ 1)),
            ("geometry", lambda value: value["geometry"].__setitem__("selected_warp_point_count", 0)),
            ("positive_x_panel_hit", lambda value: value["wall_visibility"].__setitem__("positive_x_panel_hit_counts", ())),
            ("negative_x_panel_hit", lambda value: value["wall_visibility"].__setitem__("negative_x_panel_hit_counts", ())),
            ("positive_y_panel_hit", lambda value: value["wall_visibility"].__setitem__("positive_y_panel_hit_counts", ())),
            ("negative_y_panel_hit", lambda value: value["wall_visibility"].__setitem__("negative_y_panel_hit_counts", ())),
            ("membership", lambda value: value["wall_visibility"].__setitem__("valid_clutter_hit_counts", ())),
            ("positive_x_panel_normal", lambda value: value["wall_visibility"].__setitem__("positive_x_panel_normal_negative_x_verified", False)),
            ("negative_x_panel_normal", lambda value: value["wall_visibility"].__setitem__("negative_x_panel_normal_positive_x_verified", False)),
            ("positive_y_panel_normal", lambda value: value["wall_visibility"].__setitem__("positive_y_panel_normal_negative_y_verified", False)),
            ("negative_y_panel_normal", lambda value: value["wall_visibility"].__setitem__("negative_y_panel_normal_positive_y_verified", False)),
            ("command", lambda value: value["fixture"].__setitem__("schedule", value["fixture"]["schedule"][::-1])),
            ("metric_count", lambda value: value["metric_oracle"]["tracking_rmse_m_s"].__setitem__("total_count", 15)),
            ("plan", lambda value: value["fixture"].__setitem__("plan", value["fixture"]["plan"] + " ")),
            ("receipt_record", lambda value: value.__setitem__("evaluation_record", value["evaluation_record"] + " ")),
            ("qualification_record", lambda value: value.__setitem__("qualification_record", value["qualification_record"] + " ")),
            ("private_field", lambda value: value["evaluation_inputs"].update(private_pose=torch.zeros(1))),
        )
        for name, mutate in targets:
            candidate = copy.deepcopy(first)
            mutate(candidate)
            mutations.append(_must_reject(first, candidate, name))
    output = OrderedDict((
        ("schema", "m7.cluttered-indoor-b0-comparison.v1"),
        ("status", "PASS"),
        ("complete_exact_equality", True),
        ("mutation_rejections", mutations),
        ("first", str(ARGS.first)),
        ("second", str(ARGS.second)),
    ))
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    ARGS.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "mutations": len(mutations)}, sort_keys=True))


if __name__ == "__main__":
    main()
