#!/usr/bin/env python3
"""Complete exact comparator for two M7-020 gap-doorway CUDA bundles."""

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
            ("tensor-bit", lambda value: value["evaluation_inputs"]["commanded_velocity_b_m_s"].view(torch.uint8).view(-1).__setitem__(0, value["evaluation_inputs"]["commanded_velocity_b_m_s"].view(torch.uint8).view(-1)[0] ^ 1)),
            ("geometry", lambda value: value["geometry"].__setitem__("point_count", 0)),
            ("negative-jamb hits", lambda value: value["frame_visibility"].__setitem__("negative_jamb_hit_counts", ())),
            ("positive-jamb hits", lambda value: value["frame_visibility"].__setitem__("positive_jamb_hit_counts", ())),
            ("lintel hits", lambda value: value["frame_visibility"].__setitem__("lintel_hit_counts", ())),
            ("valid hits", lambda value: value["frame_visibility"].__setitem__("valid_frame_hit_counts", ())),
            ("membership", lambda value: value["frame_visibility"].__setitem__("exclusive_membership_verified", False)),
            ("normal", lambda value: value["frame_visibility"].__setitem__("frame_normal_negative_x_verified", False)),
            ("body-command", lambda value: value["command_direction"].__setitem__("body_commanded_velocity_m_s", ())),
            ("world-projection", lambda value: value["command_direction"].__setitem__("world_commanded_velocity_m_s", ())),
            ("plane-intersection", lambda value: value["command_direction"].__setitem__("intersection_parameter", ())),
            ("aperture-margin", lambda value: value["command_direction"].__setitem__("upper_z_margin_m", ())),
            ("command", lambda value: value["fixture"].__setitem__("schedule", value["fixture"]["schedule"][1:] + value["fixture"]["schedule"][:1])),
            ("plan", lambda value: value["fixture"].__setitem__("plan", value["fixture"]["plan"] + " ")),
            ("metric", lambda value: value["metric_oracle"]["tracking_rmse_m_s"].__setitem__("total_count", 15)),
            ("receipt/record", lambda value: value.__setitem__("evaluation_record", value["evaluation_record"] + " ")),
            ("qualification", lambda value: value.__setitem__("qualification_record", value["qualification_record"] + " ")),
            ("private-field", lambda value: value["evaluation_inputs"].update(private_pose=torch.zeros(1))),
        )
        for name, mutate in targets:
            candidate = copy.deepcopy(first)
            mutate(candidate)
            mutations.append(_must_reject(first, candidate, name))
    output = OrderedDict((
        ("schema", "m7.gap-doorway-b0-comparison.v1"),
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
