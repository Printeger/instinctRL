#!/usr/bin/env python3
"""One-articulation, one-ray-caster deterministic CUDA smoke for M3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from isaaclab.app import AppLauncher

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--output", type=Path, required=True)
ARGS = PARSER.parse_args()
APP_LAUNCHER = AppLauncher(headless=True, enable_cameras=False)
SIMULATION_APP = APP_LAUNCHER.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster
from isaaclab.sim import SimulationContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instinctlab.assets.taslab_uav.articulation import TASLAB_ARTICULATION_CFG
from instinctlab.sensors.mid360 import (
    MAX_RANGE_M,
    MID360_RAY_CASTER_CFG,
    MIN_RANGE_M,
    RAY_COUNT,
    UPDATE_PERIOD_S,
    mid360_pattern,
    ranges_and_validity,
)

DEVICE = "cuda:0"
DT = 0.005
TOLERANCE = 1.0e-5
SELECTED_INDICES = (0, 179, 180, 359, 360, 10799, 10800, 20879, 21239)


def _tensor(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=DEVICE))
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/MID360StaticScene", ground_cfg)
    robot = Articulation(TASLAB_ARTICULATION_CFG.replace(prim_path="/World/TASLAB"))
    sensor_cfg = MID360_RAY_CASTER_CFG.replace(prim_path="/World/TASLAB/base_link")
    sensor = RayCaster(sensor_cfg)
    sim.reset()
    robot.update(0.0)
    sensor.reset()
    sensor.update(UPDATE_PERIOD_S, force_recompute=True)

    hits_w = sensor.data.ray_hits_w
    origins_w = sensor.data.pos_w
    raw_ranges_m, serialized_ranges_m, valid = ranges_and_validity(hits_w, origins_w)
    pattern_starts, pattern_directions = mid360_pattern(sensor_cfg.pattern_cfg, DEVICE)
    invalid = ~valid
    shape_pass = tuple(serialized_ranges_m.shape) == (1, RAY_COUNT) and tuple(valid.shape) == (1, RAY_COUNT)
    finite_serialized = bool(torch.isfinite(serialized_ranges_m).all().item())
    valid_count = int(valid.sum().item())
    invalid_count = int(invalid.sum().item())
    bounds_pass = bool(
        ((serialized_ranges_m[valid] >= MIN_RANGE_M) & (serialized_ranges_m[valid] <= MAX_RANGE_M)).all().item()
    )
    fill_pass = bool((serialized_ranges_m[invalid] == MAX_RANGE_M).all().item())
    raw_invalid_distinguishable = bool((~torch.isfinite(raw_ranges_m[invalid])).any().item())
    both_classes = valid_count > 0 and invalid_count > 0
    all_gates = shape_pass and finite_serialized and bounds_pass and fill_pass and raw_invalid_distinguishable and both_classes

    selected = [
        {
            "index": index,
            "start_sensor_frame_m": _tensor(pattern_starts[index]),
            "direction_sensor_frame": _tensor(pattern_directions[index]),
            "serialized_range_m": float(serialized_ranges_m[0, index].item()),
            "valid": bool(valid[0, index].item()),
        }
        for index in SELECTED_INDICES
    ]
    valid_ranges = serialized_ranges_m[valid]
    result = {
        "schema_version": "m3-mid360-smoke-1",
        "contract": {
            "device": DEVICE,
            "dt_s": DT,
            "ray_count": RAY_COUNT,
            "output_shape": list(serialized_ranges_m.shape),
            "min_range_m": MIN_RANGE_M,
            "max_range_m": MAX_RANGE_M,
            "update_period_s": UPDATE_PERIOD_S,
            "flatten_order": "elevation outer, azimuth fastest; index=elevation_index*360+azimuth_index",
            "sensor_axes": "+X forward, +Y left, +Z up",
            "mount_parent": sensor_cfg.prim_path,
            "mount_position_body_m": list(sensor_cfg.offset.pos),
            "mount_quaternion_wxyz": list(sensor_cfg.offset.rot),
            "mount_rpy_deg": [0.0, 30.0, 0.0],
            "ray_alignment": sensor_cfg.ray_alignment,
            "mesh_prim_paths": list(sensor_cfg.mesh_prim_paths),
            "debug_vis": sensor_cfg.debug_vis,
            "tolerance": TOLERANCE,
        },
        "sensor_pose_world": {"origin_m": _tensor(origins_w)},
        "tensor_contract": {
            "hit_device": str(hits_w.device),
            "hit_dtype": str(hits_w.dtype),
            "range_device": str(serialized_ranges_m.device),
            "range_dtype": str(serialized_ranges_m.dtype),
            "validity_device": str(valid.device),
            "validity_dtype": str(valid.dtype),
            "pattern_device": str(pattern_directions.device),
            "pattern_dtype": str(pattern_directions.dtype),
            "pattern_starts_contiguous": pattern_starts.is_contiguous(),
            "pattern_directions_contiguous": pattern_directions.is_contiguous(),
        },
        "ranges_m": _tensor(serialized_ranges_m),
        "validity": _tensor(valid),
        "summary": {
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "valid_min_range_m": float(valid_ranges.min().item()),
            "valid_max_range_m": float(valid_ranges.max().item()),
            "invalid_raw_representation": "one or more non-finite raw geometric distances; not serialized as numbers",
            "invalid_serialized_fill_m": MAX_RANGE_M,
            "selected_rays": selected,
        },
        "gates": {
            "shape_pass": shape_pass,
            "serialized_ranges_finite": finite_serialized,
            "valid_ranges_closed_interval_pass": bounds_pass,
            "invalid_fill_pass": fill_pass,
            "raw_invalid_distinguishable": raw_invalid_distinguishable,
            "both_valid_and_invalid_present": both_classes,
            "all_gates_pass": all_gates,
        },
        "operation_audit": {
            "app_launcher": "headless=True, enable_cameras=False",
            "simulation_contexts": 1,
            "articulations": 1,
            "ray_casters": 1,
            "static_ground_planes": 1,
            "simulation_reset_calls": 1,
            "explicit_physics_steps": 0,
            "robot_update_calls": 1,
            "sensor_reset_calls": 1,
            "sensor_update_calls": 1,
            "control_calls": 0,
            "shutdown": "os._exit after normalized JSON and streams are flushed",
        },
    }
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    ARGS.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if all_gates else "FAIL", "valid": valid_count, "invalid": invalid_count}, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if all_gates else 2)


if __name__ == "__main__":
    main()
