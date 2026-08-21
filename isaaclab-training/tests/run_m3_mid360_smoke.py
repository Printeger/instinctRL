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
from isaaclab.utils.math import quat_apply, quat_mul
from pxr import UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instinctlab.assets.taslab_uav.articulation import TASLAB_ARTICULATION_CFG
from instinctlab.sensors.mid360 import (
    MAX_RANGE_M,
    MID360_RAY_CASTER_CFG,
    MIN_RANGE_M,
    RAY_COUNT,
    UPDATE_PERIOD_S,
    compose_mounted_sensor_pose,
    mid360_pattern,
    ranges_and_validity,
)

DEVICE = "cuda:0"
DT = 0.005
TOLERANCE = 1.0e-5
SELECTED_INDICES = (0, 179, 180, 359, 360, 10799, 10800, 20879, 21239)


def _tensor(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def _spawn_static_plane() -> None:
    """Create one deterministic z=0 mesh with small triangles for float32 ray accuracy."""
    minimum = -42
    maximum = 42
    spacing = 2
    coordinates = list(range(minimum, maximum + spacing, spacing))
    points = [(float(x), float(y), 0.0) for y in coordinates for x in coordinates]
    width = len(coordinates)
    face_vertex_indices: list[int] = []
    for row in range(width - 1):
        for column in range(width - 1):
            lower_left = row * width + column
            lower_right = lower_left + 1
            upper_left = lower_left + width
            upper_right = upper_left + 1
            face_vertex_indices.extend((lower_left, lower_right, upper_right, lower_left, upper_right, upper_left))
    sim_utils.create_prim("/World/MID360StaticScene", "Xform")
    mesh = UsdGeom.Mesh.Define(sim_utils.get_current_stage(), "/World/MID360StaticScene/PlaneMesh")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * (len(face_vertex_indices) // 3))
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr("none")


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=DEVICE))
    _spawn_static_plane()
    robot = Articulation(TASLAB_ARTICULATION_CFG.replace(prim_path="/World/TASLAB"))
    sensor_cfg = MID360_RAY_CASTER_CFG.replace(prim_path="/World/TASLAB/base_link")
    sensor = RayCaster(sensor_cfg)
    sim.reset()
    robot.update(0.0)
    sensor.reset()
    sensor.update(UPDATE_PERIOD_S, force_recompute=True)

    hits_w = sensor.data.ray_hits_w
    parent_positions_w = sensor.data.pos_w
    parent_quaternions_w = sensor.data.quat_w
    mount_position_parent = torch.tensor(sensor_cfg.offset.pos, device=DEVICE, dtype=parent_positions_w.dtype)
    mount_quaternion_parent = torch.tensor(sensor_cfg.offset.rot, device=DEVICE, dtype=parent_quaternions_w.dtype)
    sensor_origins_w, sensor_quaternions_w = compose_mounted_sensor_pose(
        parent_positions_w,
        parent_quaternions_w,
        mount_position_parent,
        mount_quaternion_parent,
    )
    raw_ranges_m, serialized_ranges_m, valid = ranges_and_validity(hits_w, sensor_origins_w)
    pattern_starts, pattern_directions = mid360_pattern(sensor_cfg.pattern_cfg, DEVICE)
    mounted_ray_directions_parent = quat_apply(
        mount_quaternion_parent.expand(RAY_COUNT, -1),
        pattern_directions,
    )
    ray_directions_w = quat_apply(
        parent_quaternions_w[:, None, :].expand(-1, RAY_COUNT, -1).reshape(-1, 4),
        mounted_ray_directions_parent.repeat(parent_quaternions_w.shape[0], 1),
    ).reshape(sensor_quaternions_w.shape[0], RAY_COUNT, 3)

    ray_direction_norms = torch.linalg.vector_norm(ray_directions_w, dim=-1)
    analytic_ray_parameters = -sensor_origins_w[:, 2:3] / ray_directions_w[:, :, 2]
    analytic_raw_ranges_m = analytic_ray_parameters * ray_direction_norms
    analytic_valid = (
        torch.isfinite(analytic_raw_ranges_m)
        & (ray_directions_w[:, :, 2] < 0.0)
        & (analytic_raw_ranges_m >= MIN_RANGE_M)
        & (analytic_raw_ranges_m <= MAX_RANGE_M)
    )
    analytic_mask_mismatch_count = int((valid != analytic_valid).sum().item())
    analytic_valid_range_errors = torch.abs(serialized_ranges_m[analytic_valid] - analytic_raw_ranges_m[analytic_valid])
    analytic_max_valid_range_error_m = float(analytic_valid_range_errors.max().item())
    analytic_error_grid = torch.where(
        analytic_valid,
        torch.abs(serialized_ranges_m - analytic_raw_ranges_m),
        torch.full_like(serialized_ranges_m, -1.0),
    )
    analytic_max_error_flat_index = int(analytic_error_grid.argmax().item())
    analytic_max_error_batch = analytic_max_error_flat_index // RAY_COUNT
    analytic_max_error_ray = analytic_max_error_flat_index % RAY_COUNT
    measured_unit_directions_w = (hits_w - sensor_origins_w.unsqueeze(1)) / raw_ranges_m.unsqueeze(-1)
    runtime_valid_direction_max_error = float(
        torch.abs(measured_unit_directions_w[valid] - ray_directions_w[valid]).max().item()
    )
    runtime_valid_hit_max_abs_world_z_m = float(torch.abs(hits_w[valid][:, 2]).max().item())

    wrong_parent_ray_parameters = -parent_positions_w[:, 2:3] / ray_directions_w[:, :, 2]
    wrong_parent_raw_ranges_m = wrong_parent_ray_parameters * ray_direction_norms
    wrong_parent_valid = (
        torch.isfinite(wrong_parent_raw_ranges_m)
        & (ray_directions_w[:, :, 2] < 0.0)
        & (wrong_parent_raw_ranges_m >= MIN_RANGE_M)
        & (wrong_parent_raw_ranges_m <= MAX_RANGE_M)
    )
    wrong_parent_mask_mismatch_count = int((wrong_parent_valid != analytic_valid).sum().item())
    wrong_parent_common_valid = wrong_parent_valid & analytic_valid
    wrong_parent_max_range_error_m = float(
        torch.abs(wrong_parent_raw_ranges_m[wrong_parent_common_valid] - analytic_raw_ranges_m[wrong_parent_common_valid])
        .max()
        .item()
    )
    wrong_parent_origin_rejected = (
        wrong_parent_mask_mismatch_count > 0 or wrong_parent_max_range_error_m > TOLERANCE
    )

    mount_separation_m = torch.linalg.vector_norm(sensor_origins_w - parent_positions_w, dim=-1)
    mount_separation_error_m = float(torch.abs(mount_separation_m - 0.05).max().item())
    independently_composed_quaternions_w = quat_mul(
        parent_quaternions_w,
        mount_quaternion_parent.expand(parent_quaternions_w.shape[0], -1),
    )
    mounted_quaternion_max_error = float(
        torch.abs(sensor_quaternions_w - independently_composed_quaternions_w).max().item()
    )
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
    analytic_mask_pass = analytic_mask_mismatch_count == 0
    analytic_range_pass = analytic_max_valid_range_error_m <= TOLERANCE
    mount_separation_pass = mount_separation_error_m <= 1.0e-6
    mounted_quaternion_pass = mounted_quaternion_max_error <= TOLERANCE
    all_gates = (
        shape_pass
        and finite_serialized
        and bounds_pass
        and fill_pass
        and raw_invalid_distinguishable
        and both_classes
        and analytic_mask_pass
        and analytic_range_pass
        and wrong_parent_origin_rejected
        and mount_separation_pass
        and mounted_quaternion_pass
    )

    selected = [
        {
            "index": index,
            "start_sensor_frame_m": _tensor(pattern_starts[index]),
            "direction_sensor_frame": _tensor(pattern_directions[index]),
            "direction_world_frame": _tensor(ray_directions_w[0, index]),
            "analytic_plane_range_m": float(
                analytic_raw_ranges_m[0, index].item() if analytic_valid[0, index] else MAX_RANGE_M
            ),
            "hit_position_world_m": _tensor(hits_w[0, index]) if valid[0, index] else [MAX_RANGE_M] * 3,
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
            "static_plane_top_world_z_m": 0.0,
            "static_plane_xy_size_m": [84.0, 84.0],
            "static_plane_triangle_edge_m": 2.0,
            "debug_vis": sensor_cfg.debug_vis,
            "tolerance": TOLERANCE,
        },
        "parent_pose_world": {
            "position_m": _tensor(parent_positions_w),
            "quaternion_wxyz": _tensor(parent_quaternions_w),
        },
        "sensor_pose_world": {
            "origin_m": _tensor(sensor_origins_w),
            "quaternion_wxyz": _tensor(sensor_quaternions_w),
            "parent_to_sensor_translation_separation_m": _tensor(mount_separation_m),
        },
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
        "analytic_plane_oracle": {
            "plane_world_z_m": 0.0,
            "method": "distance=(-mounted_origin_z/world_direction_z)*world_direction_norm; validity from downward sign and closed range",
            "runtime_mask_mismatch_count": analytic_mask_mismatch_count,
            "runtime_max_valid_range_error_m": analytic_max_valid_range_error_m,
            "runtime_max_error_ray": analytic_max_error_ray,
            "runtime_max_error_ray_values": {
                "runtime_range_m": float(
                    serialized_ranges_m[analytic_max_error_batch, analytic_max_error_ray].item()
                ),
                "analytic_range_m": float(
                    analytic_raw_ranges_m[analytic_max_error_batch, analytic_max_error_ray].item()
                ),
                "world_direction": _tensor(
                    ray_directions_w[analytic_max_error_batch, analytic_max_error_ray]
                ),
                "world_direction_norm": float(
                    ray_direction_norms[analytic_max_error_batch, analytic_max_error_ray].item()
                ),
                "hit_world_z_m": float(hits_w[analytic_max_error_batch, analytic_max_error_ray, 2].item()),
            },
            "runtime_valid_direction_max_error": runtime_valid_direction_max_error,
            "runtime_valid_hit_max_abs_world_z_m": runtime_valid_hit_max_abs_world_z_m,
            "wrong_parent_origin_mask_mismatch_count": wrong_parent_mask_mismatch_count,
            "wrong_parent_origin_max_range_error_m": wrong_parent_max_range_error_m,
            "wrong_parent_origin_rejected": wrong_parent_origin_rejected,
        },
        "gates": {
            "shape_pass": shape_pass,
            "serialized_ranges_finite": finite_serialized,
            "valid_ranges_closed_interval_pass": bounds_pass,
            "invalid_fill_pass": fill_pass,
            "raw_invalid_distinguishable": raw_invalid_distinguishable,
            "both_valid_and_invalid_present": both_classes,
            "mounted_translation_separation_error_m": mount_separation_error_m,
            "mounted_translation_separation_pass": mount_separation_pass,
            "mounted_quaternion_max_error": mounted_quaternion_max_error,
            "mounted_quaternion_pass": mounted_quaternion_pass,
            "analytic_plane_mask_pass": analytic_mask_pass,
            "analytic_plane_ranges_pass": analytic_range_pass,
            "wrong_parent_origin_rejected": wrong_parent_origin_rejected,
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
