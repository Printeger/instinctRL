"""Deterministic Livox MID360 geometry and range-validity contract."""

from __future__ import annotations

from collections.abc import Callable

import torch
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul

AZIMUTH_MIN_DEG = -180.0
AZIMUTH_MAX_DEG_EXCLUSIVE = 180.0
AZIMUTH_STEP_DEG = 1.0
AZIMUTH_SAMPLES = 360
ELEVATION_MIN_DEG = -7.0
ELEVATION_MAX_DEG = 52.0
ELEVATION_SAMPLES = 59
RAY_COUNT = AZIMUTH_SAMPLES * ELEVATION_SAMPLES
MIN_RANGE_M = 0.1
MAX_RANGE_M = 40.0
UPDATE_PERIOD_S = 0.1
MOUNT_PARENT_PRIM_PATH = "/World/TASLAB/base_link"
MOUNT_POSITION_M = (0.0, 0.0, 0.05)
# roll=0, pitch=+30 deg, yaw=0, scalar-first (w, x, y, z).
MOUNT_QUATERNION_WXYZ = (0.9659258262890683, 0.0, 0.25881904510252074, 0.0)


def mid360_pattern(cfg: "Mid360PatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate sensor-frame rays with elevation outer and azimuth fastest."""
    azimuth_deg = torch.arange(
        cfg.azimuth_min_deg,
        cfg.azimuth_max_deg_exclusive,
        cfg.azimuth_step_deg,
        device=device,
        dtype=torch.float32,
    )
    elevation_deg = torch.linspace(
        cfg.elevation_min_deg,
        cfg.elevation_max_deg,
        cfg.elevation_samples,
        device=device,
        dtype=torch.float32,
    )
    elevation_grid, azimuth_grid = torch.meshgrid(elevation_deg, azimuth_deg, indexing="ij")
    azimuth_rad = torch.deg2rad(azimuth_grid.reshape(-1))
    elevation_rad = torch.deg2rad(elevation_grid.reshape(-1))
    cos_elevation = torch.cos(elevation_rad)
    directions = torch.stack(
        (
            cos_elevation * torch.cos(azimuth_rad),
            cos_elevation * torch.sin(azimuth_rad),
            torch.sin(elevation_rad),
        ),
        dim=-1,
    ).contiguous()
    starts = torch.zeros_like(directions).contiguous()
    return starts, directions


@configclass
class Mid360PatternCfg(PatternBaseCfg):
    """Exact static pattern parameters qualified for M3."""

    func: Callable = mid360_pattern
    azimuth_min_deg: float = AZIMUTH_MIN_DEG
    azimuth_max_deg_exclusive: float = AZIMUTH_MAX_DEG_EXCLUSIVE
    azimuth_step_deg: float = AZIMUTH_STEP_DEG
    elevation_min_deg: float = ELEVATION_MIN_DEG
    elevation_max_deg: float = ELEVATION_MAX_DEG
    elevation_samples: int = ELEVATION_SAMPLES


MID360_RAY_CASTER_CFG = RayCasterCfg(
    prim_path=MOUNT_PARENT_PRIM_PATH,
    update_period=UPDATE_PERIOD_S,
    offset=RayCasterCfg.OffsetCfg(pos=MOUNT_POSITION_M, rot=MOUNT_QUATERNION_WXYZ),
    mesh_prim_paths=["/World/MID360StaticScene"],
    ray_alignment="base",
    pattern_cfg=Mid360PatternCfg(),
    max_distance=MAX_RANGE_M,
    drift_range=(0.0, 0.0),
    ray_cast_drift_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
    debug_vis=False,
)


def compose_mounted_sensor_pose(
    parent_positions_w: torch.Tensor,
    parent_quaternions_w: torch.Tensor,
    mount_position_parent: torch.Tensor,
    mount_quaternion_parent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose a batched public parent pose with one body-frame sensor mount."""
    tensors = {
        "parent_positions_w": parent_positions_w,
        "parent_quaternions_w": parent_quaternions_w,
        "mount_position_parent": mount_position_parent,
        "mount_quaternion_parent": mount_quaternion_parent,
    }
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("all pose inputs must be torch.Tensor instances")
    if parent_positions_w.ndim != 2 or parent_positions_w.shape[-1] != 3:
        raise ValueError(f"parent_positions_w must have shape (N, 3), got {tuple(parent_positions_w.shape)}")
    if parent_quaternions_w.ndim != 2 or parent_quaternions_w.shape[-1] != 4:
        raise ValueError(
            f"parent_quaternions_w must have shape (N, 4), got {tuple(parent_quaternions_w.shape)}"
        )
    if parent_positions_w.shape[0] != parent_quaternions_w.shape[0]:
        raise ValueError("parent position and quaternion batches must match")
    if mount_position_parent.shape != (3,):
        raise ValueError(f"mount_position_parent must have shape (3,), got {tuple(mount_position_parent.shape)}")
    if mount_quaternion_parent.shape != (4,):
        raise ValueError(
            f"mount_quaternion_parent must have shape (4,), got {tuple(mount_quaternion_parent.shape)}"
        )
    devices = {value.device for value in tensors.values()}
    if len(devices) != 1:
        raise ValueError("all pose inputs must be on the same device")
    dtypes = {value.dtype for value in tensors.values()}
    if len(dtypes) != 1 or not parent_positions_w.is_floating_point():
        raise ValueError("all pose inputs must have the same floating-point dtype")
    if not all(torch.isfinite(value).all().item() for value in tensors.values()):
        raise ValueError("all pose inputs must be finite")

    batch_size = parent_positions_w.shape[0]
    mounted_offset_w = quat_apply(parent_quaternions_w, mount_position_parent.expand(batch_size, -1))
    sensor_positions_w = parent_positions_w + mounted_offset_w
    sensor_quaternions_w = quat_mul(parent_quaternions_w, mount_quaternion_parent.expand(batch_size, -1))
    return sensor_positions_w, sensor_quaternions_w


def ranges_and_validity(
    hit_positions_w: torch.Tensor,
    sensor_origins_w: torch.Tensor,
    min_range_m: float = MIN_RANGE_M,
    max_range_m: float = MAX_RANGE_M,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw ranges, finite serialized ranges, and closed-interval validity."""
    if hit_positions_w.ndim != 3 or hit_positions_w.shape[-1] != 3:
        raise ValueError(f"hit_positions_w must have shape (N, R, 3), got {tuple(hit_positions_w.shape)}")
    if sensor_origins_w.ndim != 2 or sensor_origins_w.shape[-1] != 3:
        raise ValueError(f"sensor_origins_w must have shape (N, 3), got {tuple(sensor_origins_w.shape)}")
    if hit_positions_w.shape[0] != sensor_origins_w.shape[0]:
        raise ValueError("hit positions and origins must have the same batch size")
    if min_range_m < 0.0 or max_range_m < min_range_m:
        raise ValueError("range bounds must satisfy 0 <= min_range_m <= max_range_m")

    deltas = hit_positions_w - sensor_origins_w.unsqueeze(1)
    raw_ranges_m = torch.linalg.vector_norm(deltas, dim=-1)
    finite_hits = torch.isfinite(hit_positions_w).all(dim=-1)
    valid = (
        finite_hits
        & torch.isfinite(raw_ranges_m)
        & (raw_ranges_m >= min_range_m)
        & (raw_ranges_m <= max_range_m)
    )
    serialized_ranges_m = torch.where(valid, raw_ranges_m, torch.full_like(raw_ranges_m, max_range_m))
    return raw_ranges_m, serialized_ranges_m, valid


__all__ = [
    "AZIMUTH_SAMPLES",
    "ELEVATION_SAMPLES",
    "MAX_RANGE_M",
    "MID360_RAY_CASTER_CFG",
    "MIN_RANGE_M",
    "MOUNT_POSITION_M",
    "MOUNT_QUATERNION_WXYZ",
    "Mid360PatternCfg",
    "RAY_COUNT",
    "UPDATE_PERIOD_S",
    "compose_mounted_sensor_pose",
    "mid360_pattern",
    "ranges_and_validity",
]
