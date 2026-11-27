"""Mounted MID360 geometry and measurement-history command attenuation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from instinctlab.observations.actor import M3_MAX_RANGE_M, M3_MIN_RANGE_M
from .governor import clip_vector_norm

AZIMUTH_SAMPLES = 360
ELEVATION_SAMPLES = 59
RAY_COUNT = 21_240
MOUNT_QUATERNION_WXYZ = (0.9659258262890683, 0.0, 0.25881904510252074, 0.0)

ICS_METRIC_KEYS = frozenset(
    ("beta", "active_beam_count", "minimum_clearance_m", "worst_margin_m_s", "emergency")
)
ICS_CACHE_KEYS = frozenset(
    ("active_mask", "approach_speed_m_s", "range_rate_m_s", "effective_clearance_m", "v_safe_m_s")
)


@dataclass(frozen=True)
class ICSConfig:
    d_safe_m: float = 0.8
    emergency_clearance_m: float = 0.25
    a_max_m_s2: float = 2.0
    active_horizon_margin_m: float = 0.5
    min_reliability: float = 0.1
    velocity_limit_m_s: float = 2.0
    latency_s: float = 0.0
    clearance_margin_m: float = 0.0
    range_rate_filter: bool = False
    empty_active_set_beta: float = 1.0
    motion_tube_radius_m: float = 0.0
    motion_tube_confirm_closure: bool = False
    motion_tube_neighbor_history: bool = False
    confirmed_standoff_m: float = 0.0
    confirmed_tube_radius_m: float = 0.0
    confirmed_tube_align_rotation: bool = False

    def __post_init__(self) -> None:
        for name in (
            "d_safe_m", "emergency_clearance_m", "a_max_m_s2", "active_horizon_margin_m",
            "min_reliability", "velocity_limit_m_s", "latency_s", "clearance_margin_m",
            "empty_active_set_beta", "motion_tube_radius_m", "confirmed_standoff_m", "confirmed_tube_radius_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.d_safe_m <= 0.0:
            raise ValueError("d_safe_m must be positive")
        if not 0.0 <= self.emergency_clearance_m <= self.d_safe_m:
            raise ValueError("emergency_clearance_m must be in [0, d_safe_m]")
        if self.a_max_m_s2 <= 0.0 or self.velocity_limit_m_s <= 0.0:
            raise ValueError("acceleration and velocity limits must be positive")
        if self.active_horizon_margin_m < 0.0 or self.latency_s < 0.0 or self.clearance_margin_m < 0.0:
            raise ValueError("horizon, latency, and clearance margins must be nonnegative")
        if not 0.0 <= self.min_reliability <= 1.0:
            raise ValueError("min_reliability must be in [0, 1]")
        if not 0.0 <= self.empty_active_set_beta <= 1.0:
            raise ValueError("empty_active_set_beta must be in [0, 1]")
        if not 0.0 <= self.motion_tube_radius_m <= self.d_safe_m:
            raise ValueError("motion tube radius must be in [0, d_safe_m]")
        if not 0.0 <= self.confirmed_tube_radius_m <= self.motion_tube_radius_m:
            raise ValueError("confirmed tube radius must fit inside the motion tube")
        if not isinstance(self.confirmed_tube_align_rotation, bool):
            raise TypeError("confirmed_tube_align_rotation must be boolean")
        if self.confirmed_tube_align_rotation and self.confirmed_tube_radius_m == 0.0:
            raise ValueError("rotation alignment requires an inner tube")
        if not isinstance(self.range_rate_filter, bool):
            raise TypeError("range_rate_filter must be boolean")
        if not isinstance(self.motion_tube_confirm_closure, bool):
            raise TypeError("motion_tube_confirm_closure must be boolean")
        if not isinstance(self.motion_tube_neighbor_history, bool):
            raise TypeError("motion_tube_neighbor_history must be boolean")
        if self.confirmed_standoff_m != 0.0 and not self.emergency_clearance_m <= self.confirmed_standoff_m <= self.d_safe_m:
            raise ValueError("confirmed standoff must remain between emergency clearance and the usual stopping distance")


@dataclass(frozen=True)
class ICSResult:
    v_final_b: torch.Tensor
    metrics: dict[str, torch.Tensor]
    cache: dict[str, torch.Tensor]
    bilateral_stopping: torch.Tensor | None = None


def _quat_rotate_wxyz(quaternion: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    scalar = quaternion[..., 0:1]
    imaginary = quaternion[..., 1:4]
    return (
        vectors
        + 2.0 * scalar * torch.cross(imaginary.expand_as(vectors), vectors, dim=-1)
        + 2.0 * torch.cross(
            imaginary.expand_as(vectors),
            torch.cross(imaginary.expand_as(vectors), vectors, dim=-1),
            dim=-1,
        )
    )


def mounted_mid360_rays_body(
    *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Derive M3 elevation-outer/azimuth-fastest rays and apply +30 deg pitch."""
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("mounted rays require float32 or float64")
    azimuth = torch.arange(-180.0, 180.0, 1.0, device=device, dtype=dtype)
    elevation = torch.linspace(-7.0, 52.0, 59, device=device, dtype=dtype)
    elevation_grid, azimuth_grid = torch.meshgrid(elevation, azimuth, indexing="ij")
    azimuth_rad = torch.deg2rad(azimuth_grid.reshape(-1))
    elevation_rad = torch.deg2rad(elevation_grid.reshape(-1))
    cosine = torch.cos(elevation_rad)
    local = torch.stack(
        (cosine * torch.cos(azimuth_rad), cosine * torch.sin(azimuth_rad), torch.sin(elevation_rad)),
        dim=-1,
    )
    mount = torch.tensor(MOUNT_QUATERNION_WXYZ, device=device, dtype=dtype).expand(RAY_COUNT, -1)
    return _quat_rotate_wxyz(mount, local).contiguous()


def extract_lidar_history(lidar_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact oldest-to-newest M4 range/validity/reliability histories."""
    if not isinstance(lidar_grid, torch.Tensor):
        raise TypeError("lidar_grid must be a torch.Tensor")
    if lidar_grid.ndim != 4 or lidar_grid.shape[0] <= 0 or tuple(lidar_grid.shape[1:]) != (12, 59, 360):
        raise ValueError("lidar_grid must have shape (N, 12, 59, 360)")
    return lidar_grid[:, 0::3], lidar_grid[:, 1::3].to(torch.bool), lidar_grid[:, 2::3]


def _validate(
    lidar_grid: torch.Tensor,
    rays_body: torch.Tensor,
    v_gov_b: torch.Tensor,
    history_dt_s: float,
    config: ICSConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(config, ICSConfig):
        raise TypeError("config must be ICSConfig")
    ranges, validity, reliability = extract_lidar_history(lidar_grid)
    if not isinstance(rays_body, torch.Tensor) or rays_body.shape != (RAY_COUNT, 3):
        raise ValueError("rays_body must have shape (21240, 3)")
    if not isinstance(v_gov_b, torch.Tensor) or v_gov_b.shape != (lidar_grid.shape[0], 3):
        raise ValueError("v_gov_b must have shape (N, 3)")
    if isinstance(history_dt_s, bool) or not isinstance(history_dt_s, (int, float)):
        raise TypeError("history_dt_s must be a real scalar")
    if not math.isfinite(float(history_dt_s)) or history_dt_s <= 0.0:
        raise ValueError("history_dt_s must be finite and positive")
    if lidar_grid.dtype not in (torch.float32, torch.float64):
        raise TypeError("ICS tensors must use float32 or float64")
    if rays_body.dtype != lidar_grid.dtype or v_gov_b.dtype != lidar_grid.dtype:
        raise TypeError("ICS tensors must share one dtype")
    if rays_body.device != lidar_grid.device or v_gov_b.device != lidar_grid.device:
        raise ValueError("ICS tensors must share one device")
    if not bool(torch.isfinite(lidar_grid).all().item()) or not bool(torch.isfinite(rays_body).all().item()) or not bool(torch.isfinite(v_gov_b).all().item()):
        raise ValueError("ICS tensors must be finite")
    validity_raw = lidar_grid[:, 1::3]
    if not bool(((validity_raw == 0.0) | (validity_raw == 1.0)).all().item()):
        raise ValueError("M4 validity history must be exactly binary")
    if bool(((reliability < 0.0) | (reliability > 1.0)).any().item()) or bool(
        (reliability > validity_raw).any().item()
    ):
        raise ValueError("M4 reliability history must preserve validity and [0, 1]")
    lidar_zero = (
        (ranges == 0.0).flatten(2).all(dim=-1)
        & (validity_raw == 0.0).flatten(2).all(dim=-1)
        & (reliability == 0.0).flatten(2).all(dim=-1)
    )
    populated = ~lidar_zero
    populated_expanded = populated[:, :, None, None]
    if bool((populated_expanded & ((ranges < M3_MIN_RANGE_M) | (ranges > M3_MAX_RANGE_M))).any().item()):
        raise ValueError("populated M4 ranges must preserve the M3 domain")
    if bool((populated_expanded & (~validity) & (ranges != M3_MAX_RANGE_M)).any().item()):
        raise ValueError("invalid populated ranges must use the exact M3 fill")
    ray_norm = torch.linalg.vector_norm(rays_body, dim=-1)
    tolerance = 2.0e-5 if lidar_grid.dtype == torch.float32 else 1.0e-10
    if bool((torch.abs(ray_norm - 1.0) > tolerance).any().item()):
        raise ValueError("rays_body must contain unit directions")
    return ranges, validity, reliability


def _neighbor_history(ranges, reliable, rays, command, candidates, across, parallel):
    lanes, beams = candidates.nonzero(as_tuple=True)
    if not len(lanes):
        return lanes, beams
    span = torch.arange(-3, 4, device=beams.device)
    row = beams[:, None] // AZIMUTH_SAMPLES + span.repeat_interleave(len(span))
    azimuth = (beams[:, None] % AZIMUTH_SAMPLES + span.repeat(len(span))) % AZIMUTH_SAMPLES
    valid_row = (row >= 0) & (row < ELEVATION_SAMPLES)
    neighbors = row.clamp(0, ELEVATION_SAMPLES-1)*AZIMUTH_SAMPLES + azimuth
    clocks = torch.arange(1, 3, device=beams.device)
    valid = reliable[lanes[:, None, None], clocks[None, :, None], neighbors[:, None]]
    depth = ranges[lanes[:, None, None], clocks[None, :, None], neighbors[:, None]]
    heading = command[lanes]/command[lanes].norm(dim=1, keepdim=True).clamp_min(1.e-6)
    axis = (rays[neighbors]*heading[:, None]).sum(dim=2).clamp(0., 1.)
    closer = depth*axis[:, None]
    sideways = depth*(1.-axis[:, None].square()).sqrt()
    cross_match = (sideways-across[lanes, beams, None, None]).abs() <= .045
    along_match = (closer-parallel[lanes, beams, None, None]).abs() <= .095
    supported = ((valid & valid_row[:, None] & cross_match & along_match).sum(dim=2) >= 2).all(dim=1)
    return lanes[supported], beams[supported]


def _fresh_history(state, reference, interval, *, maximum_gyro=.22):
    if state is None:
        return torch.zeros_like(reference[:, :1], dtype=torch.bool)
    if not isinstance(state, torch.Tensor):
        raise TypeError("measurement history must be a tensor")
    if state.shape != (len(reference), 52) or state.device != reference.device or state.dtype != reference.dtype:
        raise ValueError("measurement history must have shape (N, 52) and share the LiDAR dtype and device")
    history = state.reshape(-1, 4, 13)[:, 1:]
    age = history[:, :, 12]
    gap = interval-age.diff(dim=1)
    gentle_rotation = history[:, :, :3].norm(dim=2).amax(dim=1, keepdim=True) <= maximum_gyro
    fresh = (age <= .4*interval).all(dim=1, keepdim=True)
    consecutive = ((gap >= .7*interval) & (gap <= 1.3*interval)).all(dim=1, keepdim=True)
    return fresh & consecutive & gentle_rotation


def _visible_inner_tube(ranges, reliable, rays, heading, gravity, radius):
    """Certify forward depth using actual returns bracketing the request."""
    cosine = (heading @ rays.T).clamp(-1., 1.)[:, None]
    side = (torch.cross(gravity, heading, dim=1) @ rays.T)[:, None] >= 0.
    return _bracketed_inner_tube(ranges, reliable, cosine, side, radius)


def _bracketed_inner_tube(ranges, reliable, cosine, side, radius):
    depth = ranges*cosine
    across = ranges*(1.-cosine.square()).sqrt()
    occupied = reliable & (depth > 0.) & (depth <= 1.10) & (across < radius)
    witness = reliable & (cosine >= .99) & (depth > 1.10)
    columns = [((witness & flank).reshape(len(ranges), 3, ELEVATION_SAMPLES, AZIMUTH_SAMPLES)
                .any(dim=2).sum(dim=2) >= 3).all(dim=1) for flank in (side, ~side)]
    return columns[0] & columns[1] & ~occupied.flatten(1).any(dim=1)


def _rotate_inner_tube(ranges, reliable, rays, heading, state, interval, radius):
    history = state.reshape(-1, 4, 13)[:, 1:]
    gap = interval-history[:, :, 12].diff(dim=1)
    angular_step = .5*(history[:, 1:, :3]+history[:, :-1, :3])*gap[..., None]
    half_angle = .5*angular_step.norm(dim=2, keepdim=True)
    rotation = torch.cat((half_angle.cos(), .5*angular_step*torch.sinc(half_angle/math.pi)), dim=2)
    intent = history[:, :, 6:9]
    speed = intent.norm(dim=2, keepdim=True)
    unit = intent/speed.clamp_min(1.e-6)
    requested_axes = torch.stack((heading, unit[:, -1]), dim=1)
    previous = _quat_rotate_wxyz(rotation[:, 1:2], requested_axes)
    oldest = _quat_rotate_wxyz(rotation[:, 0:1], previous)
    axes = torch.stack((oldest, previous, requested_axes), dim=1)
    residual = (axes[:, :, 1]-unit).norm(dim=2)
    minimum_speed = speed[:, :, 0].amin(dim=1)
    maximum_speed = speed[:, :, 0].amax(dim=1)
    coherent = ((residual.amax(dim=1) <= .020) & (minimum_speed >= .10) & (maximum_speed <= .30)
                & (maximum_speed-minimum_speed <= .025))
    headings = axes[:, :, 0]
    cosine = (headings @ rays.T).clamp(-1., 1.)
    side = (torch.cross(history[:, :, 3:6], headings, dim=2) @ rays.T) >= 0.
    # Carry uncertainty from the measured intent alignment to the end of the observed core.
    radius_with_rotation_error = radius+1.10*residual[..., None]
    return coherent & _bracketed_inner_tube(ranges, reliable, cosine, side, radius_with_rotation_error)


def attenuate_command(
    lidar_grid: torch.Tensor,
    rays_body: torch.Tensor,
    v_gov_b: torch.Tensor,
    history_dt_s: float,
    config: ICSConfig = ICSConfig(),
    *,
    state_vec: torch.Tensor | None = None,
    allow_bilateral_stopping: bool = False,
) -> ICSResult:
    """Apply bounded scalar attenuation and zero-command emergency braking."""
    ranges, validity, reliability = _validate(lidar_grid, rays_body, v_gov_b, history_dt_s, config)
    latest_range = ranges[:, -1].flatten(1)
    latest_valid = validity[:, -1].flatten(1)
    latest_reliability = reliability[:, -1].flatten(1)
    rays = rays_body.unsqueeze(0).expand(lidar_grid.shape[0], -1, -1)
    approach = torch.einsum("nrd,nd->nr", rays, v_gov_b).clamp_min(0.0)

    consecutive_valid = validity[:, 1:] & validity[:, :-1]
    consecutive_reliable = (reliability[:, 1:] >= config.min_reliability) & (
        reliability[:, :-1] >= config.min_reliability
    )
    rates = (ranges[:, 1:] - ranges[:, :-1]) / float(history_dt_s)
    usable_rates = consecutive_valid & consecutive_reliable
    if config.range_rate_filter:
        count = usable_rates.flatten(2).sum(dim=1).clamp_min(1)
        rate = (rates * usable_rates.to(rates.dtype)).flatten(2).sum(dim=1) / count
        closing = torch.maximum(approach, (-rate).clamp_min(0.0))
    else:
        latest_pair = usable_rates[:, -1].flatten(1)
        newest_rate = rates[:, -1].flatten(1)
        rate = torch.where(latest_pair, newest_rate, torch.zeros_like(newest_rate))
        closing = approach

    speed = torch.linalg.vector_norm(v_gov_b, dim=-1, keepdim=True)
    effective = latest_range - speed * config.latency_s - config.clearance_margin_m
    v_safe = torch.sqrt(2.0 * config.a_max_m_s2 * (effective - config.d_safe_m).clamp_min(0.0))
    reliable = latest_valid & (latest_reliability >= config.min_reliability)
    if config.motion_tube_neighbor_history or config.confirmed_standoff_m > 0. or config.confirmed_tube_radius_m > 0.:
        fresh = _fresh_history(state_vec, speed, history_dt_s)
    if config.confirmed_standoff_m > 0.:
        last_two_rates = rates[:, -2:].flatten(2)
        last_three_reliable = (validity[:, -3:] & (reliability[:, -3:] >= .65)).all(dim=1).flatten(1)
        matching_speed = last_two_rates.abs().amax(dim=1) <= approach+.12
        confirmed = fresh & (speed <= .30) & last_three_reliable & matching_speed
        tighter = torch.sqrt(2.0*config.a_max_m_s2*(effective-config.confirmed_standoff_m).clamp_min(0.))
        v_safe = torch.where(confirmed, tighter, v_safe)
    active = (
        reliable
        & (closing > 0.0)
        & (effective <= config.d_safe_m + config.active_horizon_margin_m)
    )
    if config.motion_tube_radius_m > 0.0:
        # At low speed, relax only fresh off-path returns with two consistent
        # range intervals. Unexplained closure retains the radial brake.
        parallel_m = latest_range * approach / speed.clamp_min(1e-6)
        across_m = (latest_range.square() - parallel_m.square()).clamp_min(0.).sqrt()
        last_two_rates = rates[:, -2:].flatten(2)
        last_two_valid = usable_rates[:, -2:].flatten(2).all(dim=1)
        closure = -last_two_rates
        measured_closure = closure.amin(dim=1) if config.motion_tube_confirm_closure else closure.amax(dim=1)
        unexplained_closure = measured_closure > approach + .18
        outside = ((across_m > config.motion_tube_radius_m) & last_two_valid & ~unexplained_closure
                   & (latest_reliability >= .65) & (speed <= .30))
        if config.motion_tube_neighbor_history:
            ambiguous = (active & (v_safe < closing) & ~last_two_valid & (across_m > config.motion_tube_radius_m)
                         & (latest_reliability >= .65) & (speed <= .30) & fresh)
            support = (validity & (reliability >= .65)).flatten(2)
            lanes, beam_ids = _neighbor_history(ranges.flatten(2), support, rays_body, v_gov_b, ambiguous,
                                                 across_m, parallel_m)
            outside[lanes, beam_ids] = True
        if config.confirmed_tube_radius_m > 0.:
            proof = (validity[:, -3:] & (reliability[:, -3:] >= .65)).flatten(2)
            quiet = last_two_rates.abs().amax(dim=1) <= approach+.10
            shoulder = (active & ~outside & (v_safe < closing) & (across_m >= config.confirmed_tube_radius_m)
                        & proof.all(dim=1) & quiet & (speed <= .30))
            lanes = (shoulder & fresh).any(dim=1).nonzero().flatten()
            if len(lanes):
                heading = v_gov_b[lanes]/speed[lanes].clamp_min(1.e-6)
                clear = _visible_inner_tube(ranges[lanes, -3:].flatten(2), proof[lanes], rays_body,
                                            heading, state_vec[lanes, 42:45], config.confirmed_tube_radius_m)
                outside[lanes[clear]] |= shoulder[lanes[clear]]
            if config.confirmed_tube_align_rotation:
                rotating = _fresh_history(state_vec, speed, history_dt_s, maximum_gyro=.38) & ~fresh
                lanes = (shoulder & rotating).any(dim=1).nonzero().flatten()
                if len(lanes):
                    heading = v_gov_b[lanes]/speed[lanes].clamp_min(1.e-6)
                    clear = _rotate_inner_tube(ranges[lanes, -3:].flatten(2), proof[lanes], rays_body,
                                               heading, state_vec[lanes], history_dt_s, config.confirmed_tube_radius_m)
                    outside[lanes[clear]] |= shoulder[lanes[clear]]
        active = active & ~outside
    ratios = (v_safe / closing.clamp_min(torch.finfo(v_gov_b.dtype).tiny)).clamp(0.0, 1.0)
    emergency = (reliable & (effective < config.emergency_clearance_m)).any(dim=1, keepdim=True)
    bilateral = None
    if allow_bilateral_stopping:
        from .bilateral_stopping import clear_bilateral_constraints

        if state_vec is None:
            raise ValueError('Bilateral stopping requires the approved command and IMU history')
        cleared = clear_bilateral_constraints(lidar_grid, state_vec, rays_body, v_gov_b,
                                              active & ~emergency, ratios, history_dt_s)
        active &= ~cleared
        bilateral = cleared.any(dim=1)
    beta = torch.where(active, ratios, torch.ones_like(ratios)).min(dim=1, keepdim=True).values
    active_count = active.sum(dim=1, keepdim=True)
    beta = torch.where(active_count > 0, beta, torch.full_like(beta, config.empty_active_set_beta))
    beta = torch.where(emergency, torch.zeros_like(beta), beta).clamp(0.0, 1.0)
    final = clip_vector_norm(beta * v_gov_b, config.velocity_limit_m_s)
    minimum = torch.where(reliable, effective, torch.full_like(effective, float("inf"))).min(dim=1, keepdim=True).values
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.full_like(minimum, M3_MAX_RANGE_M))
    margin = v_safe - closing
    worst = torch.where(active, margin, torch.full_like(margin, float("inf"))).min(dim=1, keepdim=True).values
    worst = torch.where(torch.isfinite(worst), worst, torch.zeros_like(worst))
    metrics = {
        "beta": beta,
        "active_beam_count": active_count.to(v_gov_b.dtype),
        "minimum_clearance_m": minimum,
        "worst_margin_m_s": worst,
        "emergency": emergency.to(v_gov_b.dtype),
    }
    cache = {
        "active_mask": active,
        "approach_speed_m_s": approach,
        "range_rate_m_s": rate,
        "effective_clearance_m": effective,
        "v_safe_m_s": v_safe,
    }
    return ICSResult(final, metrics, cache, bilateral)


__all__ = [
    "AZIMUTH_SAMPLES", "ELEVATION_SAMPLES", "ICS_CACHE_KEYS", "ICS_METRIC_KEYS",
    "ICSConfig", "ICSResult", "MOUNT_QUATERNION_WXYZ", "RAY_COUNT",
    "attenuate_command", "extract_lidar_history", "mounted_mid360_rays_body",
]
