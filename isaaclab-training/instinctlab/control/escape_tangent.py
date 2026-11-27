"""Find locally feasible progress around confirmed static geometry."""

import math
import torch

from .ics import AZIMUTH_SAMPLES, ELEVATION_SAMPLES, _fresh_history


def surface_tangent(lidar, state, rays, limiting):
    """Fit the horizontal trace of a nearby static surface from delivered scans."""
    history = state.reshape(-1, 4, 13)
    gravity = history[:, -1, 3:6]
    command = history[:, -1, 6:9]
    horizontal = command-(command*gravity).sum(dim=1, keepdim=True)*gravity
    heading = horizontal/horizontal.norm(dim=1, keepdim=True).clamp_min(1.e-6)
    sideways = torch.cross(gravity, heading, dim=1)
    ranges = lidar[:, 0::3].flatten(2)
    reliable = ((lidar[:, 1::3] > .5) & (lidar[:, 2::3] >= .65)).flatten(2)
    active = limiting & reliable[:, -1]
    nearest, center = ranges[:, -1].masked_fill(~active, torch.inf).min(dim=1)
    anchor = rays[center]*nearest.clamp_max(40.)[:, None]

    span = torch.arange(-9, 10, device=rays.device)
    elevation = center[:, None]//AZIMUTH_SAMPLES+span.repeat_interleave(len(span))
    azimuth = (center[:, None]%AZIMUTH_SAMPLES+span.repeat(len(span)))%AZIMUTH_SAMPLES
    beams = elevation.clamp(0, ELEVATION_SAMPLES-1)*AZIMUTH_SAMPLES+azimuth
    indices = beams[:, None].expand(-1, 3, -1)
    recent = ranges[:, -3:].gather(2, indices)
    points = rays[beams]*recent[:, -1, :, None]-anchor[:, None]
    previous_speed = history[:, -2:, 9:12].norm(dim=2).amax(dim=1, keepdim=True)
    range_change = recent.diff(dim=1).abs().amax(dim=1)
    support = (reliable[:, -3:].gather(2, indices).all(dim=1) & (range_change <= .1*(previous_speed+.12))
               & (points.norm(dim=2) < .40) & (elevation >= 0) & (elevation < ELEVATION_SAMPLES))
    weights = support.to(points.dtype)
    local = torch.stack(((points*heading[:, None]).sum(dim=2), (points*sideways[:, None]).sum(dim=2)), dim=2)
    count = weights.sum(dim=1, keepdim=True)
    divisor = count.clamp_min(1.)
    mean = (local*weights[..., None]).sum(dim=1)/divisor
    centered = local-mean[:, None]
    covariance = (centered*weights[..., None]).transpose(1, 2)@centered/divisor[:, None]
    variance, basis = torch.linalg.eigh(covariance)
    tangent = heading*basis[:, 0, 1, None]+sideways*basis[:, 1, 1, None]
    component = (horizontal*tangent).sum(dim=1, keepdim=True)
    confirmed = ((nearest >= .57) & (nearest <= 1.05) & (count[:, 0] >= 18)
                 & (variance[:, 0] <= .025**2) & (variance[:, 1] >= .050**2)
                 & (component[:, 0].abs() >= .09) & _fresh_history(state, command, .1)[:, 0])
    return confirmed, component*tangent


def finite_edge_targets(lidar, state, rays, limiting):
    """Propose deflections only toward a flank with repeated observed free depth."""
    history = state.reshape(-1, 4, 13)
    gravity = history[:, -1, 3:6]
    command = history[:, -1, 6:9]
    horizontal = command-(command*gravity).sum(dim=1, keepdim=True)*gravity
    speed = horizontal.norm(dim=1, keepdim=True)
    heading = horizontal/speed.clamp_min(1.e-6)
    lateral = torch.cross(-gravity, heading, dim=1)
    direction = torch.stack((heading, lateral, -gravity), dim=2)
    ray_local = rays[None] @ direction
    ranges = lidar[:, 0::3].flatten(2)[:, -3:]
    reliable = ((lidar[:, 1::3] > .5) & (lidar[:, 2::3] >= .65)).flatten(2)[:, -3:]
    previous_speed = history[:, -2:, 9:12].norm(dim=2).amax(dim=1, keepdim=True)
    range_change = ranges.diff(dim=1).abs().amax(dim=1)
    stable_limiter = limiting & reliable.all(dim=1) & (range_change <= .1*(previous_speed+.20))
    distance, beam = ranges[:, -1].masked_fill(~stable_limiter, torch.inf).min(dim=1)
    forward = ray_local[:, :, 0]
    depth = (forward.gather(1, beam[:, None])[:, 0]*distance.clamp_max(40.)).clamp(.45, 1.05)
    across = depth[:, None, None]*ray_local[:, :, 1:]/forward.clamp_min(.20)[..., None]
    beyond = (ranges*forward[:, None] >= depth[:, None, None]+.30) & reliable & (range_change <= .12)[:, None]
    measured = distance.isfinite() & _fresh_history(state, command, .1)[:, 0]
    targets, supported = [], []
    for sign in (-1., 1.):
        offset = across[:, :, 0]*sign
        flank = (offset >= .20) & (offset <= .65) & (across[:, :, 1].abs() <= .35) & (forward > .20)
        columns = (beyond & flank[:, None]).reshape(len(lidar), 3, ELEVATION_SAMPLES, AZIMUTH_SAMPLES)
        visible = measured & (columns.any(dim=2).sum(dim=2) >= 4).all(dim=1)
        for angle in (.50, .73):
            targets.append(.90*speed*(math.cos(angle)*heading+sign*math.sin(angle)*lateral))
            supported.append(visible)
    return torch.stack(targets, dim=1), torch.stack(supported, dim=1)
