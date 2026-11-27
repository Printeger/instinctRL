"""Recover the feasible component of a request along a confirmed surface."""

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
