"""Find locally feasible progress around confirmed static geometry."""

import math
import torch

from .ics import AZIMUTH_SAMPLES, ELEVATION_SAMPLES, _fresh_history


def wall_opening_direction(lidar, state, rays):
    """Find a stable frontal surface with measured depth beyond exactly one flank."""
    history = state.reshape(-1, 4, 13)[:, -3:]
    gravity = history[:, -1, 3:6]
    command = history[:, -1, 6:9]
    horizontal = command-(command*gravity).sum(dim=1, keepdim=True)*gravity
    heading = horizontal/horizontal.norm(dim=1, keepdim=True).clamp_min(1.e-6)
    lateral = torch.cross(-gravity, heading, dim=1)
    forwards, sideways, upwards = (rays[None] @ torch.stack((heading, lateral, -gravity), dim=2)).unbind(2)

    ranges = lidar[:, 0::3].flatten(2)[:, -3:]
    measured = ((lidar[:, 1::3] > .5) & (lidar[:, 2::3] >= .65)).flatten(2)[:, -3:].all(dim=1)
    stable = measured & (ranges.diff(dim=1).abs().amax(dim=1) <= .12) & (forwards > .20)
    depths = ranges*forwards[:, None]
    latest = ranges[:, -1]
    frontal = (stable & (depths[:, -1] >= .55) & (depths[:, -1] <= 2.40)
               & ((latest*sideways).abs() <= .55) & ((latest*upwards).abs() <= .35))
    depth_bin = ((depths[:, -1]-.55)/.04).floor().to(torch.long).clamp(0, 46)
    counts = depth_bin.new_zeros((len(lidar), 47))
    counts.scatter_add_(1, depth_bin, frontal.to(torch.long))
    peak = .55+.04*(counts.argmax(dim=1, keepdim=True)+.5)
    face = frontal & ((depths[:, -1]-peak).abs() <= .05)
    plane = (depths[:, -1]*face).sum(dim=1, keepdim=True)/face.sum(dim=1, keepdim=True).clamp_min(1)

    def azimuth_count(observed):
        return observed.reshape(len(lidar), ELEVATION_SAMPLES, AZIMUTH_SAMPLES).any(dim=1).sum(dim=1)

    scale = plane/forwards.clamp_min(.20)
    offset = scale*sideways
    beyond = stable & (depths >= plane[:, None]+.30).all(dim=1) & ((scale*upwards).abs() <= .30)
    right = azimuth_count(beyond & (offset >= .80) & (offset <= 2.40))
    left = azimuth_count(beyond & (offset <= -.80) & (offset >= -2.40))
    one_sided = ((right >= 6) & (left <= 1)) | ((left >= 6) & (right <= 1))
    steady = (history[:, :, 6:9]-command[:, None]).norm(dim=2).amax(dim=1) <= .04
    confirmed = (one_sided & steady & (azimuth_count(face) >= 12) & (plane[:, 0] >= 1.05)
                 & (plane[:, 0] <= 1.65) & _fresh_history(state, command, .1)[:, 0])
    opening = torch.where((right > left)[:, None], lateral, -lateral)
    return confirmed, heading, opening


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
