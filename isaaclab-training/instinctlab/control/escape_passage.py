"""Use four delivered scans to recognize a clear pass at zero stick."""

import torch
from torch.nn import functional as F


def _closing_seed(ranges, reliable, elapsed):
    old = -torch.where(reliable[:, :3], ranges[:, :3], torch.inf)
    old = F.max_pool2d(F.pad(old, (8, 8, 0, 0), mode='circular'), (1, 17), stride=1)
    nearby = -F.max_pool2d(old, (17, 1), stride=1, padding=(8, 0))
    gap = -elapsed[:, :3, None, None]
    closing = nearby-ranges[:, -1:]
    support = ((gap >= .16) & (gap <= .33) & (closing > .20*gap) & (closing < 1.5*gap)
               & nearby.isfinite() & (ranges[:, -1:] < 1.9) & reliable[:, -1:])
    supported = support.clone()
    for offset in (-2, -1, 1, 2):
        supported &= support.roll(offset, dims=-1)
    best, seed = torch.where(supported, closing, 0.).amax(dim=1).flatten(1).max(dim=1)
    return best > .005, seed


def clear_to_end_escape(lidar, state):
    """Use overlapping fresh scan support before ending an older escape at zero stick."""
    history = state.reshape(-1, 4, 13)
    ranges = lidar[:, 0::3]
    reliable = (lidar[:, 1::3] > .5) & (lidar[:, 2::3] > .55)
    ages = history[:, :, 12]
    elapsed = ranges.new_tensor([-.3, -.2, -.1, 0.])-ages+ages[:, -1:]
    gap = elapsed.diff(dim=1)
    fresh = (ages <= .04).all(dim=1) & ((gap >= .07) & (gap <= .13)).all(dim=1)
    stable = history[:, :, :3].norm(dim=2).amax(dim=1) <= .50
    coverage = reliable.all(dim=1).to(ranges.dtype).flatten(1).mean(dim=1) >= .60
    nearest = ranges[:, -1].masked_fill(~reliable[:, -1], torch.inf).flatten(1).amin(dim=1)
    closing, _ = _closing_seed(ranges, reliable, elapsed)
    return fresh & stable & coverage & (nearest >= .85) & ~closing


def _fit_curvature(points, patch):
    weights = patch.to(points.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.)
    mean = (points*weights[:, :, None]).sum(dim=1)/count
    offsets = points-mean[:, None]
    weighted = offsets*weights[:, :, None]
    covariance = offsets.transpose(1,2)@weighted/count[:, :, None]
    powers = offsets.square().sum(dim=2, keepdim=True)
    right = (weighted.transpose(1,2)@powers)/(2.*count[:, :, None])
    identity = torch.eye(3, device=points.device, dtype=points.dtype)
    local, info = torch.linalg.solve_ex(covariance+1.e-8*identity, right)
    centers = mean+local.squeeze(-1)
    radii = (points-centers[:, None]).norm(dim=-1)
    radius = (radii*weights).sum(dim=1)/count[:, 0]
    residual = (((radii-radius[:, None]).square()*weights).sum(dim=1)/count[:, 0]).sqrt()
    supported = (info == 0) & (torch.linalg.eigvalsh(covariance)[:, 0] >= 1.e-5)
    supported &= (radius >= .14) & (radius <= .9) & (residual <= .008)
    return centers[:, :2], radius, supported


def track_nearby_return(lidar, state, rays, *, curved=False, maximum_previous_command=.08):
    """Associate a supported closing return across four delivered scans."""
    histories = state.reshape(-1, 4, 13)
    ranges = lidar[:, 0::3]
    reliable = (lidar[:, 1::3] > .5) & (lidar[:, 2::3] > .55)
    elapsed = ranges.new_tensor([-.3, -.2, -.1, 0.]) - histories[:, :, 12] + histories[:, -1:, 12]
    gap = elapsed.diff(dim=1)
    accepted, seed = _closing_seed(ranges, reliable, elapsed)
    accepted &= ((gap >= .065) & (gap <= .135)).all(dim=1)
    accepted &= histories[:, :, 9:12].norm(dim=2).amax(dim=1) <= maximum_previous_command
    lanes = torch.arange(len(lidar), device=lidar.device)
    rays = rays[None].expand(len(lidar), -1, -1)
    aligned = rays
    medians = []
    shapes = []
    anchor = None
    for frame in (3, 2, 1, 0):
        if frame < 3:
            gyro = (histories[:, frame, :3]+histories[:, frame+1, :3])*.5
            delta = -gyro*gap[:, frame, None]
            angle = delta.norm(dim=1, keepdim=True)
            axis = (delta/angle.clamp_min(1.e-8))[:, None].expand_as(aligned)
            cross = torch.cross(axis, aligned, dim=-1)
            aligned = aligned + cross*angle.sin()[:, None] + torch.cross(axis, cross, dim=-1)*(1.-angle.cos())[:, None]
        points = aligned*ranges[:, frame].flatten(1)[:, :, None]
        valid = reliable[:, frame].flatten(1)
        if anchor is not None:
            distance = (points-anchor[:, None]).norm(dim=-1).masked_fill(~valid, torch.inf)
            separation, seed = distance.min(dim=1)
            accepted &= separation <= .20
        anchor = points[lanes, seed]
        offsets = points-anchor[:, None]
        heading = anchor/anchor.norm(dim=1, keepdim=True).clamp_min(.01)
        patch = valid & (offsets.norm(dim=-1) < .55) & ((offsets*heading[:, None]).sum(dim=-1).abs() < .22)
        accepted &= patch.sum(dim=1) >= 20
        coords = points[:, :, :2].masked_fill(~patch[:, :, None], torch.nan)
        medians.append(coords.nanmedian(dim=1).values)
        if curved:
            shapes.append(_fit_curvature(points, patch))
    path = torch.stack(medians[::-1], dim=1)
    return accepted, path, elapsed, shapes[::-1]


def _fit_motion(path, elapsed):
    mean = path.mean(dim=1)
    t = elapsed-elapsed.mean(dim=1, keepdim=True)
    velocity = (path*t[:, :, None]).sum(dim=1)/(t.square().sum(dim=1, keepdim=True).clamp_min(.001))
    residual = path-mean[:, None]-velocity[:, None]*t[:, :, None]
    rms = residual.square().sum(dim=2).mean(dim=1).sqrt()
    now = mean-velocity*elapsed.mean(dim=1, keepdim=True)
    return now, velocity, rms


def _closest_passage(path, elapsed):
    now, velocity, rms = _fit_motion(path, elapsed)
    speed_sq = velocity.square().sum(dim=1)
    moment = -(now*velocity).sum(dim=1)/speed_sq.clamp_min(.001)
    closest = (now+moment.clamp(0., 5.)[:, None]*velocity).norm(dim=1)
    fit = (rms <= .02) & (speed_sq >= .04) & (speed_sq <= 2.25) & (moment > 0.)
    return fit, closest


def clear_hover_passage(lidar, state, rays, *, curved=False):
    """Recognize a clear pass before the vehicle leaves its stopped position."""
    accepted, path, elapsed, shapes = track_nearby_return(lidar, state, rays, curved=curved)
    fit, closest = _closest_passage(path, elapsed)
    clear = fit & (closest >= .85)
    if curved:
        centers, radii, viable = (torch.stack(parts, dim=1) for parts in zip(*shapes, strict=True))
        fitted, center_distance = _closest_passage(centers, elapsed)
        outer = radii.amax(dim=1)
        supported = viable.all(dim=1) & fitted & ((outer-radii.amin(dim=1)) <= .04)
        # Allocate .21 m to the vehicle and .15 m to clearance, with a .09 m observation margin.
        clear = torch.where(supported, center_distance-outer >= .45, clear)
    return accepted & clear


def hover_escape_scores(lidar, state, rays, issued):
    """Compare the held escape directions and zero across a fitted curved return."""
    supported, _, elapsed, shapes = track_nearby_return(lidar, state, rays, curved=True, maximum_previous_command=.30001)
    centers, radii, viable = (torch.stack(parts, dim=1) for parts in zip(*shapes, strict=True))
    position, relative, error = _fit_motion(centers, elapsed)
    previous = state.reshape(-1, 4, 13)[:, :, 9:12]
    stable = (previous-previous[:, -1:]).norm(dim=2).amax(dim=1) <= .035
    last = previous[:, -1, :2]
    motion = relative+last
    radius = radii.amax(dim=1)
    supported &= stable & viable.all(dim=1) & (error <= .02) & ((radius-radii.amin(dim=1)) <= .04)
    supported &= (motion.norm(dim=1) >= .2) & (motion.norm(dim=1) <= 1.5) & ((position*relative).sum(dim=1) < 0.)
    time = torch.arange(81, device=position.device, dtype=position.dtype)*.05
    hold = time.clamp_max(2.5)
    lag = .35
    coast = lag*(1.-torch.exp(-time/lag))
    travel = hold-lag*(1.-torch.exp(-hold/lag))*torch.exp(-(time-hold)/lag)
    predicted = position[:, None]+time[None, :, None]*motion[:, None]-coast[None, :, None]*last[:, None]
    separation = (predicted[:, None]-travel[None, None, :, None]*issued[:, :, None, :2]).norm(dim=-1)
    return supported, separation.amin(dim=2)-radius[:, None]


def approach_scores(lidar, state, rays, issued):
    """Estimate separation for admissible escape commands after a nearby reversal."""
    accepted, history, elapsed, _ = track_nearby_return(lidar, state, rays)
    recent = history.diff(dim=1)[:, -2:] / elapsed.diff(dim=1)[:, -2:, None].clamp_min(.001)
    speed = recent.norm(dim=2)
    agreement = (recent[:, 0]*recent[:, 1]).sum(dim=1)/speed.prod(dim=1).clamp_min(.001)
    velocity = .3*recent[:, 0]+.7*recent[:, 1]
    position = history[:, -1]
    accepted &= (speed >= .14).all(dim=1) & (speed <= 1.5).all(dim=1) & (agreement > .85)
    accepted &= (position.norm(dim=1) < .9) & ((position*velocity).sum(dim=1) < -.10)
    time = position.new_tensor(.05)*torch.arange(33, device=position.device, dtype=position.dtype)
    travelled = time-.35*(1.-torch.exp(-time/.35))
    path = position[:, None]+time[None, :, None]*velocity[:, None]
    separation = (path[:, None]-travelled[None, None, :, None]*issued[:, :, None, :2]).norm(dim=-1)
    return accepted, separation.amin(dim=2)
