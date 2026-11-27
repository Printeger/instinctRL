"""Use four delivered scans to recognize a clear pass at zero stick."""

import torch
from torch.nn import functional as F


def _closing_seed(ranges, reliable, elapsed, *, minimum_range_m=0., maximum_range_m=1.9):
    old = -torch.where(reliable[:, :3], ranges[:, :3], torch.inf)
    old = F.max_pool2d(F.pad(old, (8, 8, 0, 0), mode='circular'), (1, 17), stride=1)
    nearby = -F.max_pool2d(old, (17, 1), stride=1, padding=(8, 0))
    gap = -elapsed[:, :3, None, None]
    closing = nearby-ranges[:, -1:]
    current = ranges[:, -1:]
    support = ((gap >= .16) & (gap <= .33) & (closing > .20*gap) & (closing < 1.5*gap)
               & nearby.isfinite() & (current >= minimum_range_m) & (current < maximum_range_m) & reliable[:, -1:])
    supported = support.clone()
    for offset in (-2, -1, 1, 2):
        supported &= support.roll(offset, dims=-1)
    best, seed = torch.where(supported, closing, 0.).amax(dim=1).flatten(1).max(dim=1)
    return best > .005, seed


def clear_to_end_escape(lidar, state, *, minimum_coverage=.60, allow_distant_returns=False):
    """Use overlapping fresh scan support before ending an older escape at zero stick."""
    history = state.reshape(-1, 4, 13)
    ranges = lidar[:, 0::3]
    reliable = (lidar[:, 1::3] > .5) & (lidar[:, 2::3] > .55)
    ages = history[:, :, 12]
    elapsed = ranges.new_tensor([-.3, -.2, -.1, 0.])-ages+ages[:, -1:]
    gap = elapsed.diff(dim=1)
    fresh = (ages <= .04).all(dim=1) & ((gap >= .07) & (gap <= .13)).all(dim=1)
    stable = history[:, :, :3].norm(dim=2).amax(dim=1) <= .50
    coverage = reliable.all(dim=1).to(ranges.dtype).flatten(1).mean(dim=1)
    nearest = ranges[:, -1].masked_fill(~reliable[:, -1], torch.inf).flatten(1).amin(dim=1)
    closing, _ = _closing_seed(ranges, reliable, elapsed)
    observed = (coverage >= minimum_coverage) & (nearest >= .85)
    if allow_distant_returns:
        observed |= (coverage >= .55) & (nearest >= 1.35)
    return fresh & stable & observed & ~closing


def clear_receding_return_at_stop(lidar, state, rays, command_response):
    """Verify a nearby opening surface with the delayed zero-command path."""
    history = state.reshape(-1, 4, 13)
    ranges = lidar[:, 0::3]
    reliable = (lidar[:, 1::3] > .5) & (lidar[:, 2::3] > .55)
    closing, _, elapsed = _return_seed(lidar, state)
    gaps = elapsed.diff(dim=1)
    ages = history[:, :, 12]
    nearest, seed = ranges[:, -1].masked_fill(~reliable[:, -1], torch.inf).flatten(1).min(dim=1)
    coverage = reliable.all(dim=1).float().flatten(1).mean(dim=1)
    ready = ((ages <= .04).all(dim=1) & ((gaps >= .07) & (gaps <= .13)).all(dim=1)
             & (history[:, :, :3].norm(dim=2).amax(dim=1) <= .50) & ~closing
             & (coverage >= .55) & (nearest >= .70) & (nearest <= 1.35))
    candidates = ready.nonzero().flatten()
    clear = torch.zeros_like(ready)
    if not len(candidates):
        return clear
    seed_fit = (ready[candidates], seed[candidates], elapsed[candidates])
    supported, _, times, shapes = track_nearby_return(lidar[candidates], state[candidates], rays,
        curved=True, maximum_previous_command=.46, seeded=seed_fit, fit_full_center=True)
    centers = torch.stack([frame[0] for frame in shapes], dim=1)
    outward = centers.norm(dim=2).diff(dim=1)/times.diff(dim=1)
    center, sphere_radius, _ = shapes[-1]
    latest = ranges[candidates, -1].flatten(1)
    distance = (rays[None]*latest[..., None]-center[:, None]).norm(dim=2)
    associated = (distance-sphere_radius[:, None]).abs() <= .018
    other_nearby = (reliable[candidates, -1].flatten(1) & (latest < .85) & ~associated).any(dim=1)
    offsets, coast = (value[candidates] for value in command_response)
    alignment = ages[candidates, -1:]-ages[candidates]
    offsets = offsets+alignment[:, :, None]*coast[:, None]
    planar = [(center[:, :2], radius, valid) for center, radius, valid in shapes]
    tracked, position, motion, radius = _curved_motion(planar, times, turn_fallback=False, offsets=offsets)
    speed = motion.norm(dim=1)
    previous = history[candidates, -1, 9:11]
    normal = torch.stack((-motion[:, 1], motion[:, 0]), dim=1)/speed.clamp_min(.001)[:, None]
    across = (position*normal).sum(dim=1)
    coast_across = .35*(coast*normal).sum(dim=1)
    pending_across = .10*(previous*normal).sum(dim=1)
    # Bound both partial displacements before settling, even if the mover reverses along its line.
    lower = across-coast_across.clamp_min(0.)-pending_across.clamp_min(0.)
    upper = across-coast_across.clamp_max(0.)-pending_across.clamp_max(0.)
    least = torch.maximum(lower, -upper).clamp_min(0.)-radius
    clear[candidates] = (supported & tracked & ~other_nearby & (outward >= .08).all(dim=1)
                         & (speed >= .20) & (speed <= 1.5) & (least >= .65))
    return clear


def clear_fitted_crossing_at_stop(lidar, state, rays, command_response, settled, emergency_m):
    """Verify the stop on the fitted mover's line and check other reliable returns."""
    histories = state.reshape(-1, 4, 13)
    ages = histories[:, :, 12]
    accepted, _, _ = _return_seed(lidar, state)
    accepted &= (ages <= .04).all(dim=1) & (histories[:, :, :3].norm(dim=2).amax(dim=1) <= .22)
    lanes = accepted.nonzero().flatten()
    clear = torch.zeros_like(accepted)
    if not len(lanes):
        return clear

    scans, history, ages = lidar[lanes], state[lanes], ages[lanes]
    offsets, coast = (tensor[lanes] for tensor in command_response)
    offsets = offsets+(ages[:, -1:]-ages)[:, :, None]*coast[:, None]
    supported, _, times, shapes = track_nearby_return(scans, history, rays, curved=True,
                                                     maximum_previous_command=.46, fit_full_center=True)
    planar = [(center[:, :2], radius, valid) for center, radius, valid in shapes]
    tracked, position, motion, radius = _curved_motion(planar, times, turn_fallback=True, offsets=offsets)
    speed = motion.norm(dim=1).clamp_min(.001)
    normal = torch.stack((-motion[:, 1], motion[:, 0]), dim=1)/speed[:, None]
    across = (position*normal).sum(dim=1)
    direction = across.sign()
    drift = (direction*(coast*normal).sum(dim=1)).clamp_min(0.)
    pending = (direction*(history[:, 48:50]*normal).sum(dim=1)).clamp_min(0.)
    clearance = across.abs()-.35*drift-.1*pending-radius

    center, surface_radius, _ = shapes[-1]
    nearest, others, supported_surface = fitted_sphere_returns(scans, rays, center, surface_radius)
    safe = supported & tracked & (speed >= .20) & (speed <= 1.5) & ((position*(motion-coast)).sum(dim=1) < 0.)
    safe &= (others >= .85) & others.isfinite() & (nearest >= emergency_m) & supported_surface
    safe &= clearance >= torch.where(settled[lanes], .45, .65)
    strict = (safe & settled[lanes]).nonzero().flatten()
    if len(strict):
        safe[strict] = clear_hover_passage(scans[strict], history[strict], rays, curved=True,
                                         require_curvature=True, turn_fallback=True)
    clear[lanes] = safe
    return clear


def fitted_sphere_returns(scans, rays, center, surface_radius):
    """Find reliable ranges on the fitted sphere and on separately observed surfaces."""
    projection = center @ rays.T
    discriminant = surface_radius[:, None].square()-center.square().sum(dim=1, keepdim=True)+projection.square()
    first_hit = projection-discriminant.clamp_min(0.).sqrt()
    depth = scans[:, 9].flatten(1)
    reliable = ((scans[:, 10] > .5) & (scans[:, 11] > .55)).flatten(1)
    on_mover = (discriminant >= 0.) & (first_hit > 0.) & ((depth-first_hit).abs() <= .018)
    others = depth.masked_fill(~reliable | on_mover, torch.inf).amin(dim=1)
    nearest = depth.masked_fill(~reliable, torch.inf).amin(dim=1)
    return nearest, others, (on_mover & reliable).any(dim=1)


def clear_opening_crossing_at_stop(lidar, state, rays, command_response, emergency_m):
    """Verify a receding return when the closing-seeded track can no longer see it."""
    history = state.reshape(-1, 4, 13)
    ages = history[:, :, 12]
    ranges = lidar[:, 0::3]
    valid = (lidar[:, 1::3] > .5) & (lidar[:, 2::3] > .55)
    times = ranges.new_tensor([-.3, -.2, -.1, 0.])-ages+ages[:, -1:]
    gaps = times.diff(dim=1)
    nearest, seed = ranges[:, -1].masked_fill(~valid[:, -1], torch.inf).flatten(1).min(dim=1)
    ready = (ages <= .04).all(dim=1) & ((gaps >= .07) & (gaps <= .13)).all(dim=1)
    ready &= (history[:, :, :3].norm(dim=2).amax(dim=1) <= .25)
    ready &= (nearest >= emergency_m) & (nearest <= 1.35)
    lanes = ready.nonzero().flatten()
    clear = torch.zeros_like(ready)
    if not len(lanes):
        return clear

    scans, ages, times = lidar[lanes], ages[lanes], times[lanes]
    offsets, coast = (tensor[lanes] for tensor in command_response)
    offsets = offsets+(ages[:, -1:]-ages)[:, :, None]*coast[:, None]
    seeded = (ready[lanes], seed[lanes], times)
    supported, _, _, shapes = track_nearby_return(scans, state[lanes], rays, curved=True,
        maximum_previous_command=.46, seeded=seeded, fit_full_center=True)
    centers = torch.stack([frame[0] for frame in shapes], dim=1)
    planar = [(center[:, :2], radius, valid) for center, radius, valid in shapes]
    tracked, _, motion, radius = _curved_motion(planar, times, turn_fallback=False, offsets=offsets)
    outward = centers.norm(dim=2).diff(dim=1)/gaps[lanes]
    aligned = centers[:, :, :2]+offsets
    segments = aligned.diff(dim=1)/gaps[lanes, :, None]
    directions = torch.cat((motion[:, None], segments), dim=1)
    speed = directions.norm(dim=2)
    directions = directions/speed.clamp_min(.001)[..., None]
    normals = torch.stack((-directions[..., 1], directions[..., 0]), dim=2)
    consistent = (directions[:, :1]*directions).sum(dim=2).amin(dim=1) >= .99
    across = (aligned[:, -1, None]*normals).sum(dim=2)
    consistent &= (across.sign() == across[:, :1].sign()).all(dim=1)
    coast_toward = (across.sign()*(coast[:, None]*normals).sum(dim=2)).clamp_min(0.)
    queued_toward = (across.sign()*(state[lanes, None, 48:50]*normals).sum(dim=2)).clamp_min(0.)
    # The entire fitted line must stay clear, including a change in travel direction.
    # .50 m includes the .21 m body, .15 m clearance, and .14 m fit reserve.
    line = across.abs()-.35*coast_toward-.1*queued_toward-radius[:, None]
    center, surface_radius, _ = shapes[-1]
    nearest, others, supported_surface = fitted_sphere_returns(scans, rays, center, surface_radius)
    clear[lanes] = (supported & tracked & consistent & ((speed >= .20) & (speed <= 1.5)).all(dim=1)
                    & (outward[:, -2:] >= .06).all(dim=1) & (line >= .50).all(dim=1)
                    & (others >= .85) & others.isfinite() & (others > nearest+.02) & supported_surface)
    return clear


def clear_reversing_crossing_at_hover(lidar, state, rays, command_response, mount_offset, emergency_m,
                                     previously_verified):
    """Bound clearance from every observed segment of a reversing mover's line."""
    history = state.reshape(-1, 4, 13)
    ages = history[:, :, 12]
    accepted, seeds, times = _return_seed(lidar, state)
    accepted &= (ages <= .04).all(dim=1) & (history[:, :, :3].norm(dim=2).amax(dim=1) <= .06)
    accepted &= (history[:, :, 6:9].norm(dim=2) <= 1.e-6).all(dim=1)
    accepted &= (history[:, :, 9:12].norm(dim=2) <= .02).all(dim=1)
    lanes = accepted.nonzero().flatten()
    clear = torch.zeros_like(accepted)
    if not len(lanes):
        return clear

    scans, local_history, local_times = lidar[lanes], state[lanes], times[lanes]
    seeded = (accepted[lanes], seeds[lanes], local_times)
    supported, _, _, shapes = track_nearby_return(scans, local_history, rays, curved=True,
        maximum_previous_command=.02, seeded=seeded, fit_full_center=True)
    centers, radii, valid = (torch.stack(parts, dim=1) for parts in zip(*shapes, strict=True))
    offsets, coast = (tensor[lanes] for tensor in command_response)
    offsets = offsets+(ages[lanes, -1:]-ages[lanes])[:, :, None]*coast[:, None]
    corrected = centers+F.pad(offsets, (0, 1))
    segments = corrected.diff(dim=1)/local_times.diff(dim=1)[..., None]
    speed = segments.norm(dim=2)
    axis = F.normalize(segments, dim=2)
    agreement = (axis*axis[:, -1:]).sum(dim=2)
    reversal = (agreement[:, 0] <= -.995) & (agreement[:, 1] >= .995)
    straight = (agreement[:, :2] >= .995).all(dim=1) & previously_verified[lanes]

    # Ranges start at the calibrated LiDAR, while the clearance envelope is
    # centered on the vehicle. Use the full perpendicular distance to the line.
    center_body = corrected[:, -1]+corrected.new_tensor(mount_offset)
    along = (center_body[:, None]*axis).sum(dim=2)
    lateral = center_body[:, None]-along[..., None]*axis
    radius = radii.amax(dim=1)
    drift = .35*coast.norm(dim=1)+.10*local_history[:, 48:51].norm(dim=1)
    separation = lateral.norm(dim=2).amin(dim=1)-radius-drift-.002
    nearest, others, on_mover = fitted_sphere_returns(scans, rays, centers[:, -1], radii[:, -1])
    clear[lanes] = (supported & valid.all(dim=1) & (reversal | straight) & ((speed >= .20) & (speed <= 1.5)).all(dim=1)
                    & (along[:, -1] < 0.) & (radius-radii.amin(dim=1) <= .04) & (separation >= .45)
                    & (nearest >= emergency_m) & (others >= .85) & others.isfinite() & on_mover)
    return clear


def _fit_curvature(points, patch, *, full_center=False):
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
    return (centers if full_center else centers[:, :2]), radius, supported


def _return_seed(lidar, state, *, minimum_range_m=0., maximum_range_m=1.9):
    histories = state.reshape(-1, 4, 13)
    ranges = lidar[:, 0::3]
    reliable = (lidar[:, 1::3] > .5) & (lidar[:, 2::3] > .55)
    elapsed = ranges.new_tensor([-.3, -.2, -.1, 0.]) - histories[:, :, 12] + histories[:, -1:, 12]
    gap = elapsed.diff(dim=1)
    accepted, seed = _closing_seed(ranges, reliable, elapsed, minimum_range_m=minimum_range_m,
                                   maximum_range_m=maximum_range_m)
    accepted &= ((gap >= .065) & (gap <= .135)).all(dim=1)
    return accepted, seed, elapsed


def track_nearby_return(lidar, state, rays, *, curved=False, maximum_previous_command=.08, seeded=None,
                        fit_full_center=False):
    """Associate a supported closing return across four delivered scans."""
    histories = state.reshape(-1, 4, 13)
    ranges = lidar[:, 0::3]
    reliable = (lidar[:, 1::3] > .5) & (lidar[:, 2::3] > .55)
    accepted, seed, elapsed = _return_seed(lidar, state) if seeded is None else seeded
    gap = elapsed.diff(dim=1)
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
            shapes.append(_fit_curvature(points, patch, full_center=fit_full_center))
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
    fit, closest = _closest_at_motion(now, velocity)
    return fit & (rms <= .02), closest


def _closest_at_motion(now, velocity):
    speed_sq = velocity.square().sum(dim=1)
    moment = -(now*velocity).sum(dim=1)/speed_sq.clamp_min(.001)
    closest = (now+moment.clamp(0., 5.)[:, None]*velocity).norm(dim=1)
    fit = (speed_sq >= .04) & (speed_sq <= 2.25) & (moment > 0.)
    return fit, closest


def clear_hover_passage(lidar, state, rays, *, curved=False, require_curvature=False, turn_fallback=False):
    """Recognize a clear pass before the vehicle leaves its stopped position."""
    accepted, path, elapsed, shapes = track_nearby_return(lidar, state, rays, curved=curved)
    fit, closest = _closest_passage(path, elapsed)
    clear = fit & (closest >= .85)
    if curved:
        centers, radii, viable = (torch.stack(parts, dim=1) for parts in zip(*shapes, strict=True))
        fitted, center_distance = _closest_passage(centers, elapsed)
        outer = radii.amax(dim=1)
        shape = viable.all(dim=1) & ((outer-radii.amin(dim=1)) <= .04)
        supported = shape & fitted
        # Allocate .21 m to the vehicle and .15 m to clearance, with a .09 m observation margin.
        clear = torch.where(supported, center_distance-outer >= .45, clear)
        if require_curvature:
            clear &= supported
            if turn_fallback:
                reversing, latest = _latest_turn(centers, elapsed)
                _, _, error = _fit_motion(centers, elapsed)
                valid, latest_distance = _closest_at_motion(centers[:, -1], latest)
                clear |= shape & reversing & valid & (error > .020) & (error <= .055) & (latest_distance-outer >= .45)
    return accepted & clear


def _latest_turn(centers, elapsed):
    velocity = centers.diff(dim=1)/elapsed.diff(dim=1)[:, :, None].clamp_min(.001)
    speed = velocity.norm(dim=2)
    agreement = (velocity[:, 1]*velocity[:, 2]).sum(dim=1)/(speed[:, 1]*speed[:, 2]).clamp_min(.001)
    reversal = (velocity[:, 0]*velocity[:, 2]).sum(dim=1)/(speed[:, 0]*speed[:, 2]).clamp_min(.001)
    valid = (reversal <= -.5) & (agreement >= .85) & (speed >= .2).all(dim=1) & (speed <= 1.5).all(dim=1)
    return valid, .3*velocity[:, 1]+.7*velocity[:, 2]


def _curved_motion(shapes, elapsed, *, turn_fallback, offsets=None):
    centers, radii, viable = (torch.stack(parts, dim=1) for parts in zip(*shapes, strict=True))
    if offsets is not None:
        centers = centers+offsets
    position, relative, error = _fit_motion(centers, elapsed)
    fit = error <= .02
    if turn_fallback:
        reversing, latest = _latest_turn(centers, elapsed)
        use_latest = reversing & ~fit & (error <= .055)
        position = torch.where(use_latest[:, None], centers[:, -1], position)
        relative = torch.where(use_latest[:, None], latest, relative)
        fit |= use_latest
    radius = radii.amax(dim=1)
    valid = viable.all(dim=1) & fit & ((radius-radii.amin(dim=1)) <= .04)
    return valid, position, relative, radius


def hover_escape_scores(lidar, state, rays, issued, *, turn_fallback=False):
    """Compare the held escape directions and zero across a fitted curved return."""
    supported, _, elapsed, shapes = track_nearby_return(lidar, state, rays, curved=True, maximum_previous_command=.30001)
    valid, position, relative, radius = _curved_motion(shapes, elapsed, turn_fallback=turn_fallback)
    previous = state.reshape(-1, 4, 13)[:, :, 9:12]
    stable = (previous-previous[:, -1:]).norm(dim=2).amax(dim=1) <= .035
    last = previous[:, -1, :2]
    motion = relative+last
    supported &= stable & valid
    supported &= (motion.norm(dim=1) >= .2) & (motion.norm(dim=1) <= 1.5) & ((position*relative).sum(dim=1) < 0.)
    time = torch.arange(81, device=position.device, dtype=position.dtype)*.05
    hold = time.clamp_max(2.5)
    lag = .35
    coast = lag*(1.-torch.exp(-time/lag))
    travel = hold-lag*(1.-torch.exp(-hold/lag))*torch.exp(-(time-hold)/lag)
    predicted = position[:, None]+time[None, :, None]*motion[:, None]-coast[None, :, None]*last[:, None]
    separation = (predicted[:, None]-travel[None, None, :, None]*issued[:, :, None, :2]).norm(dim=-1)
    return supported, separation.amin(dim=2)-radius[:, None]


def advance_command_response(offsets, coast, state):
    """Advance a delayed request model and keep displacements at the four scan times."""
    import math

    def rotate(v, angle):
        x, y = v.unbind(dim=-1)
        return torch.stack((x*angle.cos()-y*angle.sin(), x*angle.sin()+y*angle.cos()), dim=-1)

    yaw = -.05*(state[:, 28]+state[:, 41])
    previous_yaw = -.05*(state[:, 15]+state[:, 28])
    coast = rotate(coast, yaw)
    delayed = rotate(state[:, 35:37], yaw+previous_yaw)
    decay = math.exp(-.1/.35)
    travel = .1*delayed+.35*(1.-decay)*(coast-delayed)
    earlier = rotate(offsets[:, 1:], yaw[:, None])-travel[:, None]
    return torch.cat((earlier, torch.zeros_like(earlier[:, :1])), dim=1), delayed+decay*(coast-delayed)


def moving_intercepts(lidar, state, rays, eligible, *, command_response=None, minimum_range_m=0., maximum_range_m=1.9):
    """Return approaching curved tracks supported by the four delivered scans."""
    accepted, seed, elapsed = _return_seed(lidar, state, minimum_range_m=minimum_range_m,
                                            maximum_range_m=maximum_range_m)
    lanes = (accepted & eligible).nonzero().flatten()
    if not len(lanes):
        return lanes, None, None, None, None
    history = state[lanes]
    seeded = (accepted[lanes], seed[lanes], elapsed[lanes])
    supported, _, elapsed, shapes = track_nearby_return(lidar[lanes], history, rays, curved=True,
                                                       maximum_previous_command=.46, seeded=seeded)
    history = history.reshape(-1, 4, 13)
    if command_response is not None:
        offsets, coast = (tensor[lanes] for tensor in command_response)
        age = history[:, -1:, 12]-history[:, :, 12]
        offsets = offsets+age[:, :, None]*coast[:, None]
        valid, position, motion, radius = _curved_motion(shapes, elapsed, turn_fallback=True, offsets=offsets)
        relative = motion-coast
    else:
        valid, position, relative, radius = _curved_motion(shapes, elapsed, turn_fallback=True)
        past = history[:, :, 9:11]
        coast = (past*past.new_tensor([.424, .140, .187, .249])[None, :, None]).sum(dim=1)
        motion = relative+coast
    speed = motion.norm(dim=1)
    keep = supported & valid & (speed >= .2) & (speed <= 1.5) & ((position*relative).sum(dim=1) < 0.)
    return lanes[keep], position[keep], motion[keep], coast[keep], radius[keep]


def intercept_scores(position, motion, coast, radius, issued, *, hold_s=None, preview_steps=None):
    """Estimate closest surface separation under each admissible request."""
    # Use the 2.4 s braking preview for the finite-hold comparison.
    samples = preview_steps if preview_steps is not None else (49 if hold_s is not None else 61)
    time = torch.arange(samples, device=position.device, dtype=position.dtype)*.05
    lag = .35*(1.-torch.exp(-time/.35))
    relative = position[:, None]+time[None, :, None]*motion[:, None]-lag[None, :, None]*coast[:, None]
    travel = time-lag
    if hold_s is not None:
        hold = time.clamp_max(hold_s)
        travel = hold-.35*(1.-torch.exp(-hold/.35))*torch.exp(-(time-hold)/.35)
    paths = relative[:, None]-travel[None, None, :, None]*issued[:, :, None, :2]
    return paths.norm(dim=-1).amin(dim=2)-radius[:, None]


def observed_path_clearance(lidar, rays, coast, issued, *, hold_s=None):
    """Measure distance from a delayed request path to the currently observed returns."""
    depth = lidar[:, 9].flatten(1)
    valid = ((lidar[:, 10] > .5) & (lidar[:, 11] > .55)).flatten(1)
    points = depth[..., None]*rays
    time = issued.new_tensor([0., .4, 1.2, 2.4] if hold_s is None else [0., .4, 1.2, hold_s, hold_s+1.4])
    lag = .35*(1.-torch.exp(-time/.35))
    coast = torch.nn.functional.pad(coast, (0, 1))
    travel = time-lag
    if hold_s is not None:
        hold = time.clamp_max(hold_s)
        travel = hold-.35*(1.-torch.exp(-hold/.35))*torch.exp(-(time-hold)/.35)
    paths = lag[None, None, :, None]*coast[:, None, None]+travel[None, None, :, None]*issued[:, :, None]
    minimum = issued.new_full(issued.shape[:2], torch.inf)
    for start, end in zip(paths[:, :, :-1].unbind(dim=2), paths[:, :, 1:].unbind(dim=2), strict=True):
        segment = end-start
        offset = points[:, None]-start[:, :, None]
        projection = (offset*segment[:, :, None]).sum(dim=3, keepdim=True)
        fraction = (projection/segment.square().sum(dim=2)[:, :, None, None].clamp_min(1.e-6)).clamp(0., 1.)
        distance = (offset-fraction*segment[:, :, None]).norm(dim=3)
        minimum = torch.minimum(minimum, distance.masked_fill(~valid[:, None], torch.inf).amin(dim=2))
    return minimum


def crossing_stop_components(lidar, rays, position, motion, coast, radius, issued, *, hold_s):
    """Bound a finite escape and the eventual stationary separation from the crossing line."""
    swept = intercept_scores(position, motion, coast, radius, issued, hold_s=hold_s, preview_steps=101)
    forward = motion/motion.norm(dim=1, keepdim=True)
    across = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
    resting_offset = ((position-.35*coast)*across).sum(dim=1)
    settled = resting_offset[:, None]-hold_s*(issued[..., :2]*across[:, None]).sum(dim=2)
    line = resting_offset.sign()[:, None]*settled-radius[:, None]
    observed = observed_path_clearance(lidar, rays, coast, issued, hold_s=hold_s)
    return swept, torch.minimum(line, observed)


def crossing_stop_clearance(lidar, rays, position, motion, coast, radius, issued, *, hold_s):
    swept, settled = crossing_stop_components(lidar, rays, position, motion, coast, radius, issued, hold_s=hold_s)
    return torch.minimum(swept, settled)


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
