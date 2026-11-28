"""Check an early, capped response to a curved lidar track near a stop."""

import math

import numpy as np
import torch

from .escape_passage import _return_seed, observed_path_clearance, rotate_scan_rays
from .measured_curve_deflection import _Fit, _LAGS_S, _return_shapes
from .observed_mover_preview import _rank_continuations, _response
from .tube_stopping import _ANGLE_BINS, _MOUNT, _surface_normals


_RANGE_LIMIT = 2.65
_FIRST_CHANGE = .30
_FLOOR = .39
_DISTANCE_ERROR = .001
_BACKUP_TIME = .90


def _rotations(history, times):
    gravity = -history[:, 3:6]
    gravity = gravity/gravity.norm(dim=1, keepdim=True)
    identity = torch.eye(3, dtype=history.dtype, device=history.device)
    rotations = [identity]
    for frame in range(2, -1, -1):
        gyro = .5*(history[frame, :3]+history[frame+1, :3])
        local = rotate_scan_rays(identity, -gyro*(times[frame+1]-times[frame]))
        predicted = gravity[frame]@local
        axis = torch.cross(predicted, gravity[frame+1], dim=0)
        sine = axis.norm()
        tilt = torch.atan2(sine, predicted@gravity[frame+1])
        rotations.append(rotate_scan_rays(local, axis*tilt/sine.clamp_min(1.e-8))@rotations[-1])
    return torch.stack(rotations[::-1]).cpu().numpy()


def _patch_history(samples, normal, tangent, up, distance):
    signed = []
    split = []
    extents = []
    angles = []
    counts = []
    for points, normals in samples:
        offset = points@normal
        near = ((normals@normal >= math.cos(.045)) & (abs(offset-distance) < .12))
        if near.sum() < 24:
            return None
        median = float(np.median(offset[near]))
        matched = near & (abs(offset-median) <= .013)
        patch = points[matched]
        along = patch@tangent
        if len(patch) < 24 or np.ptp(along) < .14 or np.ptp(patch@up) < .20:
            return None
        residual = patch@normal-median
        if np.sqrt(np.mean(residual**2)) > .006:
            return None
        first = along <= np.median(along)
        signed.append(median)
        split.append((median+np.median(residual[first]),
                      median+np.median(residual[~first]) if (~first).any() else median))
        extents.append((along.min(), along.max()))
        direction = normals[matched].mean(axis=0)
        angles.append(math.atan2(direction@tangent, direction@normal))
        counts.append(len(patch))
    extent = np.asarray(extents)
    if np.max(abs(np.diff(signed))) > .045 or extent[:, 1].min()-extent[:, 0].max() < .10:
        return None
    return np.asarray(signed), np.asarray(split), np.asarray(angles), min(counts)


def _observed_planes(lidar, rays, rotations, basis, up):
    samples = []
    mount = np.asarray(_MOUNT)
    for scan, rotation in zip(lidar.reshape(4, 3, -1), rotations, strict=True):
        points, normals = _surface_normals(scan, rays)
        normals = normals@rotation
        use = (np.linalg.norm(points, axis=1) <= _RANGE_LIMIT) & (abs(normals@up) < .07)
        samples.append(((points[use]+mount)@rotation, normals[use]))
    points, normals = samples[-1]
    angles = np.arctan2(normals@basis[:, 1], normals@basis[:, 0])
    unused = np.ones(len(points), dtype=bool)
    distance_edges = np.arange(.20, _RANGE_LIMIT+.051, .02)
    planes = []
    for _ in range(24):
        counts, _ = np.histogram(angles[unused], bins=_ANGLE_BINS)
        peak = counts.argmax()
        if counts[peak] < 20:
            break
        angle = .5*(_ANGLE_BINS[peak]+_ANGLE_BINS[peak+1])
        near = unused & (abs((angles-angle+np.pi)%(2.*np.pi)-np.pi) <= .045)
        unused &= ~near
        normal = normals[near].mean(axis=0)
        normal -= up*(normal@up)
        normal /= np.linalg.norm(normal)
        tangent = np.cross(up, normal)
        counts, _ = np.histogram(points[near]@normal, bins=distance_edges)
        for band in np.flatnonzero(counts >= 18):
            estimate = .5*(distance_edges[band]+distance_edges[band+1])
            match = _patch_history(samples, normal, tangent, up, estimate)
            if match is None:
                continue
            if any(normal@plane[0] > .999 and abs(match[0][-1]-plane[1][-1]) < .05 for plane in planes):
                continue
            planes.append((normal, *match))
    return planes


def _stationary_translation(lidar, rays, rotations, basis, up):
    planes = _observed_planes(lidar, rays, rotations, basis, up)
    if len(planes) < 3:
        return None
    normals, distances, split, _, _ = (np.asarray(part) for part in zip(*planes, strict=True))
    directions = normals@basis
    if np.linalg.svd(directions, compute_uv=False)[-1] < .35:
        return None
    measured = distances[:, -1:]-distances
    translation = np.linalg.lstsq(directions, measured, rcond=None)[0].T
    if np.linalg.norm(directions@translation.T-measured, axis=0).max() > _DISTANCE_ERROR:
        return None
    for half in split.transpose(2, 0, 1):
        independent = np.linalg.lstsq(directions, half[:, -1:]-half, rcond=None)[0].T
        if np.linalg.norm(independent-translation, axis=1).max() > _DISTANCE_ERROR:
            return None
    return translation


def reprove_plane_alignment(lidar, state, rays):
    """Verify gyro transport with independent normals from two observed planes."""
    history = state.reshape(4, 13)
    if float(history[:, :3].norm(dim=1).max()) > .55:
        return None
    _, _, clock = _return_seed(lidar, state, maximum_range_m=_RANGE_LIMIT)
    clock = clock[0]
    gaps = clock.diff()
    if bool(((gaps < .065) | (gaps > .135)).any()):
        return None
    up = -history[-1, 3:6].cpu().numpy()
    up = up/np.linalg.norm(up)
    forward = np.array([1., 0., 0.])-up*up[0]
    forward /= np.linalg.norm(forward)
    basis = np.stack((forward, np.cross(up, forward)), axis=1)
    rotations = _rotations(history, clock)
    planes = _observed_planes(lidar.cpu().numpy(), rays.cpu().numpy(), rotations, basis, up)
    if len(planes) < 2 or any(plane[4] < 28 for plane in planes):
        return None
    normals = np.stack([plane[0] for plane in planes])@basis
    if np.linalg.svd(normals, compute_uv=False)[-1] < .50:
        return None
    angles = np.stack([plane[3] for plane in planes])
    error = float(np.max(abs(angles-angles[:, -1:])))
    return (max(.0015, error), len(planes), float(gaps.min())) if error <= .006 else None


def _registered_fit(lidar, state, rays, *, entry):
    shape = _return_shapes(lidar, state, rays, entry=entry, align_gravity=True)
    if shape is None:
        return None
    centers, radius, times, gaps = shape
    if entry:
        rate = centers.norm(dim=1).diff()/gaps
        decrease = rate[:-1]-rate[1:]
        if float(decrease.min()) < .024 or float(rate[-1]) > 1.5*float(decrease.min()):
            return None
    history = state.reshape(4, 13)
    rotation = _rotations(history, times)
    up = -history[-1, 3:6].cpu().numpy()
    up = up/np.linalg.norm(up)
    forward = np.array([1., 0., 0.])-up*up[0]
    forward /= np.linalg.norm(forward)
    basis = np.stack((forward, np.cross(up, forward)), axis=1)
    aligned = _stationary_translation(lidar.cpu().numpy(), rays.cpu().numpy(), rotation, basis, up)
    if aligned is None:
        return None
    delta = gaps.cpu().numpy()
    mount = np.asarray(_MOUNT)
    points = (centers.cpu().numpy()+mount@rotation-mount)@basis+aligned
    velocity = np.diff(points, axis=0)/delta[:, None]
    acceleration = np.diff(velocity, axis=0)/(.5*(delta[1:]+delta[:-1]))[:, None]
    speed = np.linalg.norm(velocity[-1])
    length = np.linalg.norm(acceleration, axis=1)
    if speed > 1.42 or length.max() > 1.05:
        return None
    if entry and (speed < .18 or length.min() < .32 or length.max() > .80
                  or acceleration[0]@acceleration[1] < .993*length.prod()
                  or np.linalg.norm(acceleration[1]-acceleration[0]) > .12):
        return None
    fit = _Fit(centers[-1].cpu().numpy(), float(radius), np.broadcast_to(velocity[-1]@basis.T, (3, 3)),
               np.broadcast_to(acceleration, (3, 2, 2)))
    return fit, basis, float(delta[-1])


def _piecewise_clearance(fit, coast, queued, first, following, hold, horizon, spacing):
    time = np.arange(round(horizon/.005)+1)*.005
    radius = fit.radius+.025+.05*time+.5*1.05*time**2
    # Bound the extra error in the latest secant from two registered origins.
    radius += 2.*_DISTANCE_ERROR*(1.+time/spacing)
    mover = fit.center+time[:, None]*fit.velocity[0]
    low = np.full(len(first), np.inf)
    for lag, drift in zip(_LAGS_S, coast, strict=True):
        for queue in (0., .05, .10):
            active = np.maximum(0., time-queue)
            elapsed = _response(time, lag)
            begin = _response(active, lag)
            drift_path = (time-elapsed)[:, None]*drift+(elapsed-begin)[:, None]*queued
            weight = begin-_response(np.maximum(0., active-.10), lag)
            total = begin-_response(np.maximum(0., active-hold), lag)
            path = drift_path+weight[None, :, None]*first[:, None]
            path += (total-weight)[None, :, None]*following[:, None]
            low = np.minimum(low, (np.linalg.norm(path-mover, axis=2)-radius).min(axis=1))
    return low


def registered_mover_entry(scan, state, selected, config, *, rays, coast, issued, action_for_velocity):
    request = state[0, 45:48]
    prior = issued(scan, state, selected, config)
    if bool(prior.metrics['emergency'][0, 0]) or float(prior.v_final_b[0]@request) >= .05*float(request@request):
        return None
    registered = _registered_fit(scan, state, rays, entry=True)
    if registered is None:
        return None
    fit, basis, spacing = registered
    planar_coast = (coast@basis[:2])@basis.T
    queued = state[0, 48:51].cpu().numpy()
    current = prior.v_final_b.cpu().numpy()
    previous = float(_piecewise_clearance(fit, planar_coast, queued, current, current, .60, 1., spacing)[0])
    projection = float(_rank_continuations(fit, coast, queued, current)[0])
    if previous >= .36 or projection >= .15:
        return None
    angle = np.arange(72)*np.pi/36.
    directions = np.stack((np.cos(angle), np.sin(angle)), axis=1)@basis.T
    target = state.new_tensor(np.concatenate((.2999*directions, .4499*directions)))
    actions = torch.cat((state.new_zeros((len(target), 1)), .5+target), dim=1)
    check = issued(scan.expand(len(actions), -1, -1, -1), state.expand(len(actions), -1), actions, config)
    valid = ((check.metrics['beta'][:, 0] >= .90) & (check.metrics['emergency'][:, 0] == 0.)
             & (check.v_final_b@request >= .60*(request@request)))
    if not bool(valid.any()):
        return None
    following_actions, follow = actions[valid], check.v_final_b[valid]
    delta = follow-state[:, 48:51]
    limited = state[:, 48:51]+delta*(_FIRST_CHANGE/delta.norm(dim=1, keepdim=True).clamp_min(_FIRST_CHANGE))
    actions = action_for_velocity(limited, state[:, 48:51].expand_as(limited), config.governor)
    check = issued(scan.expand(len(actions), -1, -1, -1), state.expand(len(actions), -1), actions, config)
    distance = (check.v_final_b-state[:, 48:51]).norm(dim=1)
    valid = ((check.metrics['beta'][:, 0] >= .90) & (check.metrics['emergency'][:, 0] == 0.)
             & (distance <= _FIRST_CHANGE+.00001))
    if not bool(valid.any()):
        return None
    actions, following_actions = actions[valid], following_actions[valid]
    follow, first = follow[valid], check.v_final_b[valid]
    floor = observed_path_clearance(scan, rays, state.new_tensor(coast[1:2]), follow[None], hold_s=.60)[0]
    static = (floor-.055-.10*(first-follow).norm(dim=1)).cpu().numpy()
    first, follow = first.cpu().numpy(), follow.cpu().numpy()
    planned = _piecewise_clearance(fit, planar_coast, queued, first, follow, .60, 1., spacing)
    backup = _piecewise_clearance(fit, planar_coast, queued, first, np.zeros_like(follow), .10, _BACKUP_TIME, spacing)
    ranking = _rank_continuations(fit, coast, queued, follow)
    floor = np.minimum(np.minimum(static, planned), backup)
    eligible = (floor >= _FLOOR) & (ranking >= max(.12, projection+.08))
    if not eligible.any():
        return None
    best = int(np.where(eligible, ranking, -np.inf).argmax())
    witness = (float(floor[best]), previous, projection, float(ranking[best]), float(backup[best]))
    return actions[best], following_actions[best], witness


def registered_mover_handoff(scan, state, selected, config, *, rays, coast, issued):
    """Recheck a current-directed issue before discarding the one-packet stop backup."""
    request = state[0, 45:48]
    prior = issued(scan, state, selected, config)
    if (float(request.norm()) < .03 or bool(prior.metrics['emergency'][0, 0])
            or float(prior.v_final_b[0]@request) < .60*float(request@request)):
        return False
    registered = _registered_fit(scan, state, rays, entry=False)
    if registered is None:
        return False
    fit, basis, spacing = registered
    planar_coast = (coast@basis[:2])@basis.T
    target = prior.v_final_b.cpu().numpy()
    queued = state[0, 48:51].cpu().numpy()
    dynamic = _piecewise_clearance(fit, planar_coast, queued, target, target, .50, .70, spacing)[0]
    return dynamic >= _FLOOR and _rank_continuations(fit, coast, queued, target)[0] >= .15


def reprove_registered_mover_follow(scan, state, action, count, config, *, rays, coast, issued):
    """Renew the original direction from fresh observations or keep the stop backup."""
    request = state[0, 45:48]
    if float(request.norm()) < .03:
        return None
    history = state.reshape(4, 13)
    delta = -.05*(history[-2, :3]+history[-1, :3])
    action = action.clone()
    direction = rotate_scan_rays((action[1:]-.5)[None], delta)[0]
    gravity = history[-1, 3:6]
    direction -= gravity*(direction@gravity)/gravity.square().sum()
    action[1:] = .5+direction
    result = issued(scan, state, action[None], config)
    velocity = result.v_final_b
    if (float(result.metrics['beta'][0, 0]) < .90 or bool(result.metrics['emergency'][0, 0])
            or float(velocity[0]@request) < .60*float(request@request)
            or float((velocity-state[:, 48:51]).norm()) > _FIRST_CHANGE+.00001):
        return None
    registered = _registered_fit(scan, state, rays, entry=False)
    if registered is None:
        return None
    fit, basis, spacing = registered
    planar = (coast@basis[:2])@basis.T
    queued, target = state[0, 48:51].cpu().numpy(), velocity.cpu().numpy()
    horizon, hold = max(.70, 1.-.10*count), max(.30, .60-.10*count)
    static = float(observed_path_clearance(scan, rays, state.new_tensor(coast[1:2]), velocity[None], hold_s=hold)[0, 0])-.055
    planned = float(_piecewise_clearance(fit, planar, queued, target, target, hold, horizon, spacing)[0])
    backup = float(_piecewise_clearance(fit, planar, queued, target, np.zeros_like(target), .10,
                                       .80 if count <= 2 else .60, spacing)[0])
    rank = float(_rank_continuations(fit, coast, queued, target)[0])
    floor = min(static, planned, backup)
    return (action, (floor, planned, 0., rank, backup)) if floor >= _FLOOR and rank >= .15 else None
