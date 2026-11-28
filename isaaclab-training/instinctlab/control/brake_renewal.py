"""Reassociate the same observed mover during a verified outward brake release."""

import torch
from torch.nn import functional as F

from .escape_passage import (
    _fit_motion, _return_seed, fitted_sphere_returns, rotate_scan_rays, track_nearby_return,
)


def crossing_side(position, motion):
    along = F.normalize(motion, dim=1)
    across = torch.stack((-along[:, 1], along[:, 0]), dim=1)
    return (position*across).sum(dim=1)


def _crossing_shapes(lidar, state, rays, *, require_nearest):
    ranges = lidar[:, 0::3]
    reliable = (lidar[:, 1::3] > .5) & (lidar[:, 2::3] > .55)
    closing, seed, times = _return_seed(lidar, state, maximum_range_m=2.65)
    latest = ranges[:, -1].flatten(1)
    nearest = latest.masked_fill(~reliable[:, -1].flatten(1), torch.inf).argmin(dim=1)
    seed = torch.where(closing, seed, nearest)
    distance = latest.gather(1, seed[:, None])[:, 0]
    gaps = times.diff(dim=1)
    ages = state.reshape(-1, 4, 13)[:, :, 12]
    coverage = reliable.all(dim=1).flatten(1).float().mean(dim=1)
    ready = ((ages <= .04).all(dim=1) & ((gaps >= .07) & (gaps <= .13)).all(dim=1)
             & (distance >= .32) & (distance <= 2.65) & (coverage >= .55)
             & reliable[:, -1].flatten(1).gather(1, seed[:, None])[:, 0])
    if require_nearest:
        ready &= ~closing
    lanes = ready.nonzero().flatten()
    if not len(lanes):
        return None

    times = times[lanes]
    seeded = (ready[lanes], seed[lanes], times)
    supported, _, _, shapes = track_nearby_return(lidar[lanes], state[lanes], rays, curved=True,
        maximum_previous_command=.46, seeded=seeded, fit_full_center=True)
    centers, radii, valid = (torch.stack(parts, dim=1) for parts in zip(*shapes, strict=True))
    _, other, on_mover = fitted_sphere_returns(lidar[lanes], rays, centers[:, -1], radii[:, -1])
    keep = (supported & valid.all(dim=1) & (radii.amax(dim=1)-radii.amin(dim=1) <= .04)
            & on_mover & (other >= .85) & other.isfinite())
    return lanes[keep], centers[keep], radii[keep], times[keep]


def remember_crossing(lidar, state, rays, position, motion):
    """Store the latest observed center, radius, travel vector and side for one recheck."""
    observed = _crossing_shapes(lidar, state, rays, require_nearest=False)
    if observed is None:
        return None
    lanes, centers, radii, _ = observed
    side = crossing_side(position[lanes], motion[lanes])
    return lanes, torch.cat((centers[:, -1], radii[:, -1:], motion[lanes], side[:, None]), dim=1)


def renew_crossing(lidar, state, rays, offsets, coast, anchor):
    """Return a fresh fit only when it matches the anchor on the shared sweep."""
    observed = _crossing_shapes(lidar, state, rays, require_nearest=True)
    if observed is None:
        return None
    lanes, centers, radii, times = observed
    history = state[lanes].reshape(-1, 4, 13)
    dt = times[:, -1]-times[:, -2]
    rotation = -.5*dt[:, None]*(history[:, -2, :3]+history[:, -1, :3])
    prior_center, prior_radius, prior_motion, prior_side = anchor[lanes].split((3, 1, 2, 1), dim=1)
    aligned_center = rotate_scan_rays(prior_center[:, None], rotation)[:, 0]
    identity = ((centers[:, -2]-aligned_center).norm(dim=1) <= .025) & ((radii[:, -2]-prior_radius[:, 0]).abs() <= .02)
    coast = coast[lanes]
    ages = history[:, :, 12]
    aligned = centers[:, :, :2]+offsets[lanes]+(ages[:, -1:]-ages)[:, :, None]*coast[:, None]
    position, motion, error = _fit_motion(aligned, times)
    latest_position, latest_motion, latest_error = _fit_motion(aligned[:, 1:], times[:, 1:])
    old_motion = rotate_scan_rays(F.pad(prior_motion, (0, 1))[:, None], rotation)[:, 0, :2]
    speed = torch.stack((motion.norm(dim=1), latest_motion.norm(dim=1)), dim=1)
    side = crossing_side(position, motion)
    consistent = ((speed >= .20) & (speed <= 1.5)).all(dim=1)
    consistent &= ((torch.maximum(error, latest_error) <= .02)
                   & ((position-latest_position).norm(dim=1) <= .04)
                   & (F.cosine_similarity(motion, latest_motion) >= .985)
                   & (F.cosine_similarity(motion, old_motion) >= .985))
    keep = identity & consistent & (side.abs() >= .36) & (side*prior_side[:, 0] > 0.)
    return lanes[keep], position[keep], motion[keep], coast[keep], radii[keep].amax(dim=1)
