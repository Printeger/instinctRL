"""Recheck a new tangential request immediately after an observed wall stop."""

import math

import numpy as np
import torch

from .escape_passage import _return_seed, moving_intercepts, rotate_scan_rays
from .local_wall_stopping import _align_scans, _braking_capsule, _local_planes, _opposite_wall_pair


def tangential_wall_margin(lidar, state, rays, actor, command_response, oldest_speed_bound):
    history = state.reshape(4, 13)
    request = history[-1, 6:9]
    speed = float(request.norm())
    velocity = actor.v_final_b[0]
    age = history[:, 12]
    if (not .08 <= speed <= .30 or bool(actor.metrics['emergency'][0, 0])
            or float(actor.metrics['beta'][0, 0]) < .90 or float(velocity @ request) < .66*speed**2
            or bool((age > .04).any() or (age.diff().abs() > .03).any())
            or float(history[:, :3].norm(dim=1).max()) > .18):
        return 0.

    previous = history[-2, 6:9]
    previous_speed = float(previous.norm())
    if not .08 <= previous_speed <= .30:
        return 0.
    delta = -.5*(.10-(age[-1]-age[-2]))*(history[-2, :3]+history[-1, :3])
    toward_wall = rotate_scan_rays(previous[None], delta)[0]/previous_speed
    if float(toward_wall @ request) > .25*speed:
        return 0.

    forward = (rays @ (request/speed)) >= .70
    seen = ((lidar[:, 1::3] > .5) & (lidar[:, 2::3] > .55)).all(dim=1).flatten()
    if float(seen.float().mean()) < .50 or not bool(forward.any()) or float(seen[forward].float().mean()) < .45:
        return 0.
    closing, _, _ = _return_seed(lidar, state, maximum_range_m=2.65)
    if bool(closing[0]):
        return 0.
    moving, *_ = moving_intercepts(lidar, state, rays, torch.ones(1, dtype=torch.bool, device=state.device),
                                   command_response=command_response, maximum_range_m=2.65)
    if len(moving):
        return 0.

    up = -history[-1, 3:6].detach().cpu().numpy()
    if abs(up[1]) > .80:
        return 0.
    velocity_magnitude = float(velocity.norm())
    heading = velocity.detach().cpu().numpy()/velocity_magnitude
    scans = lidar.reshape(4, 3, -1)[-3:].detach().cpu().numpy()
    aligned, mounts, prior = _align_scans(rays, history)
    valid = (scans[:, 1] > .5) & (scans[:, 2] >= .65)
    points = scans[:, 0, :, None]*aligned+mounts
    tail = .35*max(.30, oldest_speed_bound)*math.exp(-.30/.35)
    _, _, required, _, gap = _braking_capsule(points, valid, prior, heading, .60*velocity_magnitude, tail)
    if not math.isfinite(gap) or gap <= required:
        return 0.
    approach = toward_wall.detach().cpu().numpy()
    requested = request.detach().cpu().numpy()/speed
    planes = _local_planes(scans, history, rays.detach().cpu().numpy(), points, valid, up)
    sides = [(normal, distance) for normal, distance, associated in planes
             if np.count_nonzero(associated) >= 24 and abs(normal @ heading) <= .25
             and abs(normal @ requested) <= .25]
    for index, (normal, distance) in enumerate(sides):
        for other, other_distance in sides[index+1:]:
            if (_opposite_wall_pair(normal, distance, other, other_distance)
                    and max(normal @ approach, other @ approach) >= .70):
                return gap-required
    return 0.
