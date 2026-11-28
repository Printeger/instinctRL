"""Recheck a new oblique retreat when the initiating returns are stationary."""

import numpy as np
import torch

from .escape_passage import (
    _return_seed, _supported_closing_returns, clear_static_scan, moving_intercepts, observed_path_clearance,
)
from .local_wall_stopping import _align_scans, _local_planes


def static_turn_margin(detector, lidar, state, current, retreat, command_response):
    history = state.reshape(4, 13)
    previous, latest = history[-2:, 6:9]
    speed, previous_speed = float(latest.norm()), float(previous.norm())
    age = history[:, 12]
    velocity = current.v_final_b[0]
    if (not .08 <= speed <= .30 or previous_speed < .08 or float(latest @ previous) > .25*speed*previous_speed
            or float((latest-previous).norm()) < .12 or bool(((age < 0.) | (age > .04)).any())
            or float(age.diff().abs().max()) > .03 or float(history[:, :3].norm(dim=1).max()) >= .22
            or float(current.metrics['beta'][0, 0]) < .90
            or bool(current.metrics['emergency'][0, 0] or retreat.metrics['emergency'][0, 0])
            or float(velocity @ latest) < .66*speed**2):
        return 0.

    rays = detector._full_rays
    reliable = (lidar[:, 1::3] > .5) & (lidar[:, 2::3] > .55)
    support = reliable.flatten(2).all(dim=1)[0]
    forward = (rays @ (latest/speed)) >= .70
    if float((support & forward).sum()/forward.sum().clamp_min(1)) < .45:
        return 0.
    _, coast = command_response
    path = observed_path_clearance(lidar, rays, torch.zeros_like(coast), velocity[None, None], hold_s=2.4)
    reserve = .35*float(coast.norm())+.10*float(history[-1, 9:12].norm())+.058
    path_floor = float(path[0, 0])-reserve
    if path_floor < .43 or not bool(clear_static_scan(lidar, state, rays, command_response)[0]):
        return 0.
    approaching, *_ = moving_intercepts(lidar, state, rays, torch.ones(1, dtype=torch.bool, device=state.device),
                                        command_response=command_response, maximum_range_m=2.65)
    if len(approaching):
        return 0.

    _, _, elapsed = _return_seed(lidar, state, maximum_range_m=2.65)
    closing = _supported_closing_returns(lidar[:, 0::3], reliable, elapsed, maximum_range_m=2.65)[0] > 0.
    triggers = torch.zeros_like(reliable[0, 0])
    triggers[::3] = detector._closing_weights(lidar, history[None])[0] > 0.
    if not bool(triggers.any()):
        return 0.
    sources = (closing | triggers.flatten()).detach().cpu().numpy()
    scans = lidar.reshape(4, 3, -1)[-3:].detach().cpu().numpy()
    valid = reliable.flatten(2)[0, -3:].detach().cpu().numpy()
    aligned, mount, queued = _align_scans(rays, history)
    points = scans[:, 0, :, None]*aligned+mount
    up = -history[-1, 3:6].detach().cpu().numpy()
    if abs(up[1]) > .80:
        return 0.
    planes = _local_planes(scans, history, rays.detach().cpu().numpy(), points, valid, up)
    issued = velocity.detach().cpu().numpy()
    bound = np.zeros(len(rays), dtype=float)
    for normal, distance, associated in planes:
        signed = distance-.35*max(0., float((queued @ normal).max()))-.10*max(0., float(queued[-1] @ normal))
        signed -= .058+2.4*max(0., float(issued @ normal))
        if signed >= .43:
            bound[associated] = np.maximum(bound[associated], signed)
    floor = min(path_floor, float(bound[sources].min()))
    return floor if floor >= .43 else 0.
