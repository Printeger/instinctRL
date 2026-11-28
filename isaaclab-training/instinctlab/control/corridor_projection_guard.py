"""Limit collective projection to a pair of observed continuous corridor walls."""

import numpy as np
import torch

from .local_wall_stopping import _align_scans, _local_planes, _opposite_wall_pair


_ALONG_WALL_BINS = np.linspace(-1.5, 1.5, 11)
_RECHECK_PACKETS = 10


def corridor_width(lidar, history, rays):
    up = -history[-1, 3:6].detach().cpu().numpy()
    if abs(up[1]) > .80:
        return 0.
    scans = lidar.reshape(4, 3, -1)[-3:].detach().cpu().numpy()
    aligned, mounts, _ = _align_scans(rays, history)
    valid = (scans[:, 1] > .5) & (scans[:, 2] >= .65)
    points = scans[:, 0, :, None]*aligned+mounts
    planes = _local_planes(scans, history, rays.detach().cpu().numpy(), points, valid, up)
    walls = []
    for normal, distance, associated in planes:
        if np.count_nonzero(associated) < 24:
            continue
        along = points[:, associated] @ np.cross(up, normal)
        bins = [np.histogram(frame, bins=_ALONG_WALL_BINS)[0] for frame in along]
        if np.ptp(along, axis=1).min() >= 3. and (np.asarray(bins) >= 3).all():
            walls.append((normal, distance))
    for index, (normal, distance) in enumerate(walls):
        for peer, peer_distance in walls[index+1:]:
            width = distance+peer_distance
            if width <= 2.26 and _opposite_wall_pair(normal, distance, peer, peer_distance):
                return width
    return 0.


class CorridorProjectionGuard:
    def __init__(self, reference):
        self.width = reference.new_zeros(len(reference))
        self.next_recheck = 0

    def reset(self, lanes):
        self.width[lanes] = 0.

    def update(self, lidar, state, rays):
        history = state.reshape(-1, 4, 13)
        ages = history[:, :, 12]
        fresh = (((ages >= 0.) & (ages <= .04)).all(dim=1)
                 & (ages.diff(dim=1).abs() <= .03).all(dim=1)
                 & (history[:, :, :3].norm(dim=2).amax(dim=1) <= .22))
        near = ((lidar[:, 9] <= 1.65) & (lidar[:, 10] > .5) & (lidar[:, 11] >= .65)).any(dim=1)
        eligible = fresh & ((near & near.roll(180, dims=1)).sum(dim=1) >= 12)
        self.width[~eligible] = 0.
        if self.next_recheck == 0:
            self.width.zero_()
            for lane in eligible.nonzero().flatten().tolist():
                self.width[lane] = corridor_width(lidar[lane], history[lane], rays)
        self.next_recheck = (self.next_recheck+1) % _RECHECK_PACKETS
        return self.width > 0.
