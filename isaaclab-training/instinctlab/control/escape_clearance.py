"""Observe lateral wall clearance during a stable rear escape start."""

import torch


def wider_side(lidar, state, rays, retreat, side):
    """Return a stable side sign from three distinct delivered frames."""
    history = state.reshape(-1, 4, 13)[:, 1:]
    gap = .1 + history[:, :-1, 12] - history[:, 1:, 12]
    stable = (gap >= .065).all(1) & (gap <= .135).all(1)
    stable &= (history[:, :, :3].norm(dim=2) <= .03).all(1)
    scans = lidar[:, 3:].reshape(len(state), 3, 3, -1)
    distance = scans[:, :, 0]
    axial = distance * (retreat @ rays.T)[:, None]
    lateral = distance * (side @ rays.T)[:, None]
    height = distance * rays[:, 2]
    usable = (scans[:, :, 1] > .5) & (scans[:, :, 2] > .55)
    usable &= (axial.abs() < .4) & (height.abs() < .15)
    estimates = []
    support = stable
    for sign in (1., -1.):
        projected = lateral * sign
        wall = usable & (projected > 0.)
        support &= (wall.sum(dim=-1) >= 20).all(1)
        estimates.append(torch.nanquantile(torch.where(wall, projected, float('nan')), .2, dim=-1))
    difference = estimates[0] - estimates[1]
    return (support & (difference > .10).all(1)).to(lateral.dtype) - (support & (difference < -.10).all(1)).to(lateral.dtype)
