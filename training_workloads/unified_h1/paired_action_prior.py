"""Training-only paired motion examples on actual M4 rollout backgrounds."""

import math

import torch

from instinctlab.control.ics import mounted_mid360_rays_body


CANDIDATE = 'R6-CANDIDATE-INSTINCT-GPU-R1'
PAIRS = 4
WEIGHT = .18


@torch.no_grad()
def twins(policy, rays, *, tick, period_s):
    """Add moving spherical returns to an unchanged observed background."""
    grid = policy['lidar_grid'][:PAIRS]
    state = policy['state_vec'][:PAIRS].reshape(-1, 4, 13).clone()
    lanes = len(grid)
    phase = torch.arange(lanes, device=grid.device) + tick * lanes
    yaw = phase.remainder(12).to(grid.dtype) * (math.pi / 6.)
    forward = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), dim=-1)
    side = torch.stack((-yaw.sin(), yaw.cos(), torch.zeros_like(yaw)), dim=-1)
    signed = (phase.div(12, rounding_mode='floor').remainder(2).to(grid.dtype) * 2 - 1)[:, None]
    command = (.23 + .025 * phase.remainder(3).to(grid.dtype))[:, None] * forward
    state[:, :, 6:9] = command[:, None]
    age = state[:, :, 12]
    offset = torch.arange(3, -1, -1, device=grid.device) * period_s
    elapsed = (offset[None, :] + age - age[:, -1:]).clamp(0., .45)
    center = 1.48 * forward + .20 * signed * side
    displacement = .65 * elapsed[:, :, None] * forward[:, None]
    centers = torch.stack((center[:, None] + displacement, center[:, None] - displacement))
    dot = centers @ rays.T
    delta = dot.square() - centers.square().sum(dim=-1, keepdim=True) + .36**2
    hit = dot - delta.clamp_min(0.).sqrt()
    hit_valid = (delta >= 0.) & (hit >= .1) & (hit <= 40.)
    hit = torch.where(hit_valid, hit, 40.)
    background = grid.reshape(lanes, 4, 3, -1)
    base_valid = background[:, :, 1].bool()
    base_range = torch.where(base_valid, background[:, :, 0], 40.)
    ranges = torch.minimum(hit, base_range)
    valid = hit_valid | base_valid
    reliability = valid * (-age[:, :, None] / .5).exp()
    frames = torch.stack((ranges, valid.to(grid.dtype), reliability), dim=3).reshape(-1, 12, 59, 360)
    states = state.flatten(1).repeat(2, 1)
    keep = .94 * command
    avoid = .32 * command - .17 * signed * side
    target = torch.cat((avoid, keep))
    moved = ((ranges[0] - ranges[1]).abs() > .03).any(dim=2).any(dim=1)
    latest_visible = (hit[0, :, -1] < base_range[:, -1]).sum(1) >= 8
    return dict(lidar_grid=frames, state_vec=states), target, moved & latest_visible


def add_to_trainer(trainer, *, period_s):
    rays = mounted_mid360_rays_body(device=next(trainer.actor.parameters()).device)
    previous = trainer.actor_auxiliary
    tick = 0

    def auxiliary(policy, output):
        nonlocal tick
        matched, target, eligible = twins(policy, rays, tick=tick, period_s=period_s)
        tick += 1
        action = trainer.actor(matched).mean_action
        requested = action[:, :1] * matched['state_vec'][:, 45:48] + action[:, 1:] - .5
        error = ((requested - target) / .23).square().sum(1)
        pair_loss = WEIGHT * (error * eligible.repeat(2)).mean()
        auxiliary.last_metrics = dict(paired_examples=int(eligible.sum().item()),
                                      paired_action_loss=float(pair_loss.detach().item()))
        return previous(policy, output) + pair_loss

    trainer.actor_auxiliary = auxiliary
    return dict(pairs_per_minibatch=PAIRS, weight=WEIGHT, policy_period_s=period_s,
                inputs=['lidar_grid', 'state_vec'], targets='decoded_velocity_direction')
