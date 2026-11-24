"""CPU learnability check for fixed-latest hover reflexes, independent of PPO."""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from instinctlab.control.ics import mounted_mid360_rays_body
from tools.r6_direct.import_legacy import pin
from tools.r6_direct.instinct_cycle_finish import guard
from tools.r6_direct.source import git
from tools.r6_direct.temporal_probe import (
    AGES_S, FIXTURES, PERIOD_S, RATES, body_to_world, load_actor, observation,
)
from training_workloads.unified_h1.dynamic_motion_training import plausible_approach


TRAIN_YAWS = (0., 45., 90., 135.)
TEST_YAWS = (22.5, 67.5, 112.5, 157.5)
HELD_OUT = 'pipe_end_wall'
STEPS = 96


@torch.no_grad()
def pairs(yaws, *, held_out):
    rays = mounted_mid360_rays_body()
    selected = FIXTURES[0:3] if not held_out else tuple((name, fixture) for name, fixture in FIXTURES if name == HELD_OUT)
    copies = []
    for name, fixture in selected:
        for yaw in yaws:
            rotation = body_to_world(yaw, rays)
            for age in AGES_S:
                trilogy = [observation(rays, rotation, name, fixture, rate, age, command_speed=0.)[0]
                           for rate in RATES]
                policy = {key: torch.cat([item[key] for item in trilogy]) for key in trilogy[0]}
                latest = policy['lidar_grid'][:, -3:]
                if not torch.equal(latest, latest[:1].expand_as(latest)):
                    raise ValueError('Fixed-latest range inputs changed across trajectories')
                visible = plausible_approach(policy, period_s=PERIOD_S).tolist()
                if visible == [True, False, False]:
                    copies.append((policy, rotation[0]))
    if not copies:
        raise ValueError('No eligible paired observations')
    return (
        {key: torch.stack([item[key] for item, _ in copies]) for key in copies[0][0]},
        torch.stack([heading for _, heading in copies]),
    )


@torch.no_grad()
def retreat_cue(policy, *, angular_neighborhood=0):
    full_scan = policy['lidar_grid']
    scan = full_scan[:, :, ::3]
    state = policy['state_vec'].reshape(-1, 4, 13)
    weight = torch.zeros_like(scan[:, 9])
    for k in range(3):
        gap = (3 - k) * PERIOD_S + state[:, k, 12] - state[:, -1, 12]
        gap = gap[:, None, None]
        old = scan[:, k * 3]
        valid_old = scan[:, k * 3 + 2] > .55
        if angular_neighborhood:
            span = angular_neighborhood
            old_full = full_scan[:, k * 3:k * 3 + 1]
            reliable = full_scan[:, k * 3 + 2:k * 3 + 3] > .55
            low = -torch.where(reliable, old_full, torch.inf)
            low = functional.pad(low, (span, span, 0, 0), mode='circular')
            low = functional.max_pool2d(low, (1, 2 * span + 1), stride=1)
            low = functional.max_pool2d(low, (2 * span + 1, 1), stride=1, padding=(span, 0))
            old = -low[:, 0, ::3]
            valid_old = old.isfinite()
        difference = old - scan[:, 9]
        coherent = (gap >= .16) & (gap <= .33) & (difference > .20 * gap) & (difference < 1.5 * gap)
        coherent &= (scan[:, 9] < 1.9) & valid_old & (scan[:, 11] > .55)
        neighbors = coherent.clone()
        for offset in (-2, -1, 1, 2):
            neighbors &= coherent.roll(offset, dims=-1)
        weight = torch.maximum(weight, torch.where(neighbors, difference, 0.))
    rays = mounted_mid360_rays_body().reshape(59, 360, 3)[::3].reshape(-1, 3)
    bearing = weight.flatten(1) @ rays
    bearing[:, 2] = 0.
    active = (state[:, :, :3].norm(dim=2).amax(dim=1) < .22) & (state[:, -1, 12] <= .22)
    return -bearing / bearing.norm(dim=1, keepdim=True).clamp_min(1.e-6) * active[:, None]


def residual(model, policy, *, enhance, cue=None, gain=None):
    if not enhance and cue is None:
        return model(policy).mean_action[:, 1:] - .5
    lidar, state = policy['lidar_grid'], policy['state_vec']
    ranges, validity, reliability = lidar[:, 0::3], lidar[:, 1::3], lidar[:, 2::3]
    scaled = ranges / 40.
    if enhance:
        latest = ranges[:, -1:]
        difference = latest - ranges[:, :3]
        reliable_pair = (reliability[:, -1:] > .55) & (reliability[:, :3] > .55)
        salient = reliable_pair & (latest < 2.2) & (difference.abs() > .035)
        scaled[:, :3] += torch.where(salient, .5 * difference.clamp(-.5, .5), 0.)
    normalized = torch.cat((scaled, validity, reliability), dim=1)
    raw = model.trunk(torch.cat((model.lidar_encoder(normalized), model.state_encoder(state)), dim=-1))
    alpha_logits, beta_logits = raw.chunk(2, dim=-1)
    floor = model.config.concentration_min
    span = model.config.concentration_max - floor
    if cue is not None:
        if gain is None:
            # The first action scales the operator command. Its value has no
            # effect on translation at zero command, so test reuse for reflex magnitude.
            alpha = floor + span * alpha_logits[:, :1].sigmoid()
            beta = floor + span * beta_logits[:, :1].sigmoid()
            gain = 8. * beta / (alpha + beta)
            alpha_logits, beta_logits = alpha_logits.detach(), beta_logits.detach()
        shift = torch.cat((torch.zeros_like(cue[:, :1]), gain * cue), dim=1)
        alpha_logits, beta_logits = alpha_logits + shift, beta_logits - shift
    alpha, beta = floor + span * alpha_logits.sigmoid(), floor + span * beta_logits.sigmoid()
    return (alpha / (alpha + beta))[:, 1:] - .5


def batch(triplet, selected):
    policy, heading = triplet
    return {key: values[selected].flatten(0, 1) for key, values in policy.items()}, heading[selected]


@torch.no_grad()
def summary(model, triplet, *, enhance, gain=None, reuse_channel=False):
    policy, heading = batch(triplet, slice(None))
    cue = None if gain is None else retreat_cue(policy)
    if reuse_channel:
        cue = retreat_cue(policy, angular_neighborhood=8)
    velocity = residual(model, policy, enhance=enhance, cue=cue, gain=gain).reshape(-1, 3, 3)
    forward = (velocity * heading[:, None]).sum(-1)
    return dict(pairs=len(heading), mean_retreat_difference_mm_s=1000 * (forward[:, 2] - forward[:, 0]).mean().item(),
                mean_stationary_horizontal_mm_s=1000 * velocity[:, 1, :2].norm(dim=1).mean().item())


def travel_policy(policy, heading):
    state = policy['state_vec'].clone().reshape(-1, 3, 4, 13)
    state[..., 6:9] = .20 * heading[:, None, None]
    state[..., 9:12] = .15 * heading[:, None, None]
    return dict(policy, state_vec=state.flatten(0, 1).flatten(1))


def try_command_channel(actor, training, validation):
    model = copy.deepcopy(actor)
    model.trunk[-1].requires_grad_(True)
    optimizer = torch.optim.Adam(model.trunk[-1].parameters(), lr=.003)
    sampling = torch.Generator().manual_seed(41)
    for _ in range(STEPS):
        selected = torch.randint(len(training[1]), (4,), generator=sampling)
        policy, heading = batch(training, selected)
        target = torch.zeros(len(selected), 3, 3)
        target[:, 0] = -.18 * heading
        prediction = residual(model, policy, enhance=False, cue=retreat_cue(policy, angular_neighborhood=8))
        moving = travel_policy(policy, heading)
        baseline = actor(moving).mean_action[:, 0]
        scale = model(moving).mean_action[:, 0]
        optimizer.zero_grad()
        ((prediction.reshape_as(target) - target).square().mean() + .15 * (scale - baseline).square().mean()).backward()
        optimizer.step()
    outcomes = {key: summary(model, value, enhance=False, reuse_channel=True) for key, value in validation.items()}
    with torch.no_grad():
        for key, outcome in outcomes.items():
            moving = travel_policy(*batch(validation[key], slice(None)))
            delta = model(moving).mean_action[:, 0] - actor(moving).mean_action[:, 0]
            outcome['mean_forward_change_mm_s'] = 200. * delta.mean().item()
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    guard()
    torch.set_num_threads(2)
    checkpoint = pin(args.checkpoint)
    actor = load_actor(checkpoint)
    training = pairs(TRAIN_YAWS, held_out=False)
    validation = dict(unseen_yaw=pairs(TEST_YAWS, held_out=False), unseen_geometry=pairs(TEST_YAWS, held_out=True))
    results = {}
    for label, enhance in (('original', False), ('range_difference', True)):
        model = copy.deepcopy(actor)
        optimizer = torch.optim.Adam(model.parameters(), lr=3.e-4)
        sampling = torch.Generator().manual_seed(41)
        before = {key: summary(model, value, enhance=enhance) for key, value in validation.items()}
        for _ in range(STEPS):
            selected = torch.randint(len(training[1]), (4,), generator=sampling)
            policy, heading = batch(training, selected)
            target = torch.zeros(len(selected), 3, 3)
            target[:, 0] = -.18 * heading
            prediction = residual(model, policy, enhance=enhance).reshape_as(target)
            optimizer.zero_grad()
            (prediction - target).square().mean().backward()
            optimizer.step()
        results[label] = dict(before=before, after={key: summary(model, value, enhance=enhance)
                                                    for key, value in validation.items()})
    actor.requires_grad_(False)
    gain = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([gain], lr=.06)
    sampling = torch.Generator().manual_seed(41)
    before = {key: summary(actor, value, enhance=False, gain=gain) for key, value in validation.items()}
    for _ in range(STEPS):
        selected = torch.randint(len(training[1]), (4,), generator=sampling)
        policy, heading = batch(training, selected)
        target = torch.zeros(len(selected), 3, 3)
        target[:, 0] = -.18 * heading
        prediction = residual(actor, policy, enhance=False, cue=retreat_cue(policy), gain=gain).reshape_as(target)
        optimizer.zero_grad()
        (prediction - target).square().mean().backward()
        optimizer.step()
    results['gated_direction'] = dict(learned_gain=gain.detach().item(), before=before,
        after={key: summary(actor, value, enhance=False, gain=gain) for key, value in validation.items()})
    results['reuse_command_channel'] = try_command_channel(actor, training, validation)
    guard()
    print(json.dumps(dict(source_commit=git('rev-parse','HEAD'), checkpoint=checkpoint, optimization_steps=STEPS,
                          training_pairs=len(training[1]), validation=results), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
