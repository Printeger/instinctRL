"""CPU learnability check for fixed-latest hover reflexes, independent of PPO."""

import argparse
import copy
import json
from pathlib import Path

import torch

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


def residual(model, policy, *, enhance):
    if not enhance:
        return model(policy).mean_action[:, 1:] - .5
    lidar, state = policy['lidar_grid'], policy['state_vec']
    ranges, validity, reliability = lidar[:, 0::3], lidar[:, 1::3], lidar[:, 2::3]
    latest = ranges[:, -1:]
    difference = latest - ranges[:, :3]
    reliable_pair = (reliability[:, -1:] > .55) & (reliability[:, :3] > .55)
    salient = reliable_pair & (latest < 2.2) & (difference.abs() > .035)
    historical = ranges[:, :3] / 40. + torch.where(salient, .5 * difference.clamp(-.5, .5), 0.)
    normalized = torch.cat((historical, latest / 40., validity, reliability), dim=1)
    raw = model.trunk(torch.cat((model.lidar_encoder(normalized), model.state_encoder(state)), dim=-1))
    alpha_logits, beta_logits = raw.chunk(2, dim=-1)
    floor = model.config.concentration_min
    span = model.config.concentration_max - floor
    alpha, beta = floor + span * alpha_logits.sigmoid(), floor + span * beta_logits.sigmoid()
    return (alpha / (alpha + beta))[:, 1:] - .5


def batch(triplet, selected):
    policy, heading = triplet
    return {key: values[selected].flatten(0, 1) for key, values in policy.items()}, heading[selected]


@torch.no_grad()
def summary(model, triplet, *, enhance):
    policy, heading = batch(triplet, slice(None))
    velocity = residual(model, policy, enhance=enhance).reshape(-1, 3, 3)
    forward = (velocity * heading[:, None]).sum(-1)
    return dict(pairs=len(heading), mean_retreat_difference_mm_s=1000 * (forward[:, 2] - forward[:, 0]).mean().item(),
                mean_stationary_horizontal_mm_s=1000 * velocity[:, 1, :2].norm(dim=1).mean().item())


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
    guard()
    print(json.dumps(dict(source_commit=git('rev-parse','HEAD'), checkpoint=checkpoint, optimization_steps=STEPS,
                          training_pairs=len(training[1]), validation=results), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
