"""Check false hover reflexes on matched static and moving LiDAR histories."""

import argparse
import json
import math
from pathlib import Path

import torch

from instinctlab.control.ics import ICSConfig, attenuate_command, mounted_mid360_rays_body
from instinctlab.observations.actor import ActorObservationHistory
from tools.r6_direct.hover_representation_probe import residual, retreat_cue
from tools.r6_direct.import_legacy import pin
from tools.r6_direct.instinct_cycle_finish import guard
from tools.r6_direct.source import git
from tools.r6_direct.temporal_probe import FIXTURES, body_to_world, load_actor, scan_at


def observations(rays, angular_rate, noise_m, delivery_delay_s):
    """Each static/moving pair shares the same final heading, gyro, and noise."""
    zero = torch.zeros(1, 3)
    gravity = torch.tensor([[0., 0., -1.]])
    gyro = torch.tensor([[0., 0., angular_rate]])
    policies, moving = [], []
    for index, (name, fixture) in enumerate(FIXTURES):
        if name == 'street_crossing_object':
            continue
        for degrees in range(0, 360, 30):
            for approaching in (False, True):
                if approaching and name == 'uniform_pipe':
                    continue
                history = ActorObservationHistory()
                random = torch.Generator().manual_seed(719 + index * 360 + degrees)
                for frame in range(4):
                    time = (frame - 3) * .1
                    rotation = body_to_world(degrees + math.degrees(angular_rate * time), rays)
                    scan, valid = scan_at(rays, rotation, name, fixture, -.5 if approaching else 0., time)
                    noisy = scan + noise_m * torch.randn(scan.shape, generator=random)
                    scan = torch.where(valid, noisy.clamp(.1, 40.), 40.)
                    timestamp = torch.tensor([[frame * .1]])
                    policy = history.update(scan[None], valid[None], timestamp, timestamp + delivery_delay_s,
                                            gyro, gravity, zero, zero)
                policies.append(policy)
                moving.append(approaching)
    return {key: torch.cat([p[key] for p in policies]) for key in policies[0]}, torch.tensor(moving)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prototype_result', type=Path)
    args = parser.parse_args()
    guard()
    torch.set_num_threads(2)
    prototype = pin(args.prototype_result)
    previous = json.loads(args.prototype_result.read_bytes())
    actor = load_actor(previous['checkpoint'])
    gain = previous['validation']['gated_direction']['learned_gain']
    rays = mounted_mid360_rays_body()
    config = ICSConfig(clearance_margin_m=.15)
    results = []
    for delay in (0., .1):
        for angular_rate in (0., .10, -.21, .21):
            for noise in (.01, .03):
                policy, moving = observations(rays, angular_rate, noise, delay)
                lidar = policy['lidar_grid']
                original = residual(actor, policy, enhance=False)
                issued = attenuate_command(lidar, rays, original, .1, config).v_final_b
                for neighborhood in (0, 8):
                    cue = retreat_cue(policy, angular_neighborhood=neighborhood)
                    active = cue.norm(dim=1) > .1
                    proposed = residual(actor, policy, enhance=False, cue=cue, gain=gain)
                    changed = attenuate_command(lidar, rays, proposed, .1, config).v_final_b - issued
                    stationary = changed[~moving, :2].norm(dim=1)
                    results.append(dict(noise_m=noise, angular_rate_rad_s=angular_rate, delay_s=delay,
                                        neighbor_radius=neighborhood, static_count=int((~moving).sum()),
                                        false_triggers=int((~moving & active).sum()), moving_count=int(moving.sum()),
                                        moving_triggers=int((moving & active).sum()),
                                        static_p99_added_mm_s=1000 * stationary.quantile(.99).item()))
    guard()
    print(json.dumps(dict(source_commit=git('rev-parse', 'HEAD'), prototype=prototype, results=results),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
