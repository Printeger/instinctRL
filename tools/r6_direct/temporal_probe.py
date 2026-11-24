"""Compare fixed-scan history selectivity of the frozen actors on CPU."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from instinctlab.control.governor import decode_governor
from instinctlab.control.ics import ICSConfig, attenuate_command, mounted_mid360_rays_body
from instinctlab.observations.actor import ActorObservationHistory
from instinctlab.training.checkpoint_h1 import checkpoint_actor
from tools.instinct_tube_probe import body_to_world
from tools.instinct_tube_shadow import scene_scan
from tools.r6_direct.evidence import read
from tools.r6_direct.import_legacy import pin
from tools.r6_direct.instinct_next import PROJECT, FIRST
from tools.r6_direct.instinct_cycle_finish import guard
from tools.r6_direct.source import git
from training_workloads.unified_h1.range_closure_prior import command_gain_ceiling, sampled_rays


PERIOD_S = .1
ORIENTATIONS_DEG = (0., 45., 90., 135.)
AGES_S = (0., .1)
RATES = (-.5, 0., .5)
LONGITUDINAL = ('closing', 'stationary', 'opening')
LATERAL = ('positive_y_motion', 'stationary', 'negative_y_motion')
FIXTURES = (
    ('indoor_moving_panel', dict(half_width_m=2.6, doorway=dict(distance_m=1.04, half_opening_m=.20))),
    ('grove_discrete_obstacles', dict(half_width_m=6., posts=[(1.20, .03, .16), (2.10, -.55, .24), (2.7, .75, .29)])),
    ('street_longitudinal_object', dict(half_width_m=4., posts=[(1.24, -.04, .20), (3.6, -.85, .5)])),
    ('street_crossing_object', dict(half_width_m=4., posts=[(1.22, .02, .18), (3.6, -.85, .5)])),
    ('pipe_end_wall', dict(half_width_m=1.2, endcap_m=1.04)),
    ('uniform_pipe', dict(half_width_m=1.2)),
)


def scan_at(rays, rotation, name, fixture, rate, time):
    latest_geometry = fixture.copy()
    if 'doorway' in fixture:
        latest_geometry['doorway'] = dict(fixture['doorway'], distance_m=fixture['doorway']['distance_m'] + rate * time)
    if 'posts' in fixture:
        (x, y, radius), *rest = fixture['posts']
        moving = (x, y - rate * time, radius) if name == 'street_crossing_object' else (x + rate * time, y, radius)
        latest_geometry['posts'] = [moving, *rest]
    if 'endcap_m' in fixture:
        latest_geometry['endcap_velocity_m_s'] = rate
    return scene_scan(rays, rotation, latest_geometry, time)


def observation(rays, rotation, name, fixture, rate, age, *, command_speed=.2):
    history = ActorObservationHistory()
    zero = torch.zeros(1, 3)
    gravity = torch.tensor([[0., 0., -1.]])
    command_w = torch.tensor([[command_speed, 0., 0.]])
    command = command_w @ rotation
    for idx in range(4):
        distance, valid = scan_at(rays, rotation, name, fixture, rate, (idx - 3) * PERIOD_S)
        sample_time = torch.full((1, 1), idx * PERIOD_S)
        policy = history.update(distance[None], valid[None], sample_time, sample_time + age, zero, gravity,
                                command, .75 * command)
    return policy, command


def load_actor(checkpoint):
    path = Path(checkpoint['path'])
    if pin(path) != checkpoint:
        raise ValueError(f'Checkpoint drift {path}')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    actor = checkpoint_actor(payload)
    actor.load_state_dict(payload['actor'], strict=True)
    return actor.eval()


@torch.inference_mode()
def evaluate(models, *, command_speed=.2):
    rays = mounted_mid360_rays_body()
    subsampled = sampled_rays('cpu')
    config = ICSConfig(clearance_margin_m=.15)
    for name, fixture in FIXTURES:
        for yaw in ORIENTATIONS_DEG:
            rotation = body_to_world(yaw, rays)
            for age in AGES_S:
                by_trend = [observation(rays, rotation, name, fixture, rate, age,
                                        command_speed=command_speed) for rate in RATES]
                latest = by_trend[0][0]['lidar_grid'][:, -3:]
                if not all(torch.equal(latest, p['lidar_grid'][:, -3:]) for p, _ in by_trend):
                    raise ValueError(f'Latest measurement mismatch {name} {yaw} {age}')
                policy = {key: torch.cat([p[key] for p, _ in by_trend]) for key in by_trend[0][0]}
                heading = rotation[:1] if command_speed == 0. else by_trend[0][1] / command_speed
                teacher = command_gain_ceiling(policy, subsampled, period_s=PERIOD_S,
                                               clearance_m=config.d_safe_m + config.clearance_margin_m)
                signals = {}
                for label, model in models.items():
                    mean = model(policy).mean_action
                    governor = decode_governor(mean, policy['state_vec'])
                    guard = attenuate_command(policy['lidar_grid'], rays, governor.v_gov_b, PERIOD_S, config)
                    signals[label] = dict(alpha=mean[:, 0].tolist(),
                                          before_guard_m_s=(governor.v_gov_b * heading).sum(1).tolist(),
                                          after_guard_m_s=(guard.v_final_b * heading).sum(1).tolist(),
                                          guard_beta=guard.metrics['beta'].flatten().tolist())
                    if command_speed == 0.:
                        signals[label].update(before_guard_world_m_s=(governor.v_gov_b @ rotation.T).tolist(),
                                              after_guard_world_m_s=(guard.v_final_b @ rotation.T).tolist())
                order = LATERAL if name == 'street_crossing_object' else LONGITUDINAL
                yield dict(fixture=name, yaw_deg=yaw, age_s=age, history_order=order,
                           teacher=teacher.tolist(), signals=signals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round-2-results', type=Path, default=PROJECT / 'round-2' / 'RESULTS.json')
    parser.add_argument('--round-3-results', type=Path, default=PROJECT / 'round-3' / 'RESULTS.json')
    parser.add_argument('--checkpoint', nargs=2, metavar=('LABEL', 'PATH'), action='append')
    parser.add_argument('--compare-with-zero-gain', metavar='LABEL')
    parser.add_argument('--command-speed', choices=(0., .2), type=float, default=.2)
    args = parser.parse_args()
    guard()
    torch.set_num_threads(2)
    if args.checkpoint:
        checkpoints = {label: pin(Path(path).resolve(strict=True)) for label, path in args.checkpoint}
        if len(checkpoints) != len(args.checkpoint):
            parser.error('Checkpoint labels must be unique')
        inputs = {}
    else:
        inputs = {key: pin(path) for key, path in
                  (('r1', FIRST / 'RESULTS.json'), ('r2', args.round_2_results), ('r3', args.round_3_results))}
        r1, = (m['checkpoint'] for m in read(inputs['r1'])['models'] if m['arm'] == 'candidate')
        checkpoints = dict(r1=r1, **{key: read(inputs[key])['candidate']['checkpoint'] for key in ('r2', 'r3')})
    actors = {key: load_actor(value) for key, value in checkpoints.items()}
    ablation = {}
    if label := args.compare_with_zero_gain:
        baseline = f'{label}-zero-gain'
        if label not in actors or actors[label].hover_reflex is None or baseline in actors:
            parser.error('Zero-gain comparison requires a unique saved actor with a hover reflex')
        actors[baseline] = deepcopy(actors[label])
        with torch.no_grad():
            actors[baseline].hover_reflex.gain.zero_()
        checkpoints[baseline] = checkpoints[label]
        ablation['zero_gain_comparison'] = dict(saved_model=label, zero_gain_clone=baseline)
    rows = list(evaluate(actors, command_speed=args.command_speed))
    guard()
    print(json.dumps(dict(source_commit=git('rev-parse', 'HEAD'), command_speed_mps=args.command_speed,
                          inputs=inputs, checkpoints=checkpoints, **ablation,
                          rows=rows), allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
