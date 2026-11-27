"""Freeze seeded simulated command streams with persistent, aggressive changes."""
import argparse
import json
import math
from pathlib import Path
import random

from tools.r6_direct.import_legacy import stop_paths
from tools.r6_evidence import file_pin
from training_workloads.unified_h1.command_replay import CRAZY_MODES, CRAZY_SCHEMA, STEP_MS, load_route_streams


MAX_SPEED_M_S = .299
# Authored stimulus directions. No simulated pose, motion or obstacle readback.
HAZARD_HEADING = {'room_corner': 0., 'trunk_aisle': .34, 'street_gap': 0.,
                  'corridor': math.pi/2, 'open_tube': -math.pi/2}


def challenges(seed, layout, start_ms, duration_ms):
    rng = random.Random(f'{seed}:{layout}')
    current_ms = start_ms
    heading = HAZARD_HEADING[layout]
    while current_ms < duration_ms:
        modes = rng.sample(CRAZY_MODES, len(CRAZY_MODES))
        for mode in modes:
            if current_ms >= duration_ms:
                return
            ticks = rng.randint(10, 40)  # historical one-to-four second holds
            frequency = 0.
            amplitude = 0.
            if mode == 'normal':
                heading = rng.uniform(-math.pi, math.pi)
                speed = rng.uniform(.07, .18)
            elif mode == 'aggressive':
                heading += math.pi if rng.random() < .65 else rng.choice((-math.pi/2, math.pi/2))
                speed = rng.uniform(.24, MAX_SPEED_M_S)
            elif mode == 'obstacleward':
                heading = HAZARD_HEADING[layout] + rng.uniform(-.18, .18)
                speed = rng.uniform(.25, MAX_SPEED_M_S)
            elif mode == 'oscillation':
                heading += rng.choice((-math.pi/2, math.pi/2))
                speed = rng.uniform(.06, .18)
                frequency = rng.uniform(.5, 2.)
                amplitude = rng.uniform(.03, .12)
            else:
                speed = 0.
            for tick in range(ticks):
                if current_ms >= duration_ms:
                    return
                if tick == 0 or mode == 'oscillation':
                    speed_now = min(MAX_SPEED_M_S, speed+amplitude*math.sin(math.tau*frequency*tick*STEP_MS/1000.))
                    command = [speed_now*math.cos(heading), speed_now*math.sin(heading), 0.] if speed_now else [0., 0., 0.]
                    yield dict(t_ms=current_ms, velocity_w_m_s=command, mode=mode)
                current_ms += STEP_MS


def generate(prefix, seed, duration_ms):
    streams = load_route_streams(prefix)
    start = next(iter(streams.values()))['duration_ms']
    if (type(seed) is not int or not 0 <= seed < 2**31 or type(duration_ms) is not int
            or duration_ms < start+len(CRAZY_MODES)*4000 or duration_ms > 3_600_000 or duration_ms % STEP_MS):
        raise ValueError('duration must leave one complete cycle after the route prefix')
    layouts = {}
    for layout, stream in streams.items():
        layouts[layout] = [*stream['events'], *challenges(seed, layout, start, duration_ms)]
    return dict(schema_version='instinctrl-command-route-streams/1', duration_ms=duration_ms, layouts=layouts,
                generator=dict(schema_version=CRAZY_SCHEMA, seed=seed, prefix=prefix, prefix_duration_ms=start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--duration-ms', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if stop_paths():
        raise RuntimeError('STOP')
    result = generate(file_pin(args.prefix), args.seed, args.duration_ms)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
