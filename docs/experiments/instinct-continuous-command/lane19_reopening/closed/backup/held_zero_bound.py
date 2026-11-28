"""Calculate the mover clearance if the queued neutral continues past handoff."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not stop_paths(), stop_paths()
    script = Path(__file__).resolve()
    registry = script.with_name('HELD-ZERO-INPUTS.json')
    sources = load(registry)
    hard = load(checked(sources['hard_pair']))
    target = hard['comparison']['first_backup']
    arm = hard['arm']
    applied = arm['records']
    assert hard['comparison']['identical_prefix'] and target == 1067
    start, end, lane, last_certificate = 1060, 1149, 19, hard['comparison']['valid_through_exclusive']
    dense = np.memmap(checked(sources['dense']), '<f4', 'r', shape=(1200, 21, 32, 7))
    mover = np.concatenate((dense[start:end, :-1, lane, 3:6].reshape(-1, 3),
                            dense[end - 1, -1:, lane, 3:6])).astype(float)
    radius = np.concatenate((dense[start:end, :-1, lane, 6].reshape(-1),
                             dense[end - 1, -1:, lane, 6]))
    run_time = np.arange(len(mover)) * .005
    offset = run_time - (arm['case']['fork_step'] - start) * .1
    rise = np.clip(offset / .15, 0., 1.)
    fall = np.clip((1.20 - offset) / .20, 0., 1.)
    accel = arm['case']['sign'] * .22 * rise**2 * (3. - 2. * rise) * fall**2 * (3. - 2. * fall)
    shift = cumulative_trapezoid(cumulative_trapezoid(accel, dx=.005, initial=0.), dx=.005, initial=0.)
    mover += shift[:, None] * np.array([1., 1., 0.]) / np.sqrt(2.)
    lag = sources['actuator_lag_s']
    tick = target - start
    pos = np.array(applied[tick]['position_before'])
    velocity = np.array(applied[tick]['velocity_before'])
    previous = np.array(applied[tick - 1]['issued_w'])
    seconds = run_time[tick * 20:] - run_time[tick * 20]
    queued = seconds.clip(max=.1)
    decay = np.exp(-queued / lag)
    queue_path = pos + queued[:, None] * previous + (lag * (1. - decay))[:, None] * (velocity - previous)
    queue_speed = previous + decay[:, None] * (velocity - previous)
    brake_for = (seconds - .1).clip(min=0.)
    trajectory = queue_path + (lag * (-np.expm1(-brake_for / lag)))[:, None] * queue_speed
    clearance = np.linalg.norm(trajectory - mover[tick * 20:], axis=1) - .34 - radius[tick * 20:]
    expiry_index = (last_certificate - target) * 20

    # The actual arm issues neutral from t1067--1071, so its position at
    # the start of t1073 must still agree, before t1072's intercept takes effect.
    for step in range(target, hard['comparison']['last_zero_issue'] + 3):
        offset = (step - target) * 20
        assert np.allclose(trajectory[offset], applied[step - start]['position_before'], atol=1.e-7, rtol=0.)
    worst = int(clearance.argmin())
    misses = np.flatnonzero(clearance < .15)
    result = dict(inputs=pin(registry), method=pin(script), mode='offline_mover_only_held_neutral',
                  held_from_packet=target, queue_s=.10, observed_until_s=end * .1,
                  proof_expiry_s=last_certificate * .1, before_expiry_m=float(clearance[:expiry_index + 1].min()),
                  after_expiry_m=float(clearance[expiry_index:].min()),
                  minimum_m=float(clearance[worst]), worst_time_s=target * .1 + float(seconds[worst]),
                  first_breach_s=None if not len(misses) else target * .1 + float(seconds[misses[0]]),
                  stopped_position=trajectory[-1].tolist())
    assert not stop_paths(), stop_paths()
    print(json.dumps(dict(raw=save(args.output, result), result=result)), flush=True)


if __name__ == '__main__':
    main()
