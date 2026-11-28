"""Bound and screen common parking schedules under the frozen lane 19 response."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
import torch

from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.replay_clearance import static_distance
from tools.r6_direct.source import checked, git
from training_workloads.unified_h1.escape_holdout_scene import scene_geometry


def response(seconds, lag):
    time = np.maximum(seconds, 0.)
    return time - lag * (-np.expm1(-time / lag))


def propagation(time, packets, lag, p, velocity, pending):
    interval_start = .1 * np.arange(1, packets + 1)
    weights = response(time[:, None] - interval_start, lag) - response(time[:, None] - interval_start - .10, lag)
    gain = lambda t: -np.expm1(-np.maximum(t, 0.) / lag)
    speed_weights = gain(time[:, None] - interval_start) - gain(time[:, None] - interval_start - .1)
    base = p + lag * (-np.expm1(-time / lag))[:, None] * velocity
    base += (response(time, lag) - response(time - .1, lag))[:, None] * pending
    base_speed = np.exp(-time / lag)[:, None] * velocity + (gain(time) - gain(time - .1))[:, None] * pending
    return base, weights, base_speed, speed_weights


def mover_bank(dense, cases, time, start):
    centers = []
    for case in cases.values():
        phase = time - (case['fork_step'] - start) * .10
        rise = np.clip(phase / .15, 0., 1.)
        fall = np.clip((1.20 - phase) / .20, 0., 1.)
        a = case['sign'] * .22 * rise**2 * (3. - 2. * rise) * fall**2 * (3. - 2. * fall)
        shift = cumulative_trapezoid(cumulative_trapezoid(a, dx=.005, initial=0.), dx=.005, initial=0.)
        centers.append(dense[:, 3:6] + shift[:, None] * np.array([1., 1., 0.]) / np.sqrt(2.))
    return np.array(centers)


def early_parking_upper(base, weights, centers, sphere, time, first, count, resume, cases):
    bounds = {}
    for label, fixed in (('fixed_witness', True), ('any_first', False)):
        zero = 1 if fixed else 0
        # Doubling down on a failed bound would be meaningless if the bounds
        # themselves clipped an issue. Allow the full speed immediately here.
        before = .45 * weights[:, zero:count].sum(axis=1)
        after = .45 * weights[:, resume:].sum(axis=1)
        origin = base + (weights[:, :1] * first if fixed else 0.)
        d = origin[None] - centers
        for timing, disk in (('held_neutral', before), ('free_resume_at_1084', before + after)):
            for space in ('horizontal', '3d'):
                upper = (np.hypot(np.linalg.norm(d[:, :, :2], axis=2) + disk[None], d[:, :, 2])
                         if space == 'horizontal' else np.linalg.norm(d, axis=2) + disk[None])
                bound = upper - sphere
                result = {}
                for name, row in zip(cases, bound, strict=True):
                    index = int(row.argmin())
                    result[name] = dict(m=float(row[index]), seconds_after_start=float(time[index]),
                                        reachable_radius_m=float(disk[index]))
                bounds[f'{label}_{timing}_{space}'] = result
    return bounds


def schedules(first, cap, slew, release_count, switches, angle_count, second_speeds):
    headings = np.arange(angle_count)*2.*np.pi/angle_count
    direction = np.stack((np.cos(headings), np.sin(headings), np.zeros_like(headings)), axis=1)
    initial, later, change, speed = np.meshgrid(np.arange(len(direction)), np.arange(len(direction)),
                                               switches, second_speeds, indexing='ij')
    spec = np.stack((initial, later, change, speed), axis=-1).reshape(-1, 4)
    actions = np.repeat(first[None, None], len(spec), axis=0)
    actions = np.repeat(actions, release_count, axis=1)
    for k in range(1, release_count):
        goal = np.where((k < spec[:, 2])[:, None], cap * direction[spec[:, 0].astype(int)],
                        spec[:, 3:4] * direction[spec[:, 1].astype(int)])
        difference = goal - actions[:, k - 1]
        actions[:, k] = actions[:, k - 1] + difference * (
            slew / np.maximum(slew, np.linalg.norm(difference, axis=1, keepdims=True)))
    assert np.linalg.norm(actions, axis=2).max() <= cap+1.e-9
    assert np.linalg.norm(np.diff(actions, axis=1), axis=2).max() <= slew+1.e-9
    return spec, actions


def append_requests(actions, requests, count, resumed_at, slew):
    issued = np.zeros((len(actions), len(requests), 3))
    issued[:, :count] = actions
    for k in range(resumed_at, len(requests)):
        if not requests[k].any():
            continue
        delta = requests[k] - issued[:, k - 1]
        issued[:, k] = issued[:, k - 1] + delta * (slew / np.maximum(slew, np.linalg.norm(delta, axis=1, keepdims=True)))
    return issued


def score_bank(models, centers, sphere, time, issued, zero_interval, provisional):
    n = len(issued)
    drift = np.zeros(n)
    peak_after_deadline = np.zeros(n)
    floors = np.full((n, len(centers)), np.inf)
    neutral_start, neutral_end = zero_interval
    zero_index = int(np.argmin(abs(time - neutral_start)))
    quiet = (time >= neutral_start + 1. - 1.e-9) & (time <= neutral_end + 1.e-9)
    zero_window = (time >= neutral_start - 1.e-9) & (time <= neutral_end + 1.e-9)
    for base, weights, base_speed, speed_weights in models:
        pos = np.einsum('tk,nkj->ntj', weights[:, :issued.shape[1]], issued) + base
        drift = np.maximum(drift, np.linalg.norm(pos[:, zero_window] - pos[:, zero_index:zero_index + 1], axis=2).max(axis=1))
        speed = np.einsum('tk,nkj->ntj', speed_weights[quiet, :issued.shape[1]], issued) + base_speed[quiet]
        peak_after_deadline = np.maximum(peak_after_deadline, np.linalg.norm(speed, axis=2).max(axis=1))
        for case, mover in enumerate(centers):
            clear = np.linalg.norm(pos - mover, axis=2) - sphere
            floors[:, case] = np.minimum(floors[:, case], clear.min(axis=1))
    if provisional:
        return floors.min(axis=1), drift, peak_after_deadline
    return floors, drift, peak_after_deadline


def select_exact(spec, issued, models, centers, sphere, time, solids, body, cfg, gates, stop_step, mode):
    coarse = slice(None, None, cfg['screening_stride'])
    small = [tuple(terms[coarse] for terms in model) for model in models]
    quiet = [(step - cfg['start']) * .10 for step in cfg['zero_window']]
    bucket = cfg['exact_top_per_bucket']
    values = [score_bank(small, centers[:, coarse], sphere[coarse], time[coarse], batch, quiet, True)
              for batch in np.array_split(issued, 8)]
    coarse_floor, drift, speed = (np.concatenate(column) for column in zip(*values, strict=True))
    acceptable = (drift <= gates['maximum_stop_drift_m']) & (speed <= .07)
    feasible = np.flatnonzero(acceptable)
    indices = np.union1d(np.argsort(coarse_floor)[-bucket:], feasible[np.argsort(coarse_floor[feasible])[-bucket:]])
    plans = issued[indices]
    scores, drift, speed = score_bank(models, centers, sphere, time, plans, quiet, False)
    static = np.full(len(indices), np.inf)
    for base, weights, _, _ in models:
        positions = np.einsum('tk,nkj->ntj', weights[:, :plans.shape[1]], plans) + base
        distances = static_distance(positions.reshape(-1, 3), solids).reshape(positions.shape[:2]) - body
        static = np.minimum(static, distances.min(axis=1))
    minimum = np.minimum(scores.min(axis=1), static)

    def best(valid):
        if not valid.any():
            return None
        i = int(np.where(valid, minimum, -np.inf).argmax())
        i_heading, j_heading, change_packet, speed_target = spec[indices[i]]
        return dict(clearance_m=float(minimum[i]), by_case=dict(zip(cfg['cases'], scores[i].tolist(), strict=True)),
                    static_m=float(static[i]), release_drift_m=float(drift[i]),
                    max_speed_after_stop_deadline_m_s=float(speed[i]), issued_world=plans[i].tolist(),
                    first_heading_deg=float(360/i_angle * i_heading), second_heading_deg=float(360/i_angle * j_heading),
                    change_step=cfg['start']+int(change_packet), second_target_speed_m_s=float(speed_target),
                    zero_issue_step=stop_step)

    i_angle = cfg['angle_count']
    return dict(mode=mode, stop_step=stop_step, total_schedules=len(spec), dense_finalists=len(indices),
                dense_stop_finalists=int(((drift <= gates['maximum_stop_drift_m']) & (speed <= .07)).sum()),
                best=best(np.ones(len(indices), dtype=bool)),
                best_stop=best((drift <= gates['maximum_stop_drift_m']) & (speed <= .07)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not stop_paths(), stop_paths()
    script = Path(__file__).resolve()
    registry = script.with_name('INPUTS.json')
    contract = load(registry)
    cfg = contract['protocol']
    source = cfg['v60_source']
    assert git('rev-parse','HEAD') == source and not git('branch','--show-current')
    gates = load(checked(contract['gates']))['by_layout']['street_gap']
    pairs = load(checked(contract['hard_backup']))
    origin = pairs['arm']['records'][0]
    assert origin['step'] == cfg['start'] and origin['proof']['witness_reproduced']
    first = np.array(origin['issued_w'])
    p, velocity = np.array(origin['position_before']), np.array(origin['velocity_before'])
    start, end, lane = cfg['start'], cfg['end'], cfg['lane']
    with gzip.open(checked(contract['trace']), 'rt') as stream:
        next(stream)
        for _ in range(start-1):
            next(stream)
        previous = json.loads(next(stream))
        rows = [json.loads(next(stream)) for _ in range(start, end)]
    assert previous['step'] == start-1
    assert [r['step'] for r in rows]==list(range(start,end))
    pending = np.array(previous['v_final_w'][lane])
    commands = np.array([r['raw_event'][lane]['velocity_w_m_s'] for r in rows])
    zero_start, resume = (x-start for x in cfg['zero_window'])
    assert not commands[zero_start:resume].any() and np.linalg.norm(commands[resume]) > .03
    assert np.linalg.norm(first - pending) <= cfg['slew_m_s']
    grid = np.memmap(checked(contract['dense']), '<f4', 'r', shape=(1200, 21, 32, 7))
    dense = np.concatenate((grid[start:end, :-1, lane].reshape(-1, 7), grid[end-1, -1:, lane])).astype(float)
    assert np.allclose(dense[0, :3], p, atol=2.e-5, rtol=0.)
    t = np.arange(len(dense)) * .005
    centers = mover_bank(dense, cfg['cases'], t, start)
    sphere = .34 + dense[:, 6]
    origins = torch.tensor(load(checked(contract['geometry']))['env_origins'])
    _, _, vertices, indices = scene_geometry(origins, cfg['scene_seed'], command_routes=True)
    solids = np.asarray(vertices)[np.asarray(indices).reshape(-1, 3)]
    model = [propagation(t, end-start, lag, p, velocity, pending) for lag in cfg['lags_s']]
    reach = {str(lag):early_parking_upper(base, weight, centers, sphere, t, first, zero_start, resume, cfg['cases'])
             for lag, (base, weight, _, _) in zip(cfg['lags_s'], model, strict=True)}
    screened = []
    for stop_step in cfg['stop_steps']:
        assert not stop_paths(), stop_paths()
        spec, sequences = schedules(first, cfg['speed_cap_m_s'], cfg['slew_m_s'], stop_step-start,
                                    cfg['switch_packets'], cfg['angle_count'], cfg['second_speeds'])
        resumed = append_requests(sequences, commands, stop_step-start, resume, cfg['slew_m_s'])
        for mode, plans in (('held_neutral', sequences), ('latest_request_at_1084', resumed)):
            score = select_exact(spec, plans, model, centers, sphere, t, solids, dense[:, 6], cfg, gates, stop_step, mode)
            screened.append(score)
            summary = {key:value for key,value in score.items() if key not in ('best','best_stop')}
            summary.update({k:None if score[k] is None else [score[k]['clearance_m'],score[k]['release_drift_m'],
                           score[k]['max_speed_after_stop_deadline_m_s']] for k in ('best','best_stop')})
            print(json.dumps(summary), flush=True)
    data = dict(inputs=pin(registry), script=pin(script), first_world=first.tolist(), preceding_world=pending.tolist(),
                relaxation=reach, screened=screened)
    print(json.dumps(dict(raw=save(args.output, data),relaxation=reach)), flush=True)


if __name__ == '__main__':
    main()
