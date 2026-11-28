"""Find a common issued-neutral restart schedule for the remaining exact first issues."""
import argparse
import gzip
import json
from pathlib import Path
import runpy
import shutil

import numpy as np
from scipy.optimize import LinearConstraint, minimize
import torch

from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.replay_clearance import static_distance
from tools.r6_direct.source import checked, git
from training_workloads.unified_h1.escape_holdout_scene import scene_geometry


def cap_step(target, previous, change):
    delta = target - previous
    return previous + delta * min(1., change / max(1.e-9, float(np.linalg.norm(delta))))


def layout(active, issued, models, cap, slew, drift, begin, resume, drift_headroom):
    fixed = [base + weights @ issued for base, weights in models]
    weights = np.stack([w[:, active] for _, w in models])
    slots = len(active)
    ident = np.eye(slots)
    terms, offsets, limits = [*ident], [np.zeros(2) for _ in active], [cap**2] * slots

    def append(row, offset, limit):
        terms.append(row)
        offsets.append(offset)
        limits.append(limit)

    # At t1061 inherit the exact issued first action; t1084 is the
    # screened exception. All other nonzero transitions are rate-limited.
    append(ident[0], -issued[0, :2], slew**2 - issued[0, 2]**2)
    for k in range(1, slots):
        if active[k] != resume:
            append(ident[k] - ident[k - 1], np.zeros(2), slew**2)
    append(ident[-1], -issued[active[-1] + 1, :2], slew**2)
    a, b = begin * 20, resume * 20
    for values, coefficient in zip(fixed, weights, strict=True):
        shift = values[b] - values[a]
        append(coefficient[b] - coefficient[a], shift[:2], (drift - drift_headroom)**2 - shift[2]**2)
    return np.stack(fixed), weights, np.array(terms), np.array(offsets), np.array(limits)


def support(fixed, weights, samples, pos, centers, envelope, slots):
    sample = np.array(samples, dtype=int)
    lag, case, tick = sample.T
    relative = pos[lag, tick] - centers[case, tick]
    normal = relative / np.linalg.norm(relative, axis=1, keepdims=True).clip(min=1.e-9)
    coeff = weights[lag, tick, :, None] * normal[:, None, :2]
    hard = np.column_stack((coeff.reshape(-1, slots * 2), -np.ones(len(sample))))
    baseline = np.einsum('ij,ij->i', normal, fixed[lag, tick] - centers[case, tick])
    return LinearConstraint(hard, envelope[tick] - baseline, np.inf)


def trajectory(fixed, weights, issues, centers, envelope):
    coords = fixed.copy()
    coords[..., :2] += np.einsum('ltk,kj->ltj', weights, issues)
    clear = np.linalg.norm(coords[:, None] - centers, axis=3) - envelope
    return coords, clear


def close_samples(values, count):
    i = np.argsort(values, axis=2)[:, :, :count]
    return [tuple(map(int, (lag, case, t))) for lag in range(i.shape[0])
            for case in range(i.shape[1]) for t in i[lag, case]]


def solve(active, fixed_issued, models, centers, envelope, begin, resume, config):
    slots = len(active)
    cap, slew, drift = (config[n] for n in ('cap_m_s', 'slew_m_s', 'drift_m'))
    fixed, weights, coeff, offsets, limit = layout(active, fixed_issued, models, cap, slew, drift, begin, resume,
                                                   config['drift_headroom_m'])

    def norms(y):
        u = coeff @ y[:-1].reshape(slots, 2) + offsets
        return limit - (u * u).sum(axis=1)

    def norm_jac(y):
        u = coeff @ y[:-1].reshape(slots, 2) + offsets
        grad = (-2.*coeff[:, :, None] * u[:, None]).reshape(len(coeff), slots * 2)
        return np.column_stack((grad, np.zeros(len(coeff))))

    merit = np.zeros(slots * 2 + 1)
    merit[-1] = -1.
    output = []
    for seed in config['seed_bearings_deg']:
        angle = np.deg2rad(seed)
        goal = .40 * np.array([np.cos(angle), np.sin(angle)])
        current = fixed_issued[0, :2]
        steps = []
        for i in active:
            if i == resume:
                current = goal
            else:
                tail = i >= active[-1] - 2
                target = fixed_issued[active[-1] + 1, :2] if tail else goal
                current = cap_step(target, current, slew)
            steps.append(current)
        u = np.array(steps)
        y = np.r_[u.flatten(), -.25]
        samples = set()
        best = None
        for iteration in range(config['refinements']):
            pos, clear = trajectory(fixed, weights, y[:-1].reshape(slots, 2), centers, envelope)
            samples.update(close_samples(clear, config['witnesses_per_case']))
            constraints = support(fixed, weights, sorted(samples), pos, centers, envelope, slots)
            found = minimize(lambda z: -z[-1], y, jac=lambda z: merit, method='SLSQP',
                             constraints=[constraints, dict(type='ineq', fun=norms, jac=norm_jac)],
                             bounds=[(-cap, cap)] * (slots * 2) + [(-.6, .67)],
                             options=dict(ftol=2.e-10, maxiter=config['max_solver_iterations']))
            y = found.x
            u = y[:-1].reshape(slots, 2)
            _, clear = trajectory(fixed, weights, u, centers, envelope)
            worst = float(clear.min())
            if norms(y).min() >= -2.e-8 and (best is None or worst > best['clearance_m']):
                best = dict(clearance_m=worst, solver_min_margin=float(y[-1]), seed_deg=seed,
                            solver_success=bool(found.success), solver_status=int(found.status),
                            iteration=iteration, support_samples=len(samples), issue_xy=u.copy())
            if best is not None and best['clearance_m'] >= config['wanted_margin_m']:
                break
        if best is not None:
            output.append(best)
    return output


def second_readback(active, solution, fixed_issued, prep, geometry, arch, initial, cfg):
    api, times, centers, envelope, _ = prep
    start, end, lane = (cfg[key] for key in ('start', 'score_end', 'lane'))
    base, velocity, pending = initial
    commands = fixed_issued.copy()
    commands[active, :2] = solution['issue_xy']
    matrices = [api['propagation'](times, end-start, lag, base, velocity, pending) for lag in cfg['lags_s']]
    zero = [(step - start) * .10 for step in cfg['zero_window']]
    scores, drift, speed = api['score_bank'](matrices, centers, envelope, times, commands[None], zero, False)
    origin = torch.tensor(load(checked(geometry))['env_origins'])
    _, _, vertex, faces = scene_geometry(origin, cfg['scene_seed'], command_routes=True)
    solids = np.asarray(vertex)[np.asarray(faces).reshape(-1, 3)]
    body = envelope - .34
    raw = np.array([r['raw_event'][lane]['velocity_w_m_s'] for r in arch[start:end]])
    worst = None
    static_floor = np.inf
    restoration = {}
    for i, (p, a, v, b) in enumerate(matrices):
        position = p + a @ commands
        motion = v + b @ commands
        cell = static_distance(position, solids) - body
        static_floor = min(static_floor, float(cell.min()))
        for j, center in enumerate(centers):
            ds = np.linalg.norm(position - center, axis=1) - envelope
            margin = np.minimum(cell, ds)
            near = int(margin.argmin())
            if worst is None or margin[near] < worst['clearance_m']:
                worst = dict(clearance_m=float(margin[near]), lag_s=cfg['lags_s'][i], case=j,
                             time_s=start*.1 + float(times[near]))
        start_restore = cfg['recovery_anchor'] - start
        samples = motion[20::20]
        progress = (samples * raw).sum(axis=1) >= .60 * (raw * raw).sum(axis=1)
        # This is actuator progress only. Feasibility/recovery requires a
        # causal controller, fresh observations and separate scoring.
        found = next((k for k in range(start_restore,len(progress)-4)
                      if np.linalg.norm(raw[k:k+5], axis=1).min() > .03 and progress[k:k+5].all()), None)
        restoration[str(cfg['lags_s'][i])] = dict(first_five_progress_s=None if found is None else (found+5-start_restore)*.1)
    return dict(physical=worst, static_m=static_floor, mover_min_by_case=scores[0].tolist(),
                maximum_drift_m=float(drift[0]), max_speed_after_deadline_m_s=float(speed[0]),
                open_loop_progress=restoration, issued_w=commands.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if stop_paths() or shutil.disk_usage(args.output.parent).free < 30 * 2**30 + 10 * 2**20:
        raise RuntimeError('STOP or system disk reserve')
    here=Path(__file__).resolve()
    registry=here.with_name('INPUTS.json')
    inputs = load(registry)
    cfg=inputs['protocol']
    origin = load(checked(inputs['parking_inputs']))
    first_data=load(checked(inputs['first_issue_menu']))
    drift_data=load(checked(inputs['drift_bound']))
    allowed = {row['menu_id'] for row in drift_data['modes']['full_speed_restart']['rows'] if row['worst_upper_m']>=.15}
    assert len(allowed)==cfg['expected_firsts']
    menu=[row for row in first_data['menu'] if row['menu_id'] in allowed]
    assert git('rev-parse','HEAD')==origin['protocol']['v60_source'] and not git('branch','--show-current')
    api=runpy.run_path(str(checked(inputs['parking_script'])))
    start, last, score_end, lane = (cfg[k] for k in ('start','optimization_end','score_end','lane'))
    begin, resume = (x-start for x in cfg['zero_window'])
    with gzip.open(checked(origin['trace']),'rt') as stream:
        next(stream)
        arch=[json.loads(next(stream)) for _ in range(score_end)]
    assert [r['step'] for r in arch]==list(range(len(arch)))
    initial_row=load(checked(origin['hard_backup']))['arm']['records'][0]
    assert initial_row['step']==start
    initial=(np.array(initial_row['position_before']),np.array(initial_row['velocity_before']),
             np.array(arch[start-1]['v_final_w'][lane]))
    dense=np.memmap(checked(origin['dense']), '<f4','r', shape=(1200,21,32,7))
    joint=np.concatenate((dense[start:score_end,:-1,lane].reshape(-1,7),dense[score_end-1,-1:,lane])).astype(float)
    assert np.allclose(joint[0,:3], initial[0],atol=2.e-5,rtol=0.)
    time=np.arange(len(joint))*.005
    centers=api['mover_bank'](joint, origin['protocol']['cases'], time, start)
    envelope=joint[:,6]+.34
    short=(time <= (last-start)*.10+1.e-8)
    opts = [api['propagation'](time[short],score_end-start,lag,*initial)[:2] for lag in cfg['lags_s']]
    commands=np.array([row['raw_event'][lane]['velocity_w_m_s'] for row in arch[start:score_end]])
    assert not commands[begin:resume].any() and np.linalg.norm(commands[resume])>.03
    tail=cfg['return_to_ccg']-start
    active=np.r_[np.arange(1,begin),np.arange(resume,tail)]
    fixed=np.zeros_like(commands)
    fixed[tail:]=commands[tail:]
    assert strict_tail(fixed[tail:],cfg)
    results=[]
    prep=api,time,centers,envelope,opts
    for current in menu:
        assert not stop_paths(),stop_paths()
        fixed[0]=current['world']
        found=solve(active,fixed,opts,centers[:,short],envelope[short],begin,resume,cfg)
        ranked=sorted(found,key=lambda r:r['clearance_m'],reverse=True)
        scans=[]
        for row in ranked:
            exact=second_readback(active,row,fixed,prep,origin['geometry'],arch,initial,cfg)
            sol={k:v for k,v in row.items() if k!='issue_xy'}
            scans.append(dict(optimization=sol,exact=exact))
        results.append(dict(menu_id=current['menu_id'],raw_first_action=current['action'],seeds=scans))
        print(json.dumps(dict(menu_id=current['menu_id'],seeds=len(scans),
                              best=max((x['exact']['physical']['clearance_m'] for x in scans),default=None))),flush=True)
    assert not stop_paths(),stop_paths()
    print(json.dumps(dict(raw=save(args.output,dict(registry=pin(registry),script=pin(here),
        case_order=list(origin['protocol']['cases']),results=results)))),flush=True)


def strict_tail(tail,cfg):
    return (np.linalg.norm(tail,axis=1).max()<=cfg['cap_m_s'] and
            np.linalg.norm(np.diff(tail,axis=0),axis=1).max()<=cfg['slew_m_s'])


if __name__=='__main__':
    main()
