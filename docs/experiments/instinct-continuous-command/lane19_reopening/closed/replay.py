"""Replay the v60 lane 19 reopening in a CPU actor/ICS/response loop."""
import argparse
import gzip
import json
from pathlib import Path
import runpy
import shutil

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid

from instinctlab.control.escape_passage import advance_command_response, observed_path_clearance
from instinctlab.control.escape_recovery import MOVER_PASSAGE_REPROOF_25
from instinctlab.control.measured_curve_deflection import fresh_motion_history
from instinctlab.control.observed_mover_preview import _rank_continuations
from instinctlab.control.pipeline import ControlCoreConfig
from instinctlab.control.registered_mover_entry import _piecewise_clearance, _registered_fit
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.replay_clearance import confirmed_starts, projected_clearance, static_distance
from tools.r6_direct.source import checked, git

PROTECTED = (
    'applied', 'retreating', 'braking', 'ranked', 'holding', 'static_stop_verified',
    'crossing_restart_verified', 'intercepted', 'intercept_continued', 'handed_off',
    'tracked_resume_used', 'composed_crossing_verified', 'brake_recovery_verified',
    'offset_stopped', 'measured_mover_active',
)


def check_resources(output):
    if stop_paths():
        raise RuntimeError(f'STOP {stop_paths()}')
    if shutil.disk_usage(output.parent).free < 30 * 2**30 + 10 * 2**20:
        raise RuntimeError('system disk reserve')


def issued_bank(option, scan, state, config, velocity):
    action = option.action_for_velocity(velocity, state[:, 48:51].expand_as(velocity), config.governor)
    result = [option.issued(scan.expand(len(batch), -1, -1, -1), state.expand(len(batch), -1),
                            batch, config) for batch in action.split(24)]
    valid = torch.cat([(r.metrics['beta'][:, 0] >= .90) & (r.metrics['emergency'][:, 0] == 0.)
                       for r in result]).numpy()
    return action, torch.cat([r.v_final_b for r in result]), valid


def choose(option, scan, state, selected, config, registered, coast, starting):
    fit, basis, spacing = registered
    queued = state[0, 48:51]
    request = state[0, 45:48]
    neutral = float(request.norm()) < .03
    planar = (coast @ basis[:2]) @ basis.T
    stock = option.issued(scan, state, selected, config)
    baseline = stock.v_final_b
    floor = float(_piecewise_clearance(fit, planar, queued.numpy(), baseline.numpy(),
                                       baseline.numpy(), .60, 1., spacing)[0])
    rank = float(_rank_continuations(fit, coast, queued.numpy(), baseline.numpy())[0])
    event = dict(stock_floor_m=floor, stock_rank_m=rank)
    if bool(stock.metrics['emergency'][0, 0]) or (starting and floor >= .39):
        return None, event | {'reason': 'stock_emergency_or_above_floor'}

    theta = np.arange(72) * np.pi / 36.
    directions = np.stack((np.cos(theta), np.sin(theta)), axis=1) @ basis.T
    targets = state.new_tensor(np.concatenate([s * directions for s in (.15, .225, .2999, .375, .4499)]))
    if not starting:
        targets = torch.cat((targets, baseline, torch.zeros_like(baseline)))
    _, following, valid = issued_bank(option, scan, state, config, targets)
    change = following - queued
    limited = queued + change * (.30 / change.norm(dim=1, keepdim=True).clamp_min(.30))
    actions, first, passed = issued_bank(option, scan, state, config, limited)
    valid &= passed & ((first - queued).norm(dim=1).numpy() <= .30001)
    first_n, following_n = first.numpy(), following.numpy()
    planned = _piecewise_clearance(fit, planar, queued.numpy(), first_n, following_n, .60, 1., spacing)
    backup = _piecewise_clearance(fit, planar, queued.numpy(), first_n, np.zeros_like(following_n), .10, .90, spacing)
    wall = torch.cat([observed_path_clearance(scan, option.detector._full_rays, state.new_tensor(coast[1:2]),
                                             batch[None], hold_s=.60)[0] for batch in following.split(24)])
    wall = wall.numpy() - .055 - .10 * (first - following).norm(dim=1).numpy()
    lower = np.minimum(np.minimum(planned, backup), wall)
    ranks = _rank_continuations(fit, coast, queued.numpy(), following_n)
    eligible = valid & (lower >= .39)
    if starting:
        eligible &= ranks >= max(.15, rank + .08)
    else:
        eligible &= ranks >= .15
    event.update(certified=int(eligible.sum()), best_ics_floor_m=float(lower[valid].max()) if valid.any() else None)
    if not eligible.any():
        return None, event | {'reason': 'no_verified_continuation'}

    aligned = ((first @ request >= .60 * (request @ request)) &
               (following @ request >= .60 * (request @ request))).numpy()
    if not starting and neutral and eligible[-1] and float(first[-1].norm()) <= .001:
        index, decision = len(first) - 1, 'verified_zero'
    elif not starting and not neutral and eligible[-2] and aligned[-2] and torch.allclose(first[-2], baseline[0], atol=1.e-5, rtol=0.):
        return None, event | {'reason': 'verified_stock_handback'}
    else:
        if not starting and not neutral and (eligible & aligned).any():
            eligible &= aligned
        index = int(np.where(eligible, ranks, -np.inf).argmax())
        decision = 'onset' if starting else 'renewal'
    return actions[index], event | dict(decision=decision, selected_floor_m=float(lower[index]),
        stop_backup_m=float(backup[index]), rank_m=float(ranks[index]), body=first[index].tolist(),
        following=following[index].tolist())


class Trial:
    def __init__(self, option, observers):
        self.option = option
        self.observers = observers
        self.original = option.apply_recovery
        self.phase = 'waiting'
        self.event = {}
        self.stock_action = None
        option.apply_recovery = self.apply

    def apply(self, policy, action, config):
        selected = self.original(policy, action, config)
        self.stock_action = selected[0].tolist()
        if self.phase not in ('start', 'active'):
            return selected
        starting = self.phase == 'start'
        scan, state = policy['lidar_grid'], policy['state_vec']
        option = self.option
        veto = [name for name in PROTECTED if bool(getattr(option, name)[0])]
        if bool(option.zero_crossing_commit[0] or option.observed_turn_brake_floor[0]):
            veto.append('committed_or_braking')
        if veto or (starting and not bool(option.detour_used[0] or option.tangent_used[0])):
            self.event = dict(reason='existing_protection_or_no_entry', protection=veto)
            self.phase = 'ended'
            return selected
        if not bool(fresh_motion_history(state.reshape(1, 4, 13))[0]):
            self.event = dict(reason='stale_history')
            self.phase = 'ended'
            return selected
        registered = _registered_fit(scan, state, option.detector._full_rays, entry=False)
        if registered is None:
            self.event = dict(reason='no_registration')
            self.phase = 'ended'
            return selected
        length = np.linalg.norm(registered[0].acceleration[0], axis=1)
        if starting and (min(length) < .32 or max(length) > .80 or
                         not .06 <= float(state[0, 45:48].norm()) <= .30):
            self.event = dict(reason='entry_bounds', registered_acceleration=length.tolist())
            self.phase = 'ended'
            return selected
        coast = torch.stack([value[1][0] for value in self.observers]).numpy()
        action, self.event = choose(option, scan, state, selected, config, registered, coast, starting)
        self.event['registered_acceleration'] = length.tolist()
        if action is None:
            self.phase = 'ended'
        else:
            self.phase = 'active'
            selected[0] = action
            option.supersede_mover_edge_choice([0])
            option.measured_mover_active[0] = True
        return selected


def score(record, zero, anchor, targets, centers, triangles, radius):
    first = record[0]['step']
    rows = record[zero[0] - first:zero[1] - first]
    ticks = np.array([r['speed_end_m_s'] <= .07 for r in rows])
    stable = np.flatnonzero(np.logical_and.accumulate(ticks[::-1])[::-1])
    before = np.array([r['position_before'] for r in record])
    initial_v = np.array([r['velocity_before'] for r in record])
    measured = np.array([r['velocity_after'] for r in record])
    nominal, _ = projected_clearance(before, initial_v, targets, centers, triangles, radius)
    feasible = (np.linalg.norm(targets, axis=1) > .03) & (nominal >= .15)
    tracked = feasible & (np.sum(targets * measured, axis=1) >= .60 * np.sum(targets * targets, axis=1))
    offers = confirmed_starts(feasible[anchor - first:])
    found = confirmed_starts(tracked[anchor - first:])
    worst = min(record, key=lambda r: min(r['mover_m'], r['static_m']))
    return dict(minimum_m=min(worst['mover_m'], worst['static_m']), worst_time_s=worst['minimum_time_s'],
        min_mover_m=min(r['mover_m'] for r in record), min_static_m=min(r['static_m'] for r in record),
        zero_window=dict(frozen_stop_eligible=False, sustained_speed_below_07_s=(int(stable[0]) + 1) * .1
                         if len(stable) >= 5 else None, max_drift_m=max(r['zero_drift_m'] for r in rows),
                         entry_speed_m_s=float(np.linalg.norm(rows[0]['velocity_before'])),
                         exit_speed_m_s=rows[-1]['speed_end_m_s']),
        recovery=dict(anchor=anchor, observed_offer_through=record[-1]['step'] - 24,
                      first_feasible=None if not len(offers) else anchor + int(offers[0]),
                      confirmed_step=None if not len(found) else anchor + int(found[0]) + 4,
                      elapsed_s=None if not len(found) else (int(found[0]) + 5) * .1),
        feasible_packets=int(feasible.sum()), tracked_packets=int(tracked.sum()),
        feasible_rmse_m_s=float(np.sqrt(np.square((measured - targets)[feasible]).sum(axis=1).mean()))
                        if feasible.any() else None)


def replay(h, rows, frames, dense, mesh, solids, attitude, rotation, commands, mode, case, cfg, output, native_radius):
    start, warm, end = cfg['start'], cfg['warm'], cfg['end_exclusive']
    lane = cfg['lane']
    control = h['ClosedControl'](ControlCoreConfig(ics=h['ics']), h['ray_tensor'], MOVER_PASSAGE_REPROOF_25)
    option = control.escape_option
    observers = h['observer']()
    trial = Trial(option, observers)
    def advance(state):
        for index, tau in enumerate(h['lags']):
            observers[index] = advance_command_response(*observers[index], state, lag=tau)
    for row in rows[:warm]:
        state = torch.tensor(row['actor_state'][lane]).reshape(1, 52)
        advance(state)
        option.command_offsets, option.command_coast = advance_command_response(option.command_offsets,
                                                                                  option.command_coast, state)
        observer = option.curve_deflection
        for index, tau in enumerate(h['lags']):
            observer.offsets[index], observer.coast[index] = advance_command_response(
                observer.offsets[index], observer.coast[index], state, lag=tau)

    centers = np.concatenate((dense[start:end, :-1, lane, 3:6].reshape(-1, 3),
                              dense[end - 1, -1:, lane, 3:6])).astype(float)
    steps = np.arange(len(centers))
    delay = (case['fork_step'] - start) * 20
    seconds = (steps - delay) * .005
    rise = np.clip(seconds / .15, 0., 1.)
    fall = np.clip((1.20 - seconds) / .20, 0., 1.)
    acceleration = case['sign'] * .22 * rise**2 * (3. - 2. * rise) * fall**2 * (3. - 2. * fall)
    translation = cumulative_trapezoid(cumulative_trapezoid(acceleration, dx=.005, initial=0.), dx=.005, initial=0.)
    assert not translation[steps <= delay].any()
    centers += translation[:, None] * np.array([1., 1., 0.]) / np.sqrt(2.)
    base = np.array(rows[start - 1]['root_state_after_w'][lane])
    p, v = base[:3], base[7:10]
    pending = np.array(rows[start - 1]['v_final_w'][lane])
    lag = cfg['actuator_lag_s']
    times = np.arange(21)[:, None] * .005
    response = lag * (-np.expm1(-times / lag))
    records, errors = [], []
    zero_anchor = None
    for step in range(warm, end):
        check_resources(output)
        row = rows[step]
        at = min(start, step) if case['attitude'] == 'fixed' else step
        r = rotation[at]
        if step <= start:
            delivered = [frames[t] for t in range(step - 3, step + 1)]
            history = np.array(row['actor_state'][lane], dtype=np.float32)
        else:
            latest = np.array(row['actor_state'][lane][-1], dtype=np.float32)
            if case['attitude'] == 'fixed':
                latest[:3] = 0.
                latest[3:6] = r @ np.array([0., 0., -1.])
            latest[6:9] = r @ commands[step]
            latest[9:12] = rotation[at if case['attitude'] == 'fixed' else at - 1] @ pending
            history = np.vstack((history[1:], latest))
            delivered = delivered[1:] + [h['render'](p, r, centers[(step - start) * 20], mesh)]
        scan = torch.tensor(np.concatenate(delivered)).reshape(1, 12, 59, 360)
        state = torch.tensor(history).reshape(1, 52)
        policy = dict(lidar_grid=scan, state_vec=state)
        advance(state)
        trial.event = {}
        if step == start and mode == 'renewal':
            trial.phase = 'start'
        action = torch.tensor([row['action'][lane]]) if step <= start else h['actor'](policy).mean_action
        result = control.step(policy, action, torch.tensor(attitude[at:at + 1], dtype=torch.float32))
        issued = result.v_final_w[0].numpy()
        issued_b = result.v_final_b[0].numpy()
        beta = float(result.ics_metrics['beta'][0, 0])
        emergency = bool(result.ics_metrics['emergency'][0, 0])
        if step < start or step == start and mode == 'stock':
            if step >= start - 8:
                errors.append(float(np.linalg.norm(issued - row['v_final_w'][lane])))
        if step < start:
            continue
        event = trial.event
        if 'body' in event:
            event['dispatch_error_m_s'] = max(float(np.linalg.norm(issued_b - event['body'])),
                                             float(np.linalg.norm(issued - np.array(event['body']) @ r)))
            if event['dispatch_error_m_s'] > 2.e-5 or beta < .90 or emergency:
                raise ValueError(f'proof/dispatch mismatch at {step}: {event}')
        if step == start:
            onset_stock = dict(stock_action_error=max(abs(np.array(trial.stock_action) - row['selected_action'][lane])),
                               finite_edge=bool(option.detour_used[0]))
            if mode == 'stock':
                onset_stock.update(body_error_m_s=float(np.linalg.norm(issued_b - row['v_final_b'][lane])),
                                   world_error_m_s=errors[-1], beta_error=abs(beta - row['ics_beta'][lane]),
                                   archived_finite_edge=row['escape_finite_edge'][lane])
        if step == start and mode == 'renewal':
            event['witness_reproduced'] = (
                event.get('decision') == 'onset' and
                np.allclose(event['body'], cfg['expected_first_body_m_s'], atol=2.e-5, rtol=0.) and
                abs(event['stock_floor_m'] - cfg['expected_stock_floor_m']) < 2.e-5 and
                abs(event['selected_floor_m'] - cfg['expected_entry_floor_m']) < 2.e-5
            )
        if step == cfg['zero_window'][0]:
            zero_anchor = p.copy()
        path = p + times * pending + response * (v - pending)
        next_v = pending + np.exp(-.1 / lag) * (v - pending)
        index = (step - start) * 20
        radii = dense[step, :, lane, 6]
        mover = np.linalg.norm(path - centers[index:index + 21], axis=1) - .34 - radii
        walls = static_distance(path, solids) - radii
        records.append(dict(step=step, action=action[0].tolist(), issued_w=issued.tolist(), request_w=commands[step].tolist(),
            beta=beta, emergency=emergency, velocity_before=v.tolist(), velocity_after=next_v.tolist(),
            position_before=p.tolist(), speed_end_m_s=float(np.linalg.norm(next_v)),
            mover_m=float(mover.min()), static_m=float(walls.min()),
            minimum_time_s=step * .1 + int(np.minimum(mover, walls).argmin()) * .005,
            zero_drift_m=None if zero_anchor is None else float(np.linalg.norm(path - zero_anchor, axis=1).max()),
            proof=event))
        p, v, pending = path[-1], next_v, issued
    metrics = score(records, cfg['zero_window'], cfg['recovery_anchor'], commands[start:end],
                    centers[::20][:-1], mesh, native_radius)
    return dict(mode=mode, case=case, warm_issue_error_m_s=max(errors), onset_stock=onset_stock,
                metrics=metrics, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=('nominal', 'hard_positive', 'hard_negative', 'early_positive', 'early_negative'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    check_resources(args.output)
    path = Path(__file__).resolve()
    registry = path.with_name('INPUTS.json')
    pins = load(registry)
    base = load(checked(pins['base_manifest']))
    for key, ref in base.items():
        if key != 'v60_commit':
            checked(ref)
    assert git('rev-parse', 'HEAD') == base['v60_commit'] and not git('branch', '--show-current')
    cfg = pins['protocol']
    lane = cfg['lane']
    with gzip.open(base['trace']['path'], 'rt') as stream:
        initial = json.loads(next(stream))
        rows = [json.loads(next(stream)) for _ in range(cfg['end_exclusive'])]
    dense = np.memmap(checked(pins['dense']), dtype='<f4', mode='r', shape=(1200, 21, 32, 7))
    metrics = load(checked(pins['native_metrics']))
    radius = metrics['models']['candidate']['layout_routes'][lane]['physics_tick_clearance']['maximum_measured_link_envelope_m']
    h = runpy.run_path(base['fixture']['path'])['execute'].__globals__
    assert initial == h['initial'] and np.array_equal(h['rays'], np.fromfile(base['rays']['path'], dtype='<f4').reshape(-1, 3))
    assert [r['step'] for r in rows] == list(range(len(rows))) and not any(r['reset'][lane] for r in rows)
    origin = h['origins'][lane].numpy()
    def nearby(triangles):
        return triangles[((triangles.max(axis=1) >= origin - 45.) & (triangles.min(axis=1) <= origin + 45.)).all(axis=1)]
    mesh = nearby(h['triangles'])
    _, _, vertices, indices = h['scene_geometry'](h['origins'], cfg['scene_seed'], command_routes=True)
    solids = nearby(np.asarray(vertices)[np.asarray(indices).reshape(-1, 3)])
    frames = {s: h['reconstruct'](initial, rows[s], rows[s - 1], lane, h['rays'], mesh)[0]
              for s in range(cfg['warm'] - 3, cfg['start'] + 1)}
    attitude = np.array([r['attitude_wxyz'][lane] for r in rows])
    rotation = np.array([[h['body_to_world'](q, axis) for axis in np.eye(3)] for q in attitude])
    commands = np.array([r['raw_event'][lane]['velocity_w_m_s'] for r in rows])
    assert not commands[slice(*cfg['zero_window'])].any()
    case = cfg['cases'][args.case]
    with torch.inference_mode():
        arms = [replay(h, rows, frames, dense, mesh, solids, attitude, rotation, commands, mode, case, cfg, args.output, radius)
                for mode in ('stock', 'renewal')]
    saved = save(args.output, dict(inputs=pin(registry), script=pin(path), case=args.case, arms=arms))
    summaries = [dict(mode=a['mode'], warm_issue_error_m_s=a['warm_issue_error_m_s'], onset_stock=a['onset_stock'],
                      metrics=a['metrics'],
                      decisions=[(r['step'], r['proof'].get('decision', r['proof'].get('reason')))
                                 for r in a['records'] if r['proof']]) for a in arms]
    print(json.dumps(dict(raw=saved, candidate_onset=arms[1]['records'][0]['proof'], arms=summaries)), flush=True)


if __name__ == '__main__':
    main()
