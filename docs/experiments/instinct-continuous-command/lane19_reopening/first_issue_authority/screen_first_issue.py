"""Refine lane 19 strict-neutral authority once the true first issue is known."""
import argparse
import gzip
import json
from pathlib import Path
import runpy
import shutil
import sys

import numpy as np
import torch

from instinctlab.control.escape_passage import advance_command_response
from instinctlab.control.escape_recovery import MOVER_PASSAGE_REPROOF_25
from instinctlab.control.measured_curve_deflection import fresh_motion_history
from instinctlab.control.pipeline import ControlCoreConfig
from instinctlab.control.registered_mover_entry import _registered_fit
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked, git
from training_workloads.unified_h1.command_replay_audit import body_to_world


def exact_menu(choose, base, initial, rows, cfg):
    lane, onset = cfg['lane'], cfg['start']
    anchor = rows[onset]
    h = runpy.run_path(base['fixture']['path'])['execute'].__globals__
    assert initial == h['initial']
    assert np.array_equal(h['rays'], np.fromfile(base['rays']['path'], dtype='<f4').reshape(-1, 3))
    triangles = h['triangles']
    origin = h['origins'][lane].numpy()
    local = ((triangles.max(axis=1) >= origin - 45.) & (triangles.min(axis=1) <= origin + 45.)).all(axis=1)
    frames = [h['reconstruct'](initial, rows[step], rows[step - 1], lane, h['rays'], triangles[local])
              for step in range(onset - 3, onset + 1)]
    scan = torch.tensor(np.concatenate([frame for frame, _ in frames])).reshape(1, 12, 59, 360)
    option_cfg = ControlCoreConfig(ics=h['ics'])
    option = h['ClosedControl'](option_cfg, h['ray_tensor'], MOVER_PASSAGE_REPROOF_25).escape_option
    observers = h['observer']()
    for row in rows[:onset + 1]:
        state = torch.tensor(row['actor_state'][lane]).reshape(1, 52)
        observers = [advance_command_response(*obs, state, lag=lag)
                     for obs, lag in zip(observers, h['lags'], strict=True)]
    coast = torch.stack([obs[1][0] for obs in observers]).numpy()
    selected = torch.tensor([anchor['selected_action'][lane]])
    registration = _registered_fit(scan, state, h['ray_tensor'], entry=False)
    assert registration is not None and bool(fresh_motion_history(state.reshape(1, 4, 13))[0])
    stock = option.issued(scan, state, selected, option_cfg)
    assert np.allclose(stock.v_final_b[0].numpy(), anchor['v_final_b'][lane], atol=2.e-6, rtol=0.)

    captured = {}
    names = ('actions', 'first', 'following', 'eligible', 'lower', 'backup', 'ranks')

    def watch(frame, event, value):
        if frame.f_code is choose.__code__:
            if event == 'return':
                captured.update({name: np.array(frame.f_locals[name], copy=True) for name in names})
            return watch
        return None

    tracer = sys.gettrace()
    try:
        sys.settrace(watch)
        action, winner = choose(option, scan, state, selected, option_cfg, registration, coast, True)
    finally:
        sys.settrace(tracer)
    eligible = np.flatnonzero(captured['eligible'])
    assert len(eligible) == cfg['expected_certificates'] and winner['decision'] == 'onset'
    assert np.allclose(winner['body'], cfg['expected_body'], atol=2.e-6, rtol=0.)
    assert any(np.array_equal(captured['actions'][i], action.numpy()) for i in eligible)
    rotation = np.array([body_to_world(anchor['attitude_wxyz'][lane], v) for v in np.eye(3)])
    first, actions = captured['first'][eligible], captured['actions'][eligible]
    previous = state[0, 48:51].numpy()
    assert np.linalg.norm(first - previous, axis=1).max() <= .30001
    receiver = option.issued(scan.expand(len(first), -1, -1, -1), state.expand(len(first), -1),
                             torch.from_numpy(actions), option_cfg)
    issue_error = float(np.max(np.abs(receiver.v_final_b.numpy() - first)))
    assert issue_error <= 2.e-7
    assert (receiver.metrics['beta'][:, 0] >= .90).all() and not receiver.metrics['emergency'][:, 0].any()
    world = first @ rotation
    indices = [dict(menu_id=int(i), action=raw.tolist(), body=b.tolist(), world=w.tolist(),
                    follow_b=captured['following'][i].tolist(), beta=float(beta),
                    certificate_m=float(captured['lower'][i]), backup_m=float(captured['backup'][i]),
                    rank_m=float(captured['ranks'][i]))
               for i, raw, b, w, beta in zip(eligible, actions, first, world, receiver.metrics['beta'][:, 0], strict=True)]
    return world, indices, winner, max(delta or 0. for _, delta in frames), previous @ rotation, issue_error


def projected_caps(first, angles, count, speed, slew):
    support = first[:, :2] @ angles.T
    result = []
    for _ in range(1, count):
        # Zero is an allowed immediate brake. Including its projection also
        # covers issue histories that stop and start again before raw neutral.
        support = np.maximum(0., np.minimum(speed, support + slew))
        result.append(support)
    return np.stack(result, axis=1)


def fixture(config, inputs, rows):
    p_cfg = load(checked(inputs['parking_inputs']))['protocol']
    api = runpy.run_path(str(checked(inputs['parking_script'])))
    lane, start, end = (config[k] for k in ('lane', 'start', 'end'))
    origin = load(checked(inputs['hard_pair']))['arm']['records'][0]
    assert origin['step'] == start
    p, v = np.array(origin['position_before']), np.array(origin['velocity_before'])
    pending = np.array(rows[start - 1]['v_final_w'][lane])
    native = np.memmap(checked(inputs['dense']), '<f4', 'r', shape=(1200, 21, 32, 7))
    dense = np.concatenate((native[start:end, :-1, lane].reshape(-1, 7), native[end - 1, -1:, lane])).astype(float)
    assert np.allclose(dense[0, :3], p, atol=2.e-5, rtol=0.)
    time = np.arange(len(dense)) * .005
    centers = api['mover_bank'](dense, p_cfg['cases'], time, start)
    sphere = .34 + dense[:, 6]
    commands = np.array([r['raw_event'][lane]['velocity_w_m_s'] for r in rows[start:end]])
    begin, resume = (i - start for i in p_cfg['zero_window'])
    assert begin == 10 and resume == 24 and not commands[begin:resume].any()
    assert np.linalg.norm(commands[resume]) > .03
    # With at least one fully neutral packet, the post-resume commands are
    # independent of how the approach was driven.
    actual = api['append_requests'](np.zeros((1, begin, 3)), commands, begin, resume, config['slew_m_s'])[0]
    models = [api['propagation'](time, end-start, lag, p, v, pending)[:2] for lag in p_cfg['lags_s']]
    return time, centers, sphere, actual, models, list(p_cfg['cases']), p_cfg['lags_s'], begin, resume


def scores(first, time, centers, sphere, actual, models, cases, lags, begin, resume, cfg):
    theta = np.arange(cfg['support_angles']) * 2. * np.pi / cfg['support_angles']
    angles = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    enclosure = np.cos(np.pi / len(angles))
    cap, slew = cfg['speed_m_s'], cfg['slew_m_s']
    pre = projected_caps(first, angles, begin, cap, slew)
    modes = ('free_planar_resume', 'full_speed_restart', 'latest_ccg_resume')
    best = np.full((len(modes), len(first), len(lags), len(cases)), np.inf)
    when = np.zeros(best.shape, dtype=int)

    for lag_i, (base, weights) in enumerate(models):
        post = weights[:, resume:] @ np.minimum(cap, slew * np.arange(1, weights.shape[1] - resume + 1))
        unrestricted = cap * weights[:, resume:].sum(axis=1)
        ccg = weights[:, resume:] @ actual[resume:]
        for start in range(0, len(first), 12):
            stop = start + 12
            available = np.einsum('tk,bka->bta', weights[:, 1:begin], pre[start:stop], optimize=True)
            first_path = base + first[start:stop, None, :] * weights[None, :, :1]
            for ii, center in enumerate(centers):
                relative = first_path - center
                for jj, mode in enumerate(modes):
                    d = relative + ccg if mode == 'latest_ccg_resume' else relative
                    projected = available + d[:, :, :2] @ angles.T
                    if mode == 'free_planar_resume':
                        projected += post[None, :, None]
                    if mode == 'full_speed_restart':
                        projected += unrestricted[None, :, None]
                    # Every realized horizontal displacement has a sample
                    # heading within pi/N. Its projected norm there is at
                    # least its true length times cos(pi/N).
                    radius = np.maximum(0., projected.max(axis=2)) / enclosure
                    clear = np.hypot(radius, d[:, :, 2]) - sphere + 1.e-9
                    idx = clear.argmin(axis=1)
                    best[jj, start:stop, lag_i, ii] = clear[np.arange(len(idx)), idx]
                    when[jj, start:stop, lag_i, ii] = idx

    return {name: [
        [dict(zip(cases, (dict(upper_m=float(best[j, i, k, l]), time_s=cfg['start']*.10 + float(time[when[j, i, k, l]]))
                         for l in range(len(cases))), strict=True))
         for k in range(len(lags))] for i in range(len(first))] for j, name in enumerate(modes)}


def summarize(rows, values, lags):
    all_out = {}
    for mode, bound in values.items():
        minima, per_lag = [], []
        for row, per in zip(rows, bound, strict=True):
            floors = [min((info['upper_m'], case, info['time_s']) for case, info in scores.items())
                      for scores in per]
            i = int(np.argmin([x[0] for x in floors]))
            worst, case, instant = floors[i]
            minima.append(worst)
            per_lag.append([f[0] for f in floors])
            row[mode] = dict(worst_upper_m=worst, lag_s=lags[i], case=case, time_s=instant,
                             upper_by_lag=dict(zip(map(str, lags), per, strict=True)))
        winner = int(np.argmax(minima))
        hard = [min(scores['hard_positive']['upper_m'] for scores in per) for per in bound]
        all_out[mode] = dict(survivors=int((np.asarray(minima) >= .15).sum()),
                             upper_m=float(minima[winner]), menu_id=rows[winner]['menu_id'],
                             best_upper_by_lag=dict(zip(map(str, lags), np.max(per_lag, axis=0).tolist(), strict=True)),
                             best_hard_positive_upper_m=float(max(hard)),
                             hard_positive_survivors=int((np.asarray(hard) >= .15).sum()))
    return all_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if stop_paths() or shutil.disk_usage(args.output.parent).free < 30 * 2**30 + 10 * 2**20:
        raise RuntimeError('STOP or system disk reserve')
    me = Path(__file__).resolve()
    registry = me.with_name('INPUTS.json')
    inputs = load(registry)
    cfg = inputs['protocol']
    earlier = load(checked(inputs['onset_inputs']))
    for name, ref in earlier.items():
        if name != 'v60_commit':
            checked(ref)
    assert git('rev-parse', 'HEAD') == earlier['v60_commit'] and not git('branch', '--show-current')
    with gzip.open(earlier['trace']['path'], 'rt') as stream:
        initial = json.loads(next(stream))
        rows = [json.loads(next(stream)) for _ in range(cfg['end'])]
    assert [r['step'] for r in rows] == list(range(len(rows)))
    choose = runpy.run_path(str(checked(inputs['closed_replay_script'])))['choose']
    with torch.inference_mode():
        first, menu, witness, delta, previous_world, issue_error = exact_menu(choose, earlier, initial, rows, cfg)
    print(json.dumps(dict(menu_size=len(menu), range_error_m=delta, winner=witness)), flush=True)
    data = fixture(cfg, inputs, rows)
    lags = data[6]
    bounds = scores(first, *data, cfg)
    summary = summarize(menu, bounds, lags)
    assert not stop_paths(), stop_paths()
    result = dict(input_registry=pin(registry), script=pin(me), reconstruction_delta_m=delta,
                  repeated_ics_issue_error_m_s=issue_error,
                  winner_from_original_rank=witness, previous_body_issue_rotated_w=previous_world.tolist(),
                  population=len(menu), summary=summary, menu=menu)
    print(json.dumps(dict(raw=save(args.output, result), summary=summary)), flush=True)


if __name__ == '__main__':
    main()
