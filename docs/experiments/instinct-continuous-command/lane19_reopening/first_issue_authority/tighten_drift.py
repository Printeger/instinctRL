"""Upper-bound the remaining issued-neutral candidates with a stop-drift constraint."""
import argparse
import gzip
import json
from pathlib import Path
import runpy

import numpy as np

from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked, git


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not stop_paths(), stop_paths()
    script = Path(__file__).resolve()
    registry = script.with_name('DRIFT-INPUTS.json')
    contract = load(registry)
    inputs = load(checked(contract['first_issue_inputs']))
    cfg = inputs['protocol']
    root = load(checked(inputs['onset_inputs']))
    assert git('rev-parse', 'HEAD') == root['v60_commit'] and not git('branch', '--show-current')
    raw = load(checked(contract['screen']))
    assert raw['script'] == contract['first_issue_script']
    api = runpy.run_path(str(checked(contract['first_issue_script'])))
    with gzip.open(checked(root['trace']), 'rt') as stream:
        next(stream)
        rows = [json.loads(next(stream)) for _ in range(cfg['end'])]
    time, centers, envelope, _, models, cases, lags, begin, resume = api['fixture'](cfg, inputs, rows)
    chosen = (time >= contract['time_after_start'][0]) & (time <= contract['time_after_start'][1])
    # One failing fixed-attitude mover and lag is sufficient to refute a row.
    sphere = centers[cases.index('hard_positive'), chosen]
    elapsed = time[chosen]
    gate = contract['max_drift_m']
    divisions = cfg['support_angles']
    bearing = np.arange(divisions) * (2. * np.pi / divisions)
    direction = np.stack((np.cos(bearing), np.sin(bearing)), axis=1)
    divisor = np.cos(np.pi / divisions)
    b0 = int(np.rint((.1 * begin) / .005))
    b1 = int(np.rint((.1 * resume) / .005))
    modes = ('free_planar_resume', 'full_speed_restart')
    entries = {}
    for mode in modes:
        head = [row for row in raw['menu'] if row[mode]['worst_upper_m'] >= .15]
        first = np.array([r['world'] for r in head])
        pre = api['projected_caps'](first, direction, begin, cfg['speed_m_s'], cfg['slew_m_s'])
        earlier = np.concatenate((pre[:, :, divisions // 2:], pre[:, :, :divisions // 2]), axis=2)
        scored = [dict(menu_id=r['menu_id'], by_lag={}) for r in head]
        for lag, (base, weights) in zip(lags, models, strict=True):
            assert not stop_paths(), stop_paths()
            delta = weights[b1, 1:begin] - weights[b0, 1:begin]
            fixed = base + weights[:, :1] * first[:, None, :]
            offset = (fixed[:, b1, :2] - fixed[:, b0, :2]) @ direction.T
            diff = fixed[:, chosen] - sphere
            distances = diff[:, :, :2] @ direction.T
            restart = cfg['speed_m_s'] * np.ones(weights.shape[1] - resume)
            if mode == 'free_planar_resume':
                restart = np.minimum(restart, cfg['slew_m_s'] * np.arange(1, len(restart) + 1))
            post = weights[chosen, resume:] @ restart
            minima = np.full_like(distances, np.inf)
            for price in contract['drift_multipliers']:
                coefficients = weights[chosen, 1:begin] - price * delta
                supported = (np.einsum('tk,bka->bta', np.maximum(coefficients, 0.), pre, optimize=True)
                             + np.einsum('tk,bka->bta', np.maximum(-coefficients, 0.), earlier, optimize=True))
                bound = distances + post[None, :, None] + supported + price * (gate - offset[:, None, :])
                minima = np.minimum(minima, bound)
            planar = np.maximum(0., minima.max(axis=2)) / divisor
            upper = np.hypot(planar, diff[:, :, 2]) - envelope[chosen] + 1.e-9
            for row, value in zip(scored, upper, strict=True):
                i = int(value.argmin())
                row['by_lag'][str(lag)] = dict(m=float(value[i]), time_s=cfg['start']*.1 + float(elapsed[i]))
        for row in scored:
            row['worst_upper_m'] = min(result['m'] for result in row['by_lag'].values())
        entries[mode] = dict(screened=len(scored), survivors=sum(r['worst_upper_m'] >= .15 for r in scored),
                             best=max(scored, key=lambda r:r['worst_upper_m']), rows=scored)
    print(json.dumps(dict(raw=save(args.output, dict(registry=pin(registry),script=pin(script),modes=entries)),
                          modes={mode:{k:v for k,v in data.items() if k!='rows'} for mode,data in entries.items()})),flush=True)


if __name__ == '__main__':
    main()
