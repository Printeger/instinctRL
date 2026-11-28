"""Resolve adjacent split packets and compare what their common prefix actually sees."""
import argparse
import json
from pathlib import Path
import runpy
import shutil

import numpy as np
import torch

from instinctlab.control.registered_mover_entry import _DISTANCE_ERROR, _registered_fit
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked


def measurements(inputs, data):
    cfg = data['config']
    lane, onset = cfg['lane'], cfg['start']
    fixture = load(checked(inputs['reopening_inputs']))
    for name, ref in fixture.items():
        if name != 'v60_commit':
            checked(ref)
    h = runpy.run_path(fixture['fixture']['path'])['execute'].__globals__
    assert data['first']==h['initial']
    archive = data['archive']
    origin = h['origins'][lane].numpy()
    triangles = h['triangles']
    near = ((triangles.max(axis=1)>=origin-45.) & (triangles.min(axis=1)<=origin+45.)).all(axis=1)
    mesh = triangles[near]
    reconstructed = [h['reconstruct'](data['first'],archive[k],archive[k-1],lane,h['rays'],mesh)[0]
                     for k in range(onset-3,onset+1)]
    rotation = np.array([h['body_to_world'](archive[onset]['attitude_wxyz'][lane],axis) for axis in np.eye(3)])
    diagonal = rotation @ (np.array([1.,1.,0.])/np.sqrt(2.))
    gravity = rotation @ np.array([0.,0.,-1.])
    initial_state = np.array(archive[onset]['actor_state'][lane],dtype=np.float32)
    cases = list(data['origin']['protocol']['cases'])
    cap = inputs['observed_through']
    samples = []
    for lag, (initial,w,_,_) in zip(cfg['lags_s'],data['models'],strict=True):
        positions = initial+w@data['positive']
        banks = [list(reconstructed) for _ in cases]
        history = initial_state
        for step in range(onset,cap+1):
            assert not stop_paths(),stop_paths()
            i = (step-onset)*20
            if step>onset:
                row = archive[step]
                current = np.array(row['actor_state'][lane][-1],dtype=np.float32)
                current[:3] = 0.
                current[3:6] = gravity
                current[6:9] = rotation@np.array(row['raw_event'][lane]['velocity_w_m_s'])
                current[9:12] = rotation@data['positive'][step-onset-1]
                history = np.vstack((history[1:],current))
                for j,bank in enumerate(banks):
                    bank[:] = bank[1:]+[h['render'](positions[i],rotation,data['movers'][j,i],mesh)]
            reference = np.concatenate(banks[cases.index('nominal')])
            state = torch.tensor(history).reshape(1,52)
            features = {}
            for case, bank in zip(cases,banks,strict=True):
                received = np.concatenate(bank)
                scan = torch.tensor(received).reshape(1,12,59,360)
                reg = _registered_fit(scan,state,h['ray_tensor'],entry=False)
                feature = dict(history_equal=bool(np.array_equal(received,reference)),
                               changed_latest_ranges=int((received[9]!=reference[9]).sum()),
                               newest_range_delta_m=float(np.max(abs(received[9]-reference[9]))),
                               fit=None)
                if reg is not None:
                    fit, _,dt = reg
                    feature['fit'] = dict(center_projection_m=float(fit.center@diagonal),
                                          velocity_projection_m_s=float(fit.velocity[0]@diagonal),
                                          velocity_error_allowance_m_s=2.*_DISTANCE_ERROR/dt)
                features[case] = feature
            samples.append(dict(lag_s=lag, step=step, observation_age_s=float(history[-1,12]), features=features))
    return samples


def compare(samples):
    answer = []
    hard_pair = ('hard_positive','nominal')
    for step in sorted({x['step'] for x in samples}):
        observed = [x['features'] for x in samples if x['step']==step]
        pair = [tuple(row[key] for key in hard_pair) for row in observed]
        same = all(x['history_equal'] for x,_ in pair)
        fit = all(x['fit'] and n['fit'] for x,n in pair)
        gaps = [x['fit']['center_projection_m']-n['fit']['center_projection_m'] for x,n in pair] if fit else []
        # The one-millimetre allowance is the frozen geometric assumption.
        # This matched comparison does not construct a standalone selector.
        groups = ([x['fit']['velocity_projection_m_s']-x['fit']['velocity_error_allowance_m_s']
                   for row in observed for key,x in row.items() if key.endswith('_positive') and x['fit']],
                  [x['fit']['velocity_projection_m_s']+x['fit']['velocity_error_allowance_m_s']
                   for row in observed for key,x in row.items() if not key.endswith('_positive') and x['fit']])
        answer.append(dict(step=step,hard_histories_identical=same,hard_registered_all_lags=fit,
                           min_hard_paired_projection_gap_m=min(gaps) if gaps else None,
                           hard_gap_after_error_m=min(gaps)-2.*_DISTANCE_ERROR if gaps else None,
                           all_group_velocity_gap_m_s=min(groups[0])-max(groups[1])
                           if len(groups[0])==2*len(observed) and len(groups[1])==3*len(observed) else None))
    return answer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    assert not stop_paths() and shutil.disk_usage(args.output.parent).free>=30*2**30+10*2**20
    here=Path(__file__).resolve()
    registry=here.with_name('CLOCK-INPUTS.json')
    refs=load(registry)
    prior = load(checked(refs['split_result']))
    assert prior['inputs']==refs['split_inputs'] and prior['script']==refs['split_script']
    api=runpy.run_path(str(checked(refs['split_script'])))
    data=api['load_inputs'](load(checked(refs['split_inputs'])))
    adjacent = api['experiment'](data,data['config'] | {'splits':refs['adjacent']})
    with torch.inference_mode():
        observed=measurements(refs,data)
    summary=compare(observed)
    assert all(q['hard_histories_identical'] for q in summary if q['step']<=1064)
    assert not stop_paths(),stop_paths()
    raw=save(args.output,dict(registry=pin(registry),script=pin(here),adjacent=adjacent,
                              observation_summary=summary,observations=observed))
    print(json.dumps(dict(raw=raw,summary=summary)),flush=True)


if __name__=='__main__':
    main()
