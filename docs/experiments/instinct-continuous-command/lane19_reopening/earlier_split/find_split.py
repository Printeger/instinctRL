"""Find the last split before neutral using the frozen positive-branch issue history."""
import argparse
import gzip
import json
from pathlib import Path
import runpy
import shutil
from unittest.mock import patch

import numpy as np

from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked, git


def load_inputs(refs):
    base = load(checked(refs['common_inputs']))
    origin = load(checked(base['parking_inputs']))
    cfg = base['protocol'] | refs['settings']
    assert git('rev-parse','HEAD') == origin['protocol']['v60_source'] and not git('branch','--show-current')
    runtime = runpy.run_path(str(checked(refs['common_script'])))['main'].__globals__
    api = runpy.run_path(str(checked(base['parking_script'])))
    prior = load(checked(refs['branch_results']))
    winner = prior['branches']['positive']['winning_search']['exact']
    positive = np.array(winner['issued_w'])
    start, end, lane = (cfg[k] for k in ('start','score_end','lane'))
    with gzip.open(checked(origin['trace']),'rt') as f:
        first = json.loads(next(f))
        archive = [json.loads(next(f)) for _ in range(end)]
    state = load(checked(origin['hard_backup']))['arm']['records'][0]
    initial = (np.array(state['position_before']),np.array(state['velocity_before']),
               np.array(archive[start-1]['v_final_w'][lane]))
    buffer = np.memmap(checked(origin['dense']), '<f4','r', shape=(1200,21,32,7))
    points = np.concatenate((buffer[start:end,:-1,lane].reshape(-1,7),buffer[end-1,-1:,lane])).astype(float)
    t = np.arange(len(points))*.005
    movers = api['mover_bank'](points,origin['protocol']['cases'],t,start)
    envelope = .34 + points[:,6]
    model = [api['propagation'](t,end-start,lag,*initial) for lag in cfg['lags_s']]
    assert np.allclose(initial[0],points[0,:3],atol=2.e-5,rtol=0.)
    return dict(config=cfg,origin=origin,runtime=runtime,api=api,prior=prior,winner=winner,
                positive=positive,first=first,archive=archive,initial=initial,time=t,movers=movers,
                envelope=envelope,models=model)


def experiment(data, cfg):
    active_cases = cfg['released_cases']
    all_cases = list(data['origin']['protocol']['cases'])
    centers = data['movers'][[all_cases.index(name) for name in active_cases]]
    time, envelope = data['time'], data['envelope']
    initial, positive = data['initial'], data['positive']
    short = time <= (cfg['optimization_end']-cfg['start'])*.1+1.e-8
    solve_models = [(base[short], w[short]) for base,w,_,_ in data['models']]
    begin, resume = (i-cfg['start'] for i in cfg['zero_window'])
    handback = cfg['return_to_ccg']-cfg['start']
    runtime = data['runtime']
    layout = runtime['layout']

    def prefix_layout(active, issued, model, *args):
        fixed, weights, norms, offsets, limits = layout(active,issued,model,*args)
        previous = issued[active[0]-1]
        index = len(active)
        offsets[index] = -previous[:2]
        limits[index] = cfg['slew_m_s']**2 - previous[2]**2
        return fixed, weights, norms, offsets, limits

    results = {}
    for split in cfg['splits']:
        assert not stop_paths(),stop_paths()
        offset = split-cfg['start']
        assert 1 <= offset < begin
        active = np.r_[np.arange(offset,begin),np.arange(resume,handback)]
        fixed = positive.copy()
        fixed[active] = 0.
        with patch.dict(runtime,layout=prefix_layout):
            solutions = runtime['solve'](active,fixed,solve_models,centers[:,short],envelope[short],begin,resume,cfg)
        candidates = []
        for option in solutions:
            prepared = data['api'],time,centers,envelope,solve_models
            exact = runtime['second_readback'](active,option,fixed,prepared,data['origin']['geometry'],
                                               data['archive'],initial,cfg)
            issued = np.array(exact['issued_w'])
            assert np.array_equal(issued[:offset],positive[:offset]) and not issued[begin:resume].any()
            allowed_deltas = np.delete(np.linalg.norm(np.diff(issued,axis=0),axis=1),[begin-1,resume-1])
            assert allowed_deltas.max() <= cfg['slew_m_s']+2.e-7 and np.linalg.norm(issued,axis=1).max() <= .4500002
            if exact['maximum_drift_m']<=cfg['drift_m'] and exact['max_speed_after_deadline_m_s']<=.07:
                exact['physical']['case_name'] = active_cases[exact['physical']['case']]
                candidates.append(dict(optimization={k:v for k,v in option.items() if k!='issue_xy'},exact=exact))
        best = max(candidates,key=lambda x:x['exact']['physical']['clearance_m'],default=None)
        joint = min(best['exact']['physical']['clearance_m'],data['winner']['physical']['clearance_m']) if best else None
        results[str(split)] = dict(search_count=len(solutions),winner=best,joint_clearance_m=joint,candidates=candidates)
        print(json.dumps(dict(split=split,clearance=joint,
              released=None if best is None else best['exact']['physical'])),flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    assert not stop_paths() and shutil.disk_usage(args.output.parent).free >= 30*2**30+10*2**20
    here = Path(__file__).resolve()
    reg = here.with_name('INPUTS.json')
    data = load_inputs(load(reg))
    result = experiment(data,data['config'])
    assert not stop_paths(),stop_paths()
    raw = save(args.output,dict(inputs=pin(reg),script=pin(here),positive_reference=data['winner'],
                                split_results=result))
    print(json.dumps(dict(raw=raw)),flush=True)


if __name__=='__main__':
    main()
