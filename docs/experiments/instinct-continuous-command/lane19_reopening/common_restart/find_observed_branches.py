"""Test the physical benefit of a restart choice after an identical stop prefix."""
import argparse
import gzip
import json
from pathlib import Path
import runpy
from unittest.mock import patch

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
    registry = script.with_name('BRANCH-INPUTS.json')
    inputs = load(registry)
    config = load(checked(inputs['common_inputs']))
    cfg = config['protocol'] | inputs['changes']
    origin = load(checked(config['parking_inputs']))
    assert git('rev-parse','HEAD')==origin['protocol']['v60_source'] and not git('branch','--show-current')
    common = load(checked(inputs['common_results']))
    assert common['registry']==inputs['common_inputs'] and common['script']==inputs['common_script']
    api = runpy.run_path(str(checked(config['parking_script'])))
    runtime = runpy.run_path(str(checked(inputs['common_script'])))['main'].__globals__
    normal_layout = runtime['layout']
    winner = max(((item['menu_id'], seed) for item in common['results'] for seed in item['seeds']),
                 key=lambda x:x[1]['exact']['physical']['clearance_m'])
    start, lane, last, end = (cfg[k] for k in ('start','lane','optimization_end','score_end'))
    with gzip.open(checked(origin['trace']),'rt') as stream:
        next(stream)
        archive = [json.loads(next(stream)) for _ in range(end)]
    model_data = load(checked(origin['hard_backup']))['arm']['records'][0]
    initial = (np.array(model_data['position_before']),np.array(model_data['velocity_before']),
               np.array(archive[start-1]['v_final_w'][lane]))
    dense = np.memmap(checked(origin['dense']), '<f4','r', shape=(1200,21,32,7))
    raw = np.concatenate((dense[start:end,:-1,lane].reshape(-1,7), dense[end-1,-1:,lane])).astype(float)
    seconds = np.arange(len(raw))*.005
    movers = api['mover_bank'](raw, origin['protocol']['cases'], seconds, start)
    sphere = raw[:,6]+.34
    shorter = seconds <= (last-start)*.10+1.e-8
    models = [api['propagation'](seconds[shorter],end-start,lag,*initial)[:2] for lag in cfg['lags_s']]
    start_zero, resume = (x-start for x in cfg['zero_window'])
    tail = cfg['return_to_ccg']-start
    active = np.arange(resume,tail)
    fixed = np.array(winner[1]['exact']['issued_w'])
    fixed[active] = 0.
    assert not fixed[start_zero:resume].any()

    def branch_layout(active, issued, models, *args):
        fixed, weights, norms, offsets, limits = normal_layout(active, issued, models, *args)
        # The previous issued velocity is zero at the first restart
        # packet; this is the screened full-speed exception.
        assert not norms[-len(models):].any()
        drift = fixed[:, resume*20] - fixed[:, start_zero*20]
        assert np.linalg.norm(drift, axis=1).max() <= cfg['drift_m']
        ignored = [len(active), *range(len(norms) - len(models), len(norms))]
        return fixed, weights, np.delete(norms,ignored,axis=0), np.delete(offsets,ignored,axis=0), np.delete(limits,ignored)

    all_cases = list(origin['protocol']['cases'])
    result = {}
    for name, cases in inputs['branches'].items():
        assert not stop_paths(), stop_paths()
        subgroup = movers[[all_cases.index(case) for case in cases]]
        with patch.dict(runtime,layout=branch_layout):
            options = runtime['solve'](active,fixed,models,subgroup[:,shorter],sphere[shorter],start_zero,resume,cfg)
        rows = []
        for option in options:
            prepared = api, seconds, subgroup, sphere, models
            exact = runtime['second_readback'](active,option,fixed,prepared,origin['geometry'],archive,initial,cfg)
            assert exact['issued_w'][:resume]==fixed[:resume].tolist()
            rows.append(dict(optimization={k:v for k,v in option.items() if k!='issue_xy'},exact=exact))
        found = max(rows,key=lambda x:x['exact']['physical']['clearance_m'],default=None)
        result[name] = dict(cases=cases, winning_search=found, all_searches=rows)
        summary = None if found is None else {k:v for k,v in found['exact'].items() if k!='issued_w'}
        print(json.dumps(dict(branch=name, winner=summary, searches=len(rows))),flush=True)
    assert not stop_paths(), stop_paths()
    print(json.dumps(dict(raw=save(args.output,dict(registry=pin(registry),script=pin(script),
                         common_start=winner[0],common_seed=winner[1]['optimization']['seed_deg'],
                         original_schedule=winner[1],branches=result)))),flush=True)


if __name__=='__main__':
    main()
