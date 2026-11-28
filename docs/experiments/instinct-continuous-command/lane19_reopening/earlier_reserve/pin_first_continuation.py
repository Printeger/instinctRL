"""Keep the certified first continuation instead of a truth-optimized second issue."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
from unittest.mock import patch

import numpy as np
import torch

from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    assert not stop_paths() and shutil.disk_usage(args.output.parent).free>=30*2**30+10*2**20
    script=Path(__file__).resolve()
    registry=script.with_name('CONTINUATION-INPUTS.json')
    refs=load(registry)
    run=load(checked(refs['readback_inputs']))
    prior=load(checked(refs['rejected_pair']))
    assert all(r['first_rejection']['step']==1052 and
               r['first_rejection']['proof']['reason']=='following_plan_ics_mismatch'
               for r in prior['cases'].values())
    reserve=load(checked(run['reserve_inputs']))
    raw=load(checked(run['physical_receipt']))
    tool=runpy.run_path(str(checked(run['reserve_script'])))
    audit=runpy.run_path(str(checked(refs['readback_script'])))['audit']
    data=tool['inputs'](reserve)
    engine=data['engine']
    anchor=raw['anchor']
    mode=run['protocol']['mode']
    cfg=reserve['protocol']
    profile=reserve['modes'][mode]
    physic=raw['results']['modes'][mode]['best']
    head=physic['head']
    h, rows, lane = data['scene'],data['archive'],cfg['lane']
    attitude=np.array([r['attitude_wxyz'][lane] for r in rows])
    rotation=np.array([[h['body_to_world'](q,v) for v in np.eye(3)] for q in attitude])
    following=np.asarray(head['planned_next_b'])@rotation[anchor]
    assert abs(following[2]) < 2.e-6
    batch=tool['prepare'](data,cfg,anchor)
    local=batch['config']
    compiled=engine['compile_run'](batch,local,profile,head)
    assert compiled['active'][0]==1 and (compiled['mapping'][:,0]==0).all()
    solver=engine['solve'].__globals__
    minimize=solver['minimize']

    def certificate_continuation(*args,**kwargs):
        bounds=kwargs['bounds'].copy()
        bounds[:2]=[(float(x),float(x)) for x in following[:2]]
        return minimize(*args,**(kwargs | {'bounds':bounds}))

    with patch.dict(solver,minimize=certificate_continuation):
        result=engine['solve'](compiled,batch,local,profile,profile['seeds'])
    fit = None if result is None else result['clearance_m']
    print(json.dumps(dict(anchor=anchor,following_w=following.tolist(),clearance_m=fit)),flush=True)
    output=dict(registry=pin(registry),script=pin(script),open_loop=result,following_w=following.tolist(),cases={})
    if result is not None and fit>=.15:
        assert np.allclose(result['group_outputs'][0]['issues'][1],following,atol=2.e-6,rtol=0.)
        source=runpy.run_path(str(checked(reserve['closed_script'])))['replay'].__globals__
        original=load(checked(run['original_replay_inputs']))['protocol']
        settings=run['protocol']|{'return_to_ccg':cfg['return_to_ccg']}
        replay=original|{'start':anchor,'actuator_lag_s':settings['lag_s']}
        native=load(checked(run['native_metrics']))
        radius=native['models']['candidate']['layout_routes'][lane]['physics_tick_clearance']['maximum_measured_link_envelope_m']
        with torch.inference_mode():
            frames={k:h['reconstruct'](data['initial'],rows[k],rows[k-1],lane,h['rays'],data['mesh'])[0]
                    for k in range(replay['warm']-3,anchor+1)}
            requests=np.array([row['raw_event'][lane]['velocity_w_m_s'] for row in rows])
            override=audit(source,settings,dict(head=head,best=result),rotation[anchor],replay)
            with patch.dict(source,Trial=override):
                for name in settings['cases']:
                    case=dict(replay['cases'][name],attitude='fixed')
                    arm=source['replay'](h,rows,frames,data['raw'],data['mesh'],data['solids'],attitude,rotation,
                                          requests,'renewal',case,replay,args.output,radius)
                    first=next((r for r in arm['records'] if 'reason' in r['proof']),None)
                    output['cases'][name]=dict(trajectory=arm,rejection=first)
                    print(json.dumps(dict(case=name,decisions=[(r['step'],r['proof']) for r in arm['records'] if r['proof']],
                                          metrics=arm['metrics'])),flush=True)
    assert not stop_paths(),stop_paths()
    print(json.dumps(dict(raw=save(args.output,output))),flush=True)


if __name__=='__main__':
    main()
