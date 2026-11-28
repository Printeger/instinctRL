"""Locate the first registration failure on the closed CPU reserve trajectories."""
import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import torch

from instinctlab.control.measured_curve_deflection import _return_shapes, fresh_motion_history
from instinctlab.control.registered_mover_entry import _registered_fit, _stationary_translation
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked


def explain(scan,state,rays):
    checks={}
    functions={x.__code__:x.__name__ for x in (_return_shapes,_stationary_translation,_registered_fit)}

    def monitor(frame,event,result):
        name=functions.get(frame.f_code)
        if not name:
            return None
        if event=='return':
            local=frame.f_locals
            diag=dict(accepted=result is not None)
            for k in ('ready','tracked','supported','visible','distance','radii','speed','length'):
                if k in local:
                    diag[k]=np.asarray(local[k]).tolist()
            if 'near' in local and 'other' in local and name=='_return_shapes':
                diag['other_gap_m']=float((local['other']-local['near'])[0])
            if 'planes' in local:
                diag['plane_count']=len(local['planes'])
            if 'directions' in local:
                diag['minimum_singular']=float(np.linalg.svd(local['directions'],compute_uv=False)[-1])
            if 'translation' in local:
                diag['projection_residual_m']=float(np.linalg.norm(
                    local['directions']@local['translation'].T-local['measured'],axis=0).max())
            if 'independent' in local:
                diag['split_delta_m']=float(np.linalg.norm(local['independent']-local['translation'],axis=1).max())
            checks[name]=diag
        return monitor

    previous=sys.gettrace()
    try:
        sys.settrace(monitor)
        _registered_fit(scan,state,rays,entry=False)
    finally:
        sys.settrace(previous)
    checks['fresh']=bool(fresh_motion_history(state.reshape(1,4,13))[0])
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    assert not stop_paths(),stop_paths()
    file=Path(__file__).resolve()
    reg=file.with_name('REGISTRATION-INPUTS.json')
    refs=load(reg)
    live=load(checked(refs['run']))
    earlier=load(checked(refs['readback_inputs']))
    configuration=load(checked(earlier['reserve_inputs']))
    tool=runpy.run_path(str(checked(earlier['reserve_script'])))
    ctx=tool['inputs'](configuration)
    cfg=configuration['protocol']
    start=refs['start']
    lane=cfg['lane']
    h,rows=ctx['scene'],ctx['archive']
    r=np.array([h['body_to_world'](rows[start]['attitude_wxyz'][lane],e) for e in np.eye(3)])
    initial_frames=[h['reconstruct'](ctx['initial'],rows[t],rows[t-1],lane,h['rays'],ctx['mesh'])[0]
                    for t in range(start-3,start+1)]
    surface=ctx['raw'][start:refs['through']+1,:-1,lane].reshape(-1,7).astype(float)
    time=np.arange(len(surface))*.005
    mover=ctx['motion']['mover_bank'](surface,cfg['cases'],time,start)
    case_names=list(cfg['cases'])
    outcomes={}
    for mode,group in live['outcomes'].items():
        outcomes[mode]={}
        for case,arm in group.items():
            assert arm['case']['attitude']=='fixed'
            history=np.array(rows[start]['actor_state'][lane],dtype=np.float32)
            frames=list(initial_frames)
            records={x['step']:x for x in arm['records']}
            body_centers=mover[case_names.index(case)]
            samples=[]
            for step in range(start+1,refs['through']+1):
                assert not stop_paths(),stop_paths()
                archived=np.array(rows[step]['actor_state'][lane][-1],dtype=np.float32)
                archived[:3]=0.
                archived[3:6]=r@np.array([0.,0.,-1.])
                archived[6:9]=r@np.array(records[step]['request_w'])
                archived[9:12]=r@np.array(records[step-1]['issued_w'])
                history=np.vstack((history[1:],archived))
                frames=frames[1:]+[h['render'](np.array(records[step]['position_before']),r,body_centers[(step-start)*20],
                                              ctx['mesh'])]
                if step<refs['from']:
                    continue
                with torch.inference_mode():
                    checks=explain(torch.tensor(np.concatenate(frames)).reshape(1,12,59,360),
                                   torch.tensor(history).reshape(1,52),ctx['ray'])
                policy=records[step]['proof']
                if policy.get('reason')=='missing_fresh_registration':
                    assert not checks['fresh'] or not checks['_registered_fit']['accepted']
                samples.append(dict(step=step,checks=checks))
            outcomes[mode][case]=samples
    assert not stop_paths(),stop_paths()
    print(json.dumps(dict(raw=save(args.output,dict(registry=pin(reg),script=pin(file),outcomes=outcomes)),
        first_loss={m:{c:next((x for x in seq if not x['checks']['_registered_fit']['accepted']),None)
                       for c,seq in group.items()} for m,group in outcomes.items()})),flush=True)


if __name__=='__main__':
    main()
