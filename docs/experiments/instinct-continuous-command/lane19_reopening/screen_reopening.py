"""Check exact v60 observations near lane 19's first reopening after a reversal."""
import argparse
import gzip
import json
from pathlib import Path
import runpy

import numpy as np
import torch
from instinctlab.control.escape_passage import advance_command_response, observed_path_clearance
from instinctlab.control.escape_recovery import MOVER_PASSAGE_REPROOF_25
from instinctlab.control.measured_curve_deflection import _return_shapes, fresh_motion_history
from instinctlab.control.observed_mover_preview import _rank_continuations
from instinctlab.control.pipeline import ControlCoreConfig
from instinctlab.control.registered_mover_entry import _piecewise_clearance, _registered_fit
from tools.r6_direct.command_ray_shadow import reconstruct
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load,save
from tools.r6_direct.source import checked,git


def screen_actions(option,scan,state,cfg,registration,coast,issued):
    fit,basis,spacing=registration
    request=state[0,45:48]
    previous=state[0,48:51]
    planar_coast=(coast@basis[:2])@basis.T
    queued=previous.numpy()
    baseline=issued.numpy()
    previous_floor=float(_piecewise_clearance(fit,planar_coast,queued,baseline,baseline,.60,1.,spacing)[0])
    previous_rank=float(_rank_continuations(fit,coast,queued,baseline)[0])
    angle=np.arange(72)*np.pi/36.
    xy=np.stack((np.cos(angle),np.sin(angle)),axis=1)@basis.T
    goals=state.new_tensor(np.concatenate((.15*xy,.225*xy,.2999*xy,.375*xy,.4499*xy)))

    def issue(velocity):
        actions=option.action_for_velocity(velocity,previous.expand_as(velocity),cfg.governor)
        runs=[option.issued(scan.expand(len(b),-1,-1,-1),state.expand(len(b),-1),b,cfg) for b in actions.split(24)]
        valid=torch.cat([(p.metrics['beta'][:,0]>=.90)&(p.metrics['emergency'][:,0]==0.) for p in runs])
        return torch.cat([p.v_final_b for p in runs]),valid

    following,found=issue(goals)
    step=following-previous
    first,valid=issue(previous+step*(.30/step.norm(dim=1,keepdim=True).clamp_min(.30)))
    valid &= found&((first-previous).norm(dim=1)<=.30001)
    follows=following[valid]
    first=first[valid]
    ranges=dict(previous_floor_m=previous_floor,previous_rank_m=previous_rank,ics_pairs=len(first))
    if not len(first):
        return ranges
    planned=_piecewise_clearance(fit,planar_coast,queued,first.numpy(),follows.numpy(),.60,1.,spacing)
    backup=_piecewise_clearance(fit,planar_coast,queued,first.numpy(),np.zeros_like(follows),.10,.90,spacing)
    static=torch.cat([observed_path_clearance(scan,option.detector._full_rays,state.new_tensor(coast[1:2]),x[None],hold_s=.60)[0]
                      for x in follows.split(24)]).numpy()-.055-.10*(first-follows).norm(dim=1).numpy()
    min_clear=np.minimum(np.minimum(planned,backup),static)
    rank=_rank_continuations(fit,coast,queued,follows.numpy())
    along=((follows@request)/(request@request).clamp_min(1.e-6)).numpy()
    accepted=min_clear>=.39
    close=accepted&(rank>=max(.15,previous_rank+.08))

    def summarize(mask):
        if not mask.any():
            return None
        i=int(np.where(mask,rank,-np.inf).argmax())
        return dict(first_b=first[i].tolist(),following_b=follows[i].tolist(),floor_m=float(min_clear[i]),
                    backup_m=float(backup[i]),projected_m=float(rank[i]),request_along_fraction=float(along[i]))

    return dict(**ranges,certified_count=int(accepted.sum()),rank_gain_count=int(close.sum()),
                request_aligned_count=int((close&(along>=.60)).sum()),best=summarize(accepted),
                best_with_rank_gain=summarize(close),best_aligned=summarize(close&(along>=.60)),
                best_floor_m=float(min_clear.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    file=Path(__file__).resolve()
    registry=file.with_name('INPUTS.json')
    inputs=load(registry)
    assert not stop_paths(),stop_paths()
    for key,ref in inputs.items():
        if key!='v60_commit':
            checked(ref)
    assert git('rev-parse','HEAD')==inputs['v60_commit'] and not git('branch','--show-current')
    lane=19
    with gzip.open(inputs['trace']['path'],'rt') as stream:
        initial=json.loads(next(stream))
        rows=[json.loads(next(stream)) for _ in range(1063)]
    h=runpy.run_path(inputs['fixture']['path'])['execute'].__globals__
    assert initial==h['initial'] and np.array_equal(h['rays'],np.fromfile(inputs['rays']['path'],dtype='<f4').reshape(-1,3))
    origin=h['origins'][lane].numpy()
    triangles=h['triangles']
    mesh=triangles[((triangles.max(axis=1)>=origin-45.)&(triangles.min(axis=1)<=origin+45.)).all(axis=1)]
    cfg=ControlCoreConfig(ics=h['ics'])
    option=h['ClosedControl'](cfg,h['ray_tensor'],MOVER_PASSAGE_REPROOF_25).escape_option
    rays=h['ray_tensor']
    model=h['observer']()
    history=[]
    results=[]
    with torch.inference_mode():
        for step,row in enumerate(rows):
            state=torch.tensor(row['actor_state'][lane]).reshape(1,52)
            model=[advance_command_response(*old,state,lag=tau) for old,tau in zip(model,h['lags'],strict=True)]
            if step<1049:
                continue
            assert not stop_paths(),stop_paths()
            frame,delta=reconstruct(initial,row,rows[step-1],lane,h['rays'],mesh)
            history=(history+[(frame,delta)])[-4:]
            if len(history)<4:
                continue
            scan=torch.tensor(np.concatenate([x[0] for x in history])).reshape(1,12,59,360)
            action=torch.tensor([row['selected_action'][lane]])
            issued=option.issued(scan,state,action,cfg)
            fit=_registered_fit(scan,state,rays,entry=False)
            shape=_return_shapes(scan,state,rays,entry=False,align_gravity=True)
            closures=None if shape is None else (-shape[0].norm(dim=1).diff()).tolist()
            row_out=dict(step=step,max_native_range_error=max((x[1] or 0.) for x in history),
                         native_beta=row['ics_beta'][lane],native_edge=row['escape_finite_edge'][lane],
                         request_speed=float(state[0,45:48].norm()),closures_m=closures,
                         fit=fit is not None,fresh=bool(fresh_motion_history(state.reshape(1,4,13))[0]))
            if fit is not None and row_out['fresh']:
                coast=torch.stack([value[1][0] for value in model]).numpy()
                row_out.update(accels=np.linalg.norm(fit[0].acceleration[0],axis=1).tolist(),
                               **screen_actions(option,scan,state,cfg,fit,coast,issued.v_final_b))
            results.append(row_out)
    assert not stop_paths(),stop_paths()
    print(save(args.output,dict(inputs=pin(registry),script=pin(file),lane=lane,same_state_only=True,events=results)))


if __name__=='__main__':
    main()
