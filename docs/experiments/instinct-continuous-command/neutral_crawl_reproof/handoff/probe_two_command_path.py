"""Screen two-command dynamic certificates at the recorded t203 decision."""
import argparse
import gzip
import json
from pathlib import Path
import runpy

import numpy as np
import torch
from instinctlab.control.escape_passage import advance_command_response
from instinctlab.control.escape_recovery import MOVER_PASSAGE_REPROOF_25
from instinctlab.control.measured_curve_deflection import _fit_return, fresh_motion_history
from instinctlab.control.observed_mover_preview import _own_path, _response
from instinctlab.control.pipeline import ControlCoreConfig
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked, git


def packet_at(inputs, replay, h, step=203):
    lane = 22
    with gzip.open(checked(inputs['trace']),'rt') as stream:
        initial = json.loads(next(stream))
        native = [json.loads(next(stream)) for _ in range(step+1)]
    assert h['initial']==initial and native[-1]['step']==step
    alternate = {r['step']:r for r in replay['rows']}
    start = min(alternate)
    origins, triangles = h['origins'],h['triangles']
    origin = origins[lane].numpy()
    local = ((triangles.max(axis=1)>=origin-45.)&(triangles.min(axis=1)<=origin+45.)).all(axis=1)
    mesh = triangles[local]
    dense = np.memmap(checked(inputs['physical']),dtype='<f4',mode='r',shape=(1200,21,32,7))
    model = h['observer']()
    for t, row in enumerate(native):
        if t<=start:
            history = np.asarray(row['actor_state'][lane],dtype=np.float32)
            if t==start:
                scans = [h['reconstruct'](initial,native[i],native[i-1],lane,h['rays'],mesh)[0]
                         for i in range(t-3,t+1)]
        else:
            attitude = np.asarray([h['body_to_world'](row['attitude_wxyz'][lane],axis) for axis in np.eye(3)])
            updated = np.asarray(row['actor_state'][lane][-1],dtype=np.float32).copy()
            prior = alternate[t-1]
            updated[6:9] = attitude@np.asarray(alternate[t]['raw'])
            updated[9:12] = attitude@np.asarray(prior['issued'])
            history = np.concatenate((history[1:],updated[None]))
            scans = scans[1:]+[h['render'](np.asarray(prior['position']),attitude,dense[t,0,lane,3:6],mesh)]
        state = torch.tensor(history).reshape(1,52)
        model = [advance_command_response(*old,state,lag=lag) for old,lag in zip(model,h['lags'],strict=True)]
    scan = torch.tensor(np.concatenate(scans)).reshape(1,12,59,360)
    offsets = torch.stack([row[0][0] for row in model]).numpy()
    coast = torch.stack([row[1][0] for row in model]).numpy()
    return scan,state,offsets,coast


def menu(scan,state,option,config):
    request = state[0,45:48]
    up = -state[0,42:45]
    up /= up.norm()
    direction = request-(request@up)*up
    direction /= direction.norm()
    across = torch.linalg.cross(up,direction)
    angles = state.new_tensor(np.arange(72)*np.pi/36.)
    headings = angles.cos()[:,None]*direction+angles.sin()[:,None]*across
    speeds = (.075,.15,float(request.norm()),.2999,.4499)
    choices = torch.cat((state.new_zeros((1,3)),request[None],*(speed*headings for speed in speeds)))
    prior = state[0,48:51]
    delta = choices-prior
    capped = prior+delta*(.30/delta.norm(dim=1,keepdim=True).clamp_min(.30))
    first = []
    for batch in capped.split(24):
        raw = option.action_for_velocity(batch,prior.expand_as(batch),config.governor)
        issued = option.issued(scan.expand(len(batch),-1,-1,-1),state.expand(len(batch),-1),raw,config)
        val = issued.v_final_b
        valid = ((val-prior).norm(dim=1)<=.30001)&(issued.metrics['beta'][:,0]>=.90)&(issued.metrics['emergency'][:,0]==0.)
        first.append(val[valid])
    return torch.cat(first).numpy(),choices.numpy(),direction.numpy(),prior.numpy()


def best_plan(first,second,fit,coast,queued,axis,lags):
    # The recorded response needs a first projection >= -.22599 m/s even
    # under ideal subsequent positive commands to reach t211. Use the rounded
    # looser necessary threshold, so it cannot exclude a qualifying plan.
    time = np.arange(121)*.005
    pairs = [(a,b) for a in first if a@axis>=-.2261 for b in second if np.linalg.norm(a-b)<=.30001]
    floor = fit.radius+.025+.05*time+.5*1.05*time**2
    margin = np.full(len(pairs),np.inf)
    a,b = (np.array(part) for part in zip(*pairs,strict=True))
    for lag,drift,v in zip(lags,coast,fit.velocity,strict=True):
        mover = fit.center+time[:,None]*v
        for delay in (0.,.05,.10):
            own = _own_path(time,lag,delay,drift,queued,a,.10)
            second_time = _response(np.maximum(0.,time-delay-.10),lag)
            own += second_time[None,:,None]*b[:,None]
            margin = np.minimum(margin,(np.linalg.norm(own-mover,axis=-1)-floor).min(axis=1))
    peak = int(margin.argmax())
    return dict(input_pair_count=len(pairs),best_dynamic_floor_m=float(margin[peak]),
                first_along_m_s=float(a[peak]@axis),first_b=a[peak].tolist(),second_b=b[peak].tolist(),
                dynamic_pass_count=int((margin>=.39).sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    scope=Path(__file__).resolve().parent
    registration=scope/'INPUTS.json'
    release=load(registration)
    inputs=load(checked(release['inputs']))
    raw=load(checked(release['opposed_control']))
    assert not stop_paths() and git('rev-parse','HEAD')==inputs['v60_commit']
    for key in ('replay_harness','control_bridge','raw_renderer','renderer_geometry','renderer_rays','renderer_trace'):
        checked(inputs[key])
    harness=runpy.run_path(inputs['replay_harness']['path'])['execute'].__globals__
    config=ControlCoreConfig(ics=harness['ics'])
    ctrl=harness['ClosedControl'](config,harness['ray_tensor'],MOVER_PASSAGE_REPROOF_25)
    with torch.inference_mode():
        scan,state,offsets,coast=packet_at(inputs,raw,harness)
        assert bool(fresh_motion_history(state.reshape(1,4,13))[0])
        fit=_fit_return(scan,state,harness['ray_tensor'],offsets,coast,entry=False,coherent=False)
        assert fit is not None and np.linalg.norm(fit.acceleration,axis=2).max()<=1.05
        first,second,axis,queued=menu(scan,state,ctrl.escape_option,config)
        bounds=best_plan(first,second,fit,coast,queued,axis,harness['lags'])
    assert not stop_paths(),stop_paths()
    print(save(args.output,dict(inputs=pin(registration),script=pin(__file__),step=203,
        assumption='first action passes current ICS and a rounded necessary projection; second may be any menu action respecting the slew bound; dynamic only; optimistic about future execution',**bounds)))


if __name__=='__main__':
    main()
