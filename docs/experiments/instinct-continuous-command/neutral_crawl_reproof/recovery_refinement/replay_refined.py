"""Keep the original recovery certificates while refining the existing finite menu."""
import argparse
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load,save
from tools.r6_direct.source import checked

folder=Path(__file__).resolve().parent
sys.path.insert(0,str(folder.parent/'egress_direction'))
import replay_certified_handoff as replay
from replay_egress import issued_proof


def neighborhoods(state, requested, earlier, details):
    up=-state[0,42:45]
    up /= up.norm()
    coarse = [state.new_tensor(details[name]['body']) for name in (
        'strongest_stage30','strongest_latest_cap_long','strongest_long') if details[name] is not None]
    seeds=torch.stack([requested[0],earlier[0],*coarse])
    horizontal=seeds-(seeds@up)[:,None]*up
    length=horizontal.norm(dim=1)
    present=length>=.02
    length,horizontal=length[present],horizontal[present]
    directions=horizontal/length[:,None]
    side=torch.linalg.cross(up.expand_as(directions),directions)
    angles=state.new_tensor([-4.,-2.,0.,2.,4.])*np.pi/180.
    headings=angles.cos()[None,:,None]*directions[:,None]+angles.sin()[None,:,None]*side[:,None]
    speeds=(length[:,None]+state.new_tensor([-.025,0.,.025])).clamp(min=.025,max=.2999)
    around=(headings[:,:,None]*speeds[:,None,:,None]).reshape(-1,3)
    axis=requested[0]-(requested[0]@up)*up
    axis /= axis.norm()
    steps=state.new_tensor([.20,.40,.60,.80,1.])
    direct=horizontal[:,None]+steps[None,:,None]*(requested[0]-horizontal)[:,None]
    changed_along=horizontal[:,None]+steps[None,:,None]*(requested.norm()-horizontal@axis)[:,None,None]*axis
    return torch.cat((around,direct.reshape(-1,3),changed_along.reshape(-1,3))), axis


def refined(option,scan,state,cfg,fit,coast,requested,earlier,coarse_menu):
    actions, details=coarse_menu(option,scan,state,cfg,fit,coast,requested,earlier)
    targets,axis=neighborhoods(state,requested,earlier,details)
    delta=targets-earlier
    capped=earlier+delta*(.30/delta.norm(dim=1,keepdim=True).clamp_min(.30))
    extra,issued,hold,backup,valid=issued_proof(option,scan,state,cfg,(fit,coast),capped,.60)
    along=(issued@axis).numpy()
    slew=(issued-earlier).norm(dim=1).numpy()
    target=float(requested.norm())
    valid &= ((issued.norm(dim=1)<=.29999).numpy()&(along<=target+1.e-6)&(slew<=.30001))
    incumbent=details['strongest_stage30']
    details['refinement']=dict(examined=len(extra),certified=int(valid.sum()),coarse_progress=None if incumbent is None else incumbent['progress'])
    if not valid.any():
        return actions, details
    chosen=int(np.where(valid,along,-np.inf).argmax())
    if incumbent is not None and along[chosen]<=incumbent['progress']+1.e-5:
        return actions, details
    action=extra[chosen:chosen+1]
    current=option.issued(scan,state,action,cfg)
    witness=dict(idx=len(actions),body=issued[chosen].tolist(),goal=capped[chosen].tolist(),progress=float(along[chosen]),
                 delta=float(slew[chosen]),beta=float(current.metrics['beta'][0,0]),emergency=float(current.metrics['emergency'][0,0]),
                 short=float(hold[chosen]),long=float(hold[chosen]),backup=float(backup[chosen]))
    details['strongest_stage30']=witness
    if along[chosen]>=.60*target:
        details['true_latest_stage30']=witness
    return torch.cat((actions,action)),details


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane',required=True,type=int,choices=(19,22))
    parser.add_argument('--lag',required=True,type=float,choices=(.25,.35,.45))
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    assert not args.output.exists()
    registry=folder/'INPUTS.json'
    plan=load(registry)
    for ref in plan.values():
        checked(ref)

    def harness(path):
        module=runpy.run_path(path)
        context=module['execute'].__globals__
        original=context['reversed_menu']

        def candidate(option,scan,state,cfg,fit,coast,requested,earlier):
            return refined(option,scan,state,cfg,fit,coast,requested,earlier,original)

        context['reversed_menu']=candidate
        return module

    with patch.object(replay,'runpy',SimpleNamespace(run_path=harness)):
        reference,result,notes=replay.run(Path(plan['handoff_inputs']['path']),args.lane,args.lag)
    assert not stop_paths(),stop_paths()
    print(save(args.output,dict(inputs=reference['crawl_inputs'],replay=reference['crawl_replay'],
        refined_inputs=pin(registry),refined_replay=pin(__file__),accepted_continuations=notes,plane_reproof=True,**result)))


if __name__=='__main__':
    main()
