"""Rebuild delivered packets across the new early egress handoff interruption."""
import argparse
from pathlib import Path
import runpy
import sys

import numpy as np
import torch
from instinctlab.control.escape_recovery import MOVER_PASSAGE_REPROOF_25
from instinctlab.control.measured_curve_deflection import _fit_return, fresh_motion_history
from instinctlab.control.pipeline import ControlCoreConfig
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked, git

folder=Path(__file__).resolve().parent
sys.path.insert(0,str(folder.parent))
sys.path.insert(0,str(folder.parent/'handoff'))
from probe_two_command_path import packet_at
from replay_egress import issued_proof
from replay_opposed_planes import reprove_rotation


def probe(inputs, changed, support):
    assert not stop_paths(),stop_paths()
    assert git('rev-parse','HEAD')==inputs['v60_commit'] and not git('branch','--show-current')
    h=runpy.run_path(str(checked(support)))['execute'].__globals__
    cfg=ControlCoreConfig(ics=h['ics'])
    ctrl=h['ClosedControl'](cfg,h['ray_tensor'],MOVER_PASSAGE_REPROOF_25)
    rows={r['step']:r for r in changed['rows']}
    evidence=[]
    with torch.inference_mode():
        for t in (212,213,214):
            scan,state,offsets,coast=packet_at(inputs,changed,h,t)
            prior=state[:,48:51]
            request=state[:,45:48]
            proof=None if bool(fresh_motion_history(state.reshape(1,4,13))[0]) else reprove_rotation(scan,state,h['ray_tensor'])
            fit=_fit_return(scan,state,h['ray_tensor'],offsets,coast,entry=False,coherent=False,align_gravity=proof is not None)
            event=dict(step=t,selected=rows[t].get('pick'),high_priority_retreat=rows[t]['flags']['retreating'],
                       measured_mover_gap_m=rows[t]['physical_mover'],mover_fit=fit is not None,plane_proof=proof)
            evidence.append(event)
            if fit is None or np.linalg.norm(fit.acceleration,axis=2).max()>1.05:
                continue
            if proof is not None:
                fit.radius+=3.*proof[0]
            choice, details=h['reversed_menu'](ctrl.escape_option,scan,state,cfg,fit,coast,request,prior)
            _, issued, full, backup, valid=issued_proof(ctrl.escape_option,scan,state,cfg,(fit,coast),prior,.60)
            requested_projection=float((issued*request).sum()/request.norm())
            actor=h['actor']({'lidar_grid':scan,'state_vec':state}).mean_action
            actor_issued=ctrl.escape_option.issued(scan,state,actor,cfg)
            event.update(latest_actor_beta=float(actor_issued.metrics['beta'][0,0]),
                continuation=dict(passes=bool(valid[0]),planned_m=float(full[0]),backup_m=float(backup[0]),
                                  requested_projection_m_s=requested_projection),
                strongest_certified=details['true_latest_stage30'],qualified_ics=details['capped_ics'])
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    regs=folder/'CONFLICT-INPUTS.json'
    registration=load(regs)
    changed=load(checked(registration['egress_output']))
    inputs=load(checked(registration['crawl_inputs']))
    for key, ref in inputs.items():
        if key!='v60_commit':
            checked(ref)
    result=probe(inputs,changed,registration['handoff_harness'])
    assert not stop_paths(),stop_paths()
    print(save(args.output,dict(inputs=pin(regs),script=pin(__file__),events=result)))


if __name__=='__main__':
    main()
