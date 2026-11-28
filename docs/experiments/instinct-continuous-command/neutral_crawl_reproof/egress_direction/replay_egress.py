"""Check alternative certified directions at low-speed egress onset on CPU."""
import argparse
import gzip
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np
import torch
from instinctlab.control.escape_passage import observed_path_clearance
from instinctlab.control.escape_recovery import MOVER_PASSAGE_REPROOF_25
from instinctlab.control.measured_curve_deflection import fresh_motion_history
from instinctlab.control.observed_mover_preview import _clearance, _rank_continuations
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked, git

folder = Path(__file__).resolve().parent
sys.path.insert(0,str(folder.parent))
from replay_opposed_planes import reprove_rotation


def issued_proof(option,scan,state,config,fit_data,velocity,horizon):
    fit, coast = fit_data
    previous = state[0,48:51]
    actions = option.action_for_velocity(velocity,previous.expand_as(velocity),config.governor)
    outputs = [option.issued(scan.expand(len(batch),-1,-1,-1),state.expand(len(batch),-1),batch,config)
               for batch in actions.split(24)]
    issued = torch.cat([o.v_final_b for o in outputs])
    valid = torch.cat([(o.metrics['beta'][:,0]>=.90)&(o.metrics['emergency'][:,0]==0.) for o in outputs]).numpy()
    rays = option.detector._full_rays
    # Reuse the matched stop replay's static margin and lag/queue bracket.
    surfaces = []
    for hold in (.60,.10):
        floors = [observed_path_clearance(scan,rays,torch.tensor(coast[1:2]),batch[None],hold_s=hold)[0]
                  for batch in issued.split(24)]
        surfaces.append(torch.cat(floors).numpy()-.055)
    values = issued.numpy()
    prior = previous.numpy()
    planned = np.minimum(surfaces[0],_clearance(fit,coast,prior,values,.60,horizon))
    backup = np.minimum(surfaces[1],_clearance(fit,coast,prior,values,.10,.20))
    valid &= (planned>=.39)&(backup>=.39)
    return actions,issued,planned,backup,valid


def choose_onset(option,scan,state,config,fit_data,heading,old_action,arm):
    up = -state[0,42:45]
    up /= up.norm()
    forward = heading[0]-(heading[0]@up)*up
    forward /= forward.norm()
    side = torch.linalg.cross(up,forward)
    angles = state.new_tensor(np.arange(72)*np.pi/36.)
    directions = (angles.cos()[:,None]*forward+angles.sin()[:,None]*side)
    old_issued = option.issued(scan,state,old_action,config).v_final_b
    candidate_speeds = (.15,.225,.2999,.375,.4499) if arm=='forecast' else (.15,.225,.2999)
    neutral = state.new_zeros((1,3))
    candidates = torch.cat((old_issued,neutral,*(speed*directions for speed in candidate_speeds)))
    actions,issued,full,backup,qualified = issued_proof(option,scan,state,config,fit_data,candidates,.60)
    earlier = state[0,48:51]
    jumps = (issued-earlier).norm(dim=1).numpy()
    speeds = issued.norm(dim=1).numpy()
    fit,coast = fit_data
    ranks = _rank_continuations(fit,coast,earlier.numpy(),issued.numpy())
    qualified &= jumps<=jumps[0]+1.e-5
    qualified[:2] = False
    if arm=='forecast':
        eligible = qualified&(ranks>=ranks[0]+.01)
    else:
        eligible = qualified&(speeds<=.30)&(jumps<=.30)&(ranks>=ranks[1]+.01)
    event = dict(certified=int(qualified.sum()),eligible=int(eligible.sum()),native_rank=float(ranks[0]),
                 neutral_rank=float(ranks[1]),native_delta=float(jumps[0]))
    if not eligible.any():
        return None,event
    idx = int(np.where(eligible,ranks,-np.inf).argmax())
    event.update(planned=float(full[idx]),backup=float(backup[idx]),rank=float(ranks[idx]),
                 speed=float(speeds[idx]),delta=float(jumps[idx]),issued_b=issued[idx].tolist())
    return (actions[idx],issued[idx:idx+1]),event


def run(inputs,lane,lag,arm,mover):
    assert not stop_paths(),stop_paths()
    for key,ref in inputs.items():
        if key!='v60_commit':
            checked(ref)
    assert git('rev-parse','HEAD')==inputs['v60_commit'] and not git('branch','--show-current')
    archive = Path(inputs['trace']['path']).parent
    with gzip.open(archive/'candidate-layout_routes.jsonl.gz','rt') as stream:
        initial = json.loads(next(stream))
        rows = [json.loads(next(stream)) for _ in range(254)]
    harness = runpy.run_path(inputs['replay_harness']['path'])
    execute,legacy = harness['execute'],harness['StopTrial']
    context = execute.__globals__
    assert legacy.apply.__globals__ is context and initial==context['initial']
    assert np.array_equal(context['rays'],np.fromfile(inputs['rays']['path'],dtype='<f4').reshape(-1,3))
    context.update(root=archive,rows=rows,GRAVITY_PREVIEW_RENEWAL_25=MOVER_PASSAGE_REPROOF_25)
    original_fit = context['_fit_return']
    rays = context['ray_tensor']

    class DirectionTrial(legacy):
        def __init__(self,option,mode,hold_horizon,crawl_speed):
            super().__init__(option,mode,hold_horizon,crawl_speed)
            self.egress_speed = None
            self.taper = 'diagnostic'

        def apply(self,policy,action,config):
            assert not stop_paths(),stop_paths()
            scan,state = policy['lidar_grid'],policy['state_vec']
            witness = None
            fit_data = None
            prior_phase = self.phase

            def supported(history):
                nonlocal witness
                original = fresh_motion_history(history)
                if bool(original[0]):
                    return original
                no_gyro = history.clone()
                no_gyro[:,:,:3] = 0.
                if bool(fresh_motion_history(no_gyro)[0]):
                    witness = reprove_rotation(scan,state,rays)
                return torch.ones_like(original) if witness is not None else original

            def observed(local_scan,packet,calibration,offsets,coast,**kwargs):
                nonlocal fit_data
                fit = original_fit(local_scan,packet,calibration,offsets,coast,
                                   align_gravity=witness is not None,**kwargs)
                if fit is not None:
                    if witness is not None:
                        fit.radius += 3.*witness[0]
                    fit_data = fit,coast
                return fit

            with patch.dict(context,fresh_motion_history=supported,_fit_return=observed):
                selected = super().apply(policy,action,config)
            if witness is not None:
                self.last_info['gyro_reproof'] = dict(witness=witness,reason=None)
            if (fit_data is None or self.phase!='egress' or not self.last_info.get('supported')
                    or float(state[0,45:48].norm())>.001):
                return selected
            if prior_phase=='hold' and self.last_info.get('pick')==1:
                chosen,event = choose_onset(self.option,scan,state,config,fit_data,self.last_motion,selected,arm)
                self.last_info['direction_onset'] = event
                if chosen is not None:
                    selected[0],motion = chosen
                    self.egress_speed = float(motion.norm())
                    self.last_motion = motion/self.egress_speed
                    self.last_info['pick'] = 'steer'
            elif self.egress_speed is not None:
                issued = issued_proof(self.option,scan,state,config,fit_data,self.egress_speed*self.last_motion,.20)
                actions,motion,full,backup,valid = issued
                if valid[0] and float((motion-state[:,48:51]).norm())<=.30001:
                    selected[0] = actions[0]
                    self.last_info.update(pick='renew_steer',direction_renewal=dict(short=float(full[0]),backup=float(backup[0])))
                else:
                    self.last_info['direction_renewal_failed'] = True
            return selected

    with torch.inference_mode(), patch.dict(context,StopTrial=DirectionTrial):
        return execute(lane,{19:176,22:168}[lane],lag,'bridge30',.20,mover,'recorded',.065)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane',required=True,type=int,choices=(19,22))
    parser.add_argument('--lag',required=True,type=float,choices=(.25,.35,.45))
    parser.add_argument('--arm',required=True,choices=('forecast','bounded'))
    parser.add_argument('--mover',default='native',choices=('native','mirrored','earlier'))
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    assert not args.output.exists()
    registration=folder/'INPUTS.json'
    release=load(registration)
    for ref in release.values():
        checked(ref)
    common=load(checked(release['crawl_inputs']))
    inputs=dict(common,replay_harness=release['handoff_harness'])
    result=run(inputs,args.lane,args.lag,args.arm,args.mover)
    assert not stop_paths(),stop_paths()
    print(save(args.output,dict(inputs=release['crawl_inputs'],replay=release['crawl_replay'],
        direction_inputs=pin(registration),direction_replay=pin(__file__),direction_arm=args.arm,plane_reproof=True,**result)))


if __name__=='__main__':
    main()
