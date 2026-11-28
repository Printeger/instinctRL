"""Reprove an earlier lane 19 reserve against current full-ICS available actions."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
from unittest.mock import patch

import numpy as np
import torch

from instinctlab.control.escape_passage import observed_path_clearance, rotate_scan_rays
from instinctlab.control.measured_curve_deflection import fresh_motion_history
from instinctlab.control.observed_mover_preview import _rank_continuations
from instinctlab.control.registered_mover_entry import _piecewise_clearance, _registered_fit
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked


def observed_stop(scan, coast, queue, first):
    level=scan[0,9]
    valid=(scan[0,10]>.5)&(scan[0,11]>.55)&level.isfinite()
    if not bool(valid.any()):
        return np.full(len(first),-np.inf)
    drift=.45*float(np.linalg.norm(coast,axis=1).max())+.10*float(queue.norm())
    return float(level[valid].min())-drift-.10*first.norm(dim=1).numpy()-.055


def verified_pairs(source,option,scan,state,config,registration,coast,preferred,next_want,heading,forced):
    issue=source['issued_bank']
    fit,basis,spacing=registration
    plane=(coast@basis[:2])@basis.T
    tested=state[0,48:51]
    neutral=float(state[0,45:48].norm()) < .001
    angles=np.arange(36)*np.pi/18
    circle=np.stack((np.cos(angles),np.sin(angles)),axis=1)@basis.T
    zero=state.new_zeros(3)
    center=state.new_tensor(np.concatenate([s*circle for s in (.15,.225,.2999,.375,.4499)]))
    future=torch.stack((next_want,preferred,heading,zero))
    if not neutral:
        future=torch.cat((future,center))
    else:
        future=zero[None]
    _,follows,found=issue(option,scan,state,config,future)
    if not neutral:
        delta=follows-tested
        first_targets=tested+delta*(.30/delta.norm(dim=1,keepdim=True).clamp_min(.30))
        idx=torch.arange(len(future))
        pairs=[(preferred.expand(len(future),-1),idx),(first_targets,idx)]
    else:
        pairs=[(zero[None],torch.zeros(1,dtype=torch.long))]
    target=torch.cat([p[0] for p in pairs])
    index=torch.cat([p[1] for p in pairs])
    if forced is not None:
        check=option.issued(scan,state,forced,config)
        proven=check.v_final_b
        target=proven.expand(len(future),-1)
        index=torch.arange(len(future))
        first=target
        actions=forced.expand(len(future),-1)
        ok=np.full(len(future),bool(check.metrics['beta'][0,0]>=.9 and not check.metrics['emergency'][0,0]))
    else:
        actions,first,ok=issue(option,scan,state,config,target)
    f=follows[index]
    ok &= found[index]
    if not neutral:
        ok &= ((first-tested).norm(dim=1).numpy()<=.30001)
        ok &= ((f-first).norm(dim=1).numpy()<=.30001)| (f.norm(dim=1).numpy()<=.001)
        ok &= ((first@heading).numpy() >= .10*first.norm(dim=1).numpy()*float(heading.norm())-1.e-8)
        ok &= ((f@heading).numpy() >= .10*f.norm(dim=1).numpy()*float(heading.norm())-1.e-8)
    else:
        ok &= first.norm(dim=1).numpy()<=2.e-6
    picked=np.flatnonzero(ok)
    if len(picked)==0:
        return None,dict(reason='no_ics_pair',available=0)
    actions,first,f=actions[picked],first[picked],f[picked]
    planned=_piecewise_clearance(fit,plane,tested.numpy(),first.numpy(),f.numpy(),.60,1.,spacing)
    move_backup=_piecewise_clearance(fit,plane,tested.numpy(),first.numpy(),np.zeros_like(f.numpy()),.10,.90,spacing)
    wall=torch.cat([observed_path_clearance(scan,option.detector._full_rays,state.new_tensor(coast[1:2]),
                part[None],hold_s=.60)[0] for part in f.split(24)]).numpy()
    wall=wall-.055-.1*(first-f).norm(dim=1).numpy()
    static_backup=observed_stop(scan,coast,tested,first)
    floor=np.minimum.reduce((planned,move_backup,wall,static_backup))
    eligible=floor>=.39
    # Rank alone can be limited by the shared queued prefix. It serves
    # only as an absolute additional guard during the receding replay.
    rank=_rank_continuations(fit,coast,tested.numpy(),f.numpy())
    eligible &= rank>=.15
    detail=dict(available=len(first),qualified=int(eligible.sum()),max_floor_m=float(floor.max()))
    if not eligible.any():
        return None,detail|dict(reason='no_joint_plan_and_backup_certificate')
    cost=(first-preferred).norm(dim=1)+.40*(f-next_want).norm(dim=1)
    best=int(np.argmin(np.where(eligible,cost.numpy(),np.inf)))
    return (actions[best],first[best],f[best]),detail|dict(
        body=first[best].tolist(),following_b=f[best].tolist(),planned_m=float(planned[best]),
        mover_backup_m=float(move_backup[best]), static_backup_m=float(static_backup[best]),
        static_m=float(wall[best]),rank_m=float(rank[best]),
        planned_issue_difference_m_s=float((first[best]-preferred).norm()),
        preferred_continuation_difference_m_s=float((f[best]-next_want).norm()))


def adaptive_trial(source, cfg, profile, reference, rotation, mode):
    parent=source['Trial']
    vetoes=source['PROTECTED']+('opening_steered',)
    first=reference['head']
    schedule=np.array(reference['schedule'])@rotation.T

    class Trial(parent):
        def __init__(self,option,observers):
            super().__init__(option,observers)
            self.step=cfg['warm']
            self.desired=None
            self.started=False
            self.finished=False
            self.backup_from=None
            self.stopping=False

        def apply(self,policy,action,config):
            stock=self.original(policy,action,config)
            self.stock_action=stock[0].tolist()
            step=self.step
            self.step+=1
            if self.finished or step<cfg['start'] or step>=profile['max_apply_step']:
                return stock
            option=self.option
            scan,state=policy['lidar_grid'],policy['state_vec']
            blockers=[name for name in vetoes if bool(getattr(option,name)[0])]
            if bool(option.observed_turn_brake_floor[0] or option.zero_crossing_commit[0]):
                blockers.append('mover_braking_or_zero_commit')
            output=option.issued(scan,state,stock,config)
            if bool(output.metrics['emergency'][0,0]):
                blockers.append('stock_emergency')
            if blockers:
                self.event=dict(reason='preserved_stock_protection',flags=blockers)
                self.finished=True
                return stock
            k=step-cfg['start']
            coast=torch.stack([c[1][0] for c in self.observers]).numpy()
            zero=state.new_zeros((1,3))
            age=float(state[0,51])
            if k==0:
                heading=state.new_tensor(first['first_b'])
            else:
                hist=state.reshape(4,13)
                gyro=-.05*(hist[-2,:3]+hist[-1,:3])
                heading=rotate_scan_rays(self.desired[None],gyro)[0] if self.desired is not None else state.new_tensor(first['first_b'])
            if float(heading.norm())<.01:
                heading=state[0,45:48].clone()
            preferred=state.new_tensor(schedule[k]) if mode=='guided' else heading
            next_want=state.new_tensor(schedule[k+1]) if mode=='guided' else heading
            if k==0:
                preferred=state.new_tensor(first['first_b'])
                next_want=state.new_tensor(first['planned_next_b'])
            raw=state.new_tensor([first['raw_action']]) if k==0 else None
            prior_probe=option.issued(scan,state,option.action_for_velocity(
                next_want[None],state[:,48:51],config.governor),config)
            probe=dict(step=step,target_next_beta=float(prior_probe.metrics['beta'][0,0]),
                       target_next_issue_error_m_s=float((prior_probe.v_final_b[0]-next_want).norm()))
            fit=_registered_fit(scan,state,option.detector._full_rays,entry=False)
            if self.stopping:
                selected,event=None,dict(reason='finite_backup')
            elif fit is None or not bool(fresh_motion_history(state.reshape(1,4,13))[0]):
                selected,event=None,dict(reason='missing_fresh_registration')
            else:
                selected,event=verified_pairs(source,option,scan,state,config,fit,coast,preferred,next_want,heading,raw)
            self.event=probe|event
            if selected is not None:
                chosen,v,follow=selected
                stock[0]=chosen
                self.started=True
                self.desired=follow
                self.backup_from=step
                self.event.update(decision='reproved')
            else:
                neutral=option.action_for_velocity(zero,state[:,48:51],config.governor)
                read=option.issued(scan,state,neutral,config)
                still_valid=self.backup_from is not None and step-self.backup_from < 8
                stop_floor=float(observed_stop(scan,coast,state[0,48:51],zero)[0])
                if (not self.started or not still_valid or not 0. <= age <= .04 or stop_floor<.39
                        or float(read.v_final_b.norm())>2.e-6 or bool(read.metrics['emergency'][0,0])):
                    self.event.update(backup_release=True,current_static_backup_m=stop_floor if np.isfinite(stop_floor) else None)
                    self.finished=True
                    return stock
                self.stopping=True
                stock[0]=neutral[0]
                self.event.update(decision='backup_zero',current_static_backup_m=stop_floor,
                                  verified_zero_b=read.v_final_b[0].tolist(),backup_age_steps=step-self.backup_from)
            option.supersede_mover_edge_choice([0])
            option.measured_mover_active[0]=True
            return stock

    return Trial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    assert not stop_paths() and shutil.disk_usage(args.output.parent).free>=30*2**30+10*2**20
    me=Path(__file__).resolve()
    reg=me.with_name('INPUTS.json')
    inputs=load(reg)
    prior=load(checked(inputs['static_continuation']))
    read=load(checked(inputs['prior_readback_inputs']))
    plan=load(checked(read['physical_receipt']))
    reserve=load(checked(read['reserve_inputs']))
    asset=runpy.run_path(str(checked(read['reserve_script'])))
    data=asset['inputs'](reserve)
    replay=runpy.run_path(str(checked(reserve['closed_script'])))['replay'].__globals__
    params=inputs['protocol']
    cfg=load(checked(read['original_replay_inputs']))['protocol']|{'start':plan['anchor'],'actuator_lag_s':params['lag_s']}
    first=plan['results']['modes']['strict']['best']['head']
    schedule=prior['open_loop']['group_outputs'][0]['issues']
    h,rows,lane=data['scene'],data['archive'],cfg['lane']
    native=load(checked(read['native_metrics']))
    radius=native['models']['candidate']['layout_routes'][lane]['physics_tick_clearance']['maximum_measured_link_envelope_m']
    with torch.inference_mode():
        frames={k:h['reconstruct'](data['initial'],rows[k],rows[k-1],lane,h['rays'],data['mesh'])[0]
              for k in range(cfg['warm']-3,cfg['start']+1)}
        orientation=np.array([r['attitude_wxyz'][lane] for r in rows])
        rotation=np.array([[h['body_to_world'](q,v) for v in np.eye(3)] for q in orientation])
        raw=np.array([r['raw_event'][lane]['velocity_w_m_s'] for r in rows])
        outcomes={}
        for kind in params['arms']:
            trial=adaptive_trial(replay,cfg,params,dict(head=first,schedule=schedule),rotation[cfg['start']],kind)
            cases={}
            with patch.dict(replay,Trial=trial):
                for key in params['cases']:
                    case=dict(cfg['cases'][key],attitude='fixed')
                    arm=replay['replay'](h,rows,frames,data['raw'],data['mesh'],data['solids'],orientation,rotation,
                                      raw,'renewal',case,cfg,args.output,radius)
                    for item in arm['records']:
                        if item['proof'].get('decision')=='backup_zero':
                            assert item['issued_w']==[0.,0.,0.] and not item['emergency']
                    cases[key]=arm
                    print(json.dumps(dict(arm=kind,case=key,decisions=[(r['step'],r['proof']) for r in arm['records'] if r['proof']],
                                          metrics=arm['metrics'])),flush=True)
            outcomes[kind]=cases
            save(args.output.with_name(args.output.stem+'-'+kind+'.json'),
                dict(registry=pin(reg),script=pin(me),arm=kind,cases=cases))
    assert not stop_paths(),stop_paths()
    print(json.dumps(dict(raw=save(args.output,dict(registry=pin(reg),script=pin(me),outcomes=outcomes)))),flush=True)


if __name__=='__main__':
    main()
