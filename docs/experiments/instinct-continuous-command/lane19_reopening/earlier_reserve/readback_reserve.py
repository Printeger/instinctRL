"""Run exact-issue and reproof checks for an earlier reserve in the frozen CPU loop."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
from unittest.mock import patch

import numpy as np
import torch

from instinctlab.control.escape_passage import observed_path_clearance
from instinctlab.control.measured_curve_deflection import fresh_motion_history
from instinctlab.control.registered_mover_entry import _piecewise_clearance, _registered_fit
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked


def audit(replay, settings, plan, rotation, lane_cfg):
    parent = replay['Trial']
    vetoes = replay['PROTECTED'] + ('opening_steered',)
    first = plan['head']
    target = np.asarray(plan['best']['group_outputs'][0]['issues'])
    assert np.array_equal(target,np.asarray(plan['best']['group_outputs'][1]['issues']))

    class Audit(parent):
        def __init__(self, option, observers):
            super().__init__(option,observers)
            self.step=lane_cfg['warm']
            self.disabled=False

        def apply(self, policy, action, config):
            stock=self.original(policy,action,config)
            self.stock_action=stock[0].tolist()
            step=self.step
            self.step+=1
            if self.disabled or step<lane_cfg['start'] or step>=settings['return_to_ccg']:
                return stock
            current=step-lane_cfg['start']
            option=self.option
            scan,state=policy['lidar_grid'],policy['state_vec']
            register=_registered_fit(scan,state,option.detector._full_rays,entry=False)
            protected=[k for k in vetoes if bool(getattr(option,k)[0])]
            if bool(option.observed_turn_brake_floor[0] or option.zero_crossing_commit[0]):
                protected.append('mover_braking_or_zero_commit')
            previous_issue=option.issued(scan,state,stock,config)
            if bool(previous_issue.metrics['emergency'][0,0]):
                protected.append('stock_emergency')
            body=state.new_tensor(target[current]@rotation.T).reshape(1,3)
            raw=state.new_tensor([first['raw_action']]) if current==0 else option.action_for_velocity(
                body,state[:,48:51],config.governor)
            checked_issue=option.issued(scan,state,raw,config)
            predicted=checked_issue.v_final_b
            beta=float(checked_issue.metrics['beta'][0,0])
            error=float((predicted-body).norm())
            self.event=dict(step=step,requested_body=body[0].tolist(),predicted_body=predicted[0].tolist(),
                            predicted_error_m_s=error,predicted_beta=beta,priority=protected)

            def reject(reason):
                self.event['reason']=reason
                self.disabled=True
                return stock

            if protected:
                return reject('preserved_stock_protection')
            if error>settings['issue_tolerance_m_s'] or beta<.9 or bool(checked_issue.metrics['emergency'][0,0]):
                return reject('ics_or_governor_mismatch')
            if register is None or not bool(fresh_motion_history(state.reshape(1,4,13))[0]):
                return reject('missing_fresh_registration')
            fit,basis,spacing=register
            coast=torch.stack([p[1][0] for p in self.observers]).numpy()
            plane=(coast@basis[:2])@basis.T
            next_target=state.new_tensor(target[current+1]@rotation.T).reshape(1,3)
            queue=state[0,48:51]
            following_action=option.action_for_velocity(next_target,queue.reshape(1,3),config.governor)
            next_issue=option.issued(scan,state,following_action,config)
            if (float((next_issue.v_final_b-next_target).norm())>settings['issue_tolerance_m_s']
                    or float(next_issue.metrics['beta'][0,0])<.9 or bool(next_issue.metrics['emergency'][0,0])):
                return reject('following_plan_ics_mismatch')
            planned=float(_piecewise_clearance(fit,plane,queue.numpy(),predicted.numpy(),
                                               next_issue.v_final_b.numpy(),.60,1.,spacing)[0])
            backup=float(_piecewise_clearance(fit,plane,queue.numpy(),predicted.numpy(),
                                               np.zeros((1,3)),.10,.90,spacing)[0])
            static=float(observed_path_clearance(scan,option.detector._full_rays,state.new_tensor(coast[1:2]),
                       next_issue.v_final_b[None],hold_s=.60)[0,0])-.055-.1*float((predicted-next_issue.v_final_b).norm())
            self.event.update(planned_m=planned,backup_m=backup,static_m=static)
            if min(planned,backup,static)<settings['certificate_floor_m']:
                return reject('no_joint_plan_and_backup_certificate')
            stock[0]=raw[0]
            option.supersede_mover_edge_choice([0])
            option.measured_mover_active[0]=True
            self.event.update(decision='issued',body=predicted[0].tolist())
            return stock

    return Audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    assert not stop_paths() and shutil.disk_usage(args.output.parent).free>=30*2**30+10*2**20
    file=Path(__file__).resolve()
    reg=file.with_name('READBACK-INPUTS.json')
    refs=load(reg)
    plan_result=load(checked(refs['physical_receipt']))
    assert plan_result['registry']==refs['reserve_inputs'] and plan_result['script']==refs['reserve_script']
    asset=runpy.run_path(str(checked(refs['reserve_script'])))
    reserve=load(checked(refs['reserve_inputs']))
    settings=refs['protocol']
    data=asset['inputs'](reserve)
    settings=settings|{'return_to_ccg':reserve['protocol']['return_to_ccg']}
    source=runpy.run_path(str(checked(reserve['closed_script'])))['replay'].__globals__
    cfg=load(checked(refs['original_replay_inputs']))['protocol'] | {
         'start':plan_result['anchor'], 'actuator_lag_s':settings['lag_s']}
    plan=plan_result['results']['modes'][settings['mode']]['best']
    assert plan['best']['clearance_m']>=.15
    h, rows, lane = data['scene'],data['archive'],cfg['lane']
    with torch.inference_mode():
        frames={k:h['reconstruct'](data['initial'],rows[k],rows[k-1],lane,h['rays'],data['mesh'])[0]
                for k in range(cfg['warm']-3,cfg['start']+1)}
        attitude=np.array([r['attitude_wxyz'][lane] for r in rows])
        rotation=np.array([[h['body_to_world'](q,v) for v in np.eye(3)] for q in attitude])
        requests=np.array([r['raw_event'][lane]['velocity_w_m_s'] for r in rows])
        source_metrics=load(checked(refs['native_metrics']))
        radius=source_metrics['models']['candidate']['layout_routes'][lane]['physics_tick_clearance']['maximum_measured_link_envelope_m']
        results={}
        for key in settings['cases']:
            case=dict(cfg['cases'][key],attitude='fixed')
            stock=source['replay'](h,rows,frames,data['raw'],data['mesh'],data['solids'],attitude,rotation,
                                  requests,'stock',case,cfg,args.output,radius)
            wrapper=audit(source,settings,plan,rotation[cfg['start']],cfg)
            with patch.dict(source,Trial=wrapper):
                candidate=source['replay'](h,rows,frames,data['raw'],data['mesh'],data['solids'],attitude,rotation,
                                          requests,'renewal',case,cfg,args.output,radius)
            rejection=next((r for r in candidate['records'] if 'reason' in r['proof']),None)
            first_decisions=[dict(step=r['step'],proof=r['proof']) for r in candidate['records'] if r['proof']]
            results[key]=dict(stock=stock,candidate=candidate,first_rejection=rejection)
            print(json.dumps(dict(case=key,decisions=first_decisions,stock_metrics=stock['metrics'],
                                  candidate_metrics=candidate['metrics'])),flush=True)
    assert not stop_paths(),stop_paths()
    print(json.dumps(dict(raw=save(args.output,dict(registry=pin(reg),script=pin(file),
                  physical_plan=refs['physical_receipt'],cases=results)))),flush=True)


if __name__=='__main__':
    main()
