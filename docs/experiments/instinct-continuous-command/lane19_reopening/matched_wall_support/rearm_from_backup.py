"""Test rearming during an unexpired neutral backup using current wall evidence."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
from unittest.mock import patch

import numpy as np
import torch

from instinctlab.control.measured_curve_deflection import fresh_motion_history
from instinctlab.control.registered_mover_entry import _registered_fit
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked


def trial_factory(parent,registration):
    class Rearm(parent):
        def apply(self,policy,action,config):
            trying=False
            if self.stopping and not self.finished and self.backup_from is not None and self.step-self.backup_from<8:
                scan,state=policy['lidar_grid'],policy['state_vec']
                if bool(fresh_motion_history(state.reshape(1,4,13))[0]):
                    found,detail=registration(scan,state,self.option.detector._full_rays)
                    if found is not None:
                        trying=True
                        self.stopping=False
                        self.event['rearm_at']=self.step
            result=super().apply(policy,action,config)
            if trying:
                self.event.update(rearm_probe=True,registration_source=detail['reason'])
            return result

    return Rearm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    assert not stop_paths() and shutil.disk_usage(args.output.parent).free>=30*2**30+10*2**20
    script=Path(__file__).resolve()
    registry=script.with_name('REARM-INPUTS.json')
    refs=load(registry)
    original=load(checked(refs['previous_reproof']))
    previous=load(checked(original['registry']))
    support=load(checked(refs['fixed_support']))
    settings=load(checked(support['registry']))
    assert all(not row['matched_support']['mixed_surface_control']['accepted'] for row in support['scores']
               if row['step']==1063 and row['mode']=='guided')
    wall=runpy.run_path(str(checked(support['script'])))
    read=load(checked(previous['prior_readback_inputs']))
    reserve=load(checked(read['reserve_inputs']))
    plan=load(checked(read['physical_receipt']))
    preceding=load(checked(previous['static_continuation']))
    data=runpy.run_path(str(checked(read['reserve_script'])))['inputs'](reserve)
    replay=runpy.run_path(str(checked(reserve['closed_script'])))['replay'].__globals__
    new=runpy.run_path(str(checked(original['script'])))['adaptive_trial'].__globals__
    cfg=load(checked(read['original_replay_inputs']))['protocol']|{
        'start':plan['anchor'],'actuator_lag_s':previous['protocol']['lag_s']}
    head=plan['results']['modes']['strict']['best']['head']
    schedule=preceding['open_loop']['group_outputs'][0]['issues']
    h, rows,lane=data['scene'],data['archive'],cfg['lane']
    native=load(checked(read['native_metrics']))
    radius=native['models']['candidate']['layout_routes'][lane]['physics_tick_clearance']['maximum_measured_link_envelope_m']

    def strict(scan,state,rays):
        return _registered_fit(scan,state,rays,entry=False),dict(reason='unchanged_v60')

    def matched(scan,state,rays):
        return wall['fallback_fit'](scan,state,rays,settings['options'])

    with torch.inference_mode():
        anchor=cfg['start']
        frames={k:h['reconstruct'](data['initial'],rows[k],rows[k-1],lane,h['rays'],data['mesh'])[0]
               for k in range(cfg['warm']-3,anchor+1)}
        attitude=np.array([r['attitude_wxyz'][lane] for r in rows])
        rotation=np.array([[h['body_to_world'](q,v) for v in np.eye(3)] for q in attitude])
        requests=np.array([r['raw_event'][lane]['velocity_w_m_s'] for r in rows])
        results={}
        for mode,register in (('strict_rearm',strict),('matched_rearm',matched)):
            cases={}

            def local_registration(scan,state,rays,*,entry):
                assert not entry
                return register(scan,state,rays)[0]

            with patch.dict(new,_registered_fit=local_registration):
                baseline=new['adaptive_trial'](replay,cfg,previous['protocol'],dict(head=head,schedule=schedule),
                                              rotation[anchor],'guided')
                trial=trial_factory(baseline,register)
                with patch.dict(replay,Trial=trial):
                    for item in previous['protocol']['cases']:
                        case=dict(cfg['cases'][item],attitude='fixed')
                        arm=replay['replay'](h,rows,frames,data['raw'],data['mesh'],data['solids'],attitude,rotation,
                                             requests,'renewal',case,cfg,args.output,radius)
                        prior=original['outcomes']['guided'][item]['records']
                        assert all(a['issued_w']==b['issued_w'] for a,b in zip(arm['records'],prior,strict=True)
                                   if a['step'] <= 1062)
                        picks=[dict(step=r['step'],proof=r['proof']) for r in arm['records'] if r['proof'].get('rearm_probe')]
                        cases[item]=arm
                        print(json.dumps(dict(mode=mode,case=item,probes=picks,metrics=arm['metrics'])),flush=True)
            results[mode]=cases
    assert not stop_paths(),stop_paths()
    print(json.dumps(dict(raw=save(args.output,dict(registry=pin(registry),script=pin(script),results=results)))),flush=True)


if __name__=='__main__':
    main()
