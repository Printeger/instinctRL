"""Test whether a currently proven aligned command can survive retreat retriggering."""
import argparse
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from instinctlab.control.escape_passage import advance_command_response
from instinctlab.control.measured_curve_deflection import _fit_return, fresh_motion_history
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked

import replay_egress as egress
from replay_opposed_planes import reprove_rotation
from replay_projected_slew import projected_proof


def run(input_path,lane,lag):
    recipe=load(input_path)
    for ref in recipe.values():
        checked(ref)
    reference=load(checked(recipe['direction_inputs']))
    inputs=dict(load(checked(reference['crawl_inputs'])),replay_harness=reference['handoff_harness'])
    notes=[]

    def harness(path):
        module=runpy.run_path(path)
        context=module['execute'].__globals__
        bridge=context['ClosedControl']
        lags=context['lags']

        class ClosedHandoff(bridge):
            def __init__(self,config,rays,version):
                super().__init__(config,rays,version)
                self.tick=126
                self.response=context['observer']()
                self.left_stop=False
                for row in context['rows'][:self.tick]:
                    state=torch.tensor(row['actor_state'][lane]).reshape(1,52)
                    self.advance(state)

            def advance(self,state):
                self.response=[advance_command_response(*old,state,lag=tau)
                               for old,tau in zip(self.response,lags,strict=True)]

            def step(self,policy,action,quaternion):
                assert not stop_paths(),stop_paths()
                option=self.escape_option
                original=option.apply_recovery
                state=policy['state_vec']
                self.advance(state)
                history=state.reshape(4,13)
                request=history[-1,6:9]
                prior=history[-1,9:12]
                request_norm=float(request.norm())
                if request_norm<=.001 and float(prior.norm())>=.20:
                    self.left_stop=True
                if not self.left_stop or request_norm<.06:
                    result=super().step(policy,action,quaternion)
                    self.tick+=1
                    return result

                def retain(observation, actor_action, config):
                    selected=original(observation,actor_action,config)
                    # A live measured protection, braking, or emergency keeps
                    # priority. Only re-prove a preceding already-issued
                    # command that still tracks the unchanged stick direction.
                    if (not bool(option.retreating[0])
                            or any(bool(getattr(option,name)[0]) for name in (
                                'braking','measured_mover_active','intercepted','intercept_continued',
                                'static_stop_verified','crossing_restart_verified'))
                            or float((history[-2,6:9]-request).norm())>.04
                            or float(prior@request)<.60*request_norm**2):
                        return selected
                    scan=observation['lidar_grid']
                    stock=option.issued(scan,state,selected,config)
                    if bool(stock.metrics['emergency'][0,0]):
                        return selected
                    alignment=None
                    if not bool(fresh_motion_history(history[None])[0]):
                        without_gyro=history[None].clone()
                        without_gyro[:,:,:3]=0.
                        if not bool(fresh_motion_history(without_gyro)[0]):
                            return selected
                        alignment=reprove_rotation(scan,state,self.rays)
                        if alignment is None:
                            return selected
                    offsets=torch.stack([v[0][0] for v in self.response]).numpy()
                    coast=torch.stack([v[1][0] for v in self.response]).numpy()
                    fit=_fit_return(scan,state,self.rays,offsets,coast,entry=False,coherent=False,
                                    align_gravity=alignment is not None)
                    if fit is None or np.linalg.norm(fit.acceleration,axis=2).max()>1.05:
                        return selected
                    if alignment is not None:
                        fit.radius+=3.*alignment[0]
                    actions,issued,full,backup,valid=egress.issued_proof(option,scan,state,config,(fit,coast),prior[None],.60)
                    projected=float(issued[0]@request)
                    if (not valid[0] or float((issued[0]-prior).norm())>.30001
                            or projected<.60*request_norm**2 or projected<float(stock.v_final_b[0]@request)+.001):
                        return selected
                    notes.append(dict(step=self.tick,planned_m=float(full[0]),backup_m=float(backup[0]),
                                      along_m_s=projected/request_norm,stock=stock.v_final_b[0].tolist()))
                    selected[0]=actions[0]
                    return selected

                with patch.object(option,'apply_recovery',retain):
                    result=super().step(policy,action,quaternion)
                self.tick+=1
                return result

        context['ClosedControl']=ClosedHandoff
        return module

    with (patch.object(egress,'runpy',SimpleNamespace(run_path=harness)),
          patch.object(egress,'issued_proof',projected_proof)):
        result=egress.run(inputs,lane,lag,'forecast','native')
    assert not stop_paths(),stop_paths()
    return reference,result,notes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane',required=True,type=int,choices=(19,22))
    parser.add_argument('--lag',required=True,type=float,choices=(.25,.35,.45))
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    assert not args.output.exists()
    source=Path(__file__).resolve()
    registry=source.with_name('CERTIFIED-HANDOFF-INPUTS.json')
    reference,result,notes=run(registry,args.lane,args.lag)
    print(save(args.output,dict(inputs=reference['crawl_inputs'],replay=reference['crawl_replay'],
        proof_inputs=pin(registry),handoff_replay=pin(source),accepted_continuations=notes,plane_reproof=True,**result)))


if __name__=='__main__':
    main()
