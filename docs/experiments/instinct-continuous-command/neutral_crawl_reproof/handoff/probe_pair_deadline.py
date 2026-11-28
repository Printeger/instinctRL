"""Add an optimistic recovery timing check to the two-command dynamic screen."""
import argparse
import gzip
import json
import math
from pathlib import Path
import runpy

import numpy as np
import torch
from instinctlab.control.escape_recovery import MOVER_PASSAGE_REPROOF_25
from instinctlab.control.measured_curve_deflection import _fit_return, fresh_motion_history
from instinctlab.control.pipeline import ControlCoreConfig
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked, git

from probe_two_command_path import best_plan, menu, packet_at


def velocity_at_decision(inputs, raw, h, step):
    lane = raw['lane']
    rows = raw['rows']
    start = rows[0]['step']
    rho = math.exp(-.10/raw['lag'])
    with gzip.open(checked(inputs['trace']),'rt') as file:
        next(file)
        prefix = [json.loads(next(file)) for _ in range(212)]
    root = prefix[start-1]['root_state_after_w'][lane]
    velocity = np.asarray(root[7:10])
    pending = np.asarray(prefix[start-1]['v_final_w'][lane])
    for t in range(start, step+1):
        row = rows[t-start]
        velocity = pending+rho*(velocity-pending)
        assert row['step']==t and abs(np.linalg.norm(velocity)-row['speed'])<1.e-10
        pending = np.asarray(row['issued'])
    quat = prefix[step]['attitude_wxyz'][lane]
    attitude = np.asarray([h['body_to_world'](quat,axis) for axis in np.eye(3)])
    assert all(row['raw_event'][lane]['velocity_w_m_s']==rows[step-start]['raw'] for row in prefix[step:212])
    return attitude@velocity, rho


def timely_seconds(first, second, axis, v, target, rho):
    a, b = first@axis,second@axis
    after_first = a+rho*(v-a)
    after_second = b+rho*(after_first-b)
    greatest = after_second
    command = b
    # t204 and t205 execute the two issued actions. Later target updates
    # get an optimistic slew bound on the aligned component alone.
    for _ in range(206,212):
        command = np.minimum(target,command+.30)
        greatest = command+rho*(greatest-command)
    return greatest>=.60*target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    file=Path(__file__).resolve()
    reg=file.with_name('INPUTS.json')
    release=load(reg)
    inputs=load(checked(release['inputs']))
    raw=load(checked(release['opposed_control']))
    for key in ('replay_harness','control_bridge','raw_renderer','renderer_geometry','renderer_rays','renderer_trace'):
        checked(inputs[key])
    assert not stop_paths() and git('rev-parse','HEAD')==inputs['v60_commit']
    h=runpy.run_path(inputs['replay_harness']['path'])['execute'].__globals__
    config=ControlCoreConfig(ics=h['ics'])
    ctrl=h['ClosedControl'](config,h['ray_tensor'],MOVER_PASSAGE_REPROOF_25)
    velocity,rho=velocity_at_decision(inputs,raw,h,203)
    groups=[]
    with torch.inference_mode():
        scan,state,offsets,coast=packet_at(inputs,raw,h)
        assert bool(fresh_motion_history(state.reshape(1,4,13))[0])
        fit=_fit_return(scan,state,h['ray_tensor'],offsets,coast,entry=False,coherent=False)
        assert fit is not None and np.linalg.norm(fit.acceleration,axis=2).max()<=1.05
        first,second,axis,queued=menu(scan,state,ctrl.escape_option,config)
        target=float(state[0,45:48].norm())
        velocity=float(velocity@axis)
        for action in first:
            if action@axis<-.2261:
                continue
            timely=timely_seconds(action,second,axis,velocity,target,rho)
            allowed=timely&((second@axis)<=target+1.e-6)&(np.linalg.norm(second-action,axis=1)<=.30001)
            if not allowed.any():
                continue
            groups.append(best_plan(action[None],second[allowed],fit,coast,queued,axis,h['lags']))
    assert not stop_paths()
    print(save(args.output,dict(inputs=pin(reg),script=pin(file),earlier_probe=pin(file.with_name('probe_two_command_path.py')),
        initial_velocity_along_m_s=velocity,raw_magnitude_m_s=target,stage=203,
        modeled_deadline_first_tracking_packet=211,optimistic_timing_pairs=sum(g['input_pair_count'] for g in groups),
        joint_pass_count=sum(g['dynamic_pass_count'] for g in groups),
        best_joint_dynamic_floor=max(groups,key=lambda x:x['best_dynamic_floor_m'],default=None),
        interpretation='first certified ICS now; second restricted by slew and latest along speed; later issues ideal up to request; dynamic only')))


if __name__=='__main__':
    main()
