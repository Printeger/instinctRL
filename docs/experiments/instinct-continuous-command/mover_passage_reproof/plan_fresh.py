"""Preregister command streams and nominal geometry opportunities without replaying an actor."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch
from instinctlab.observations.moving_sphere import street_sweep_pose
from tools.r6_direct.generate_crazy_commands import generate
from tools.r6_direct.import_legacy import ROOT, pin, stop_paths
from tools.r6_direct.instinct_cycle import save
from tools.r6_direct.replay_clearance import hazard_responses, projected_clearance, static_distance
from training_workloads.unified_h1.command_replay import (
    NOVEL_V60_CLOSURE_FRESH_ID, LANES, STEP_MS, authored_street_sweep,
    load_lane_streams, sampled_events,
)
from training_workloads.unified_h1.escape_holdout_scene import scene_geometry
from training_workloads.unified_h1.robustness_scenarios import reset_spec

parser=ArgumentParser()
parser.add_argument('folder',type=Path)
args=parser.parse_args()
assert not stop_paths(),stop_paths()
args.folder.mkdir(exist_ok=False,parents=True)
run_id=NOVEL_V60_CLOSURE_FRESH_ID
total=120_000
prefix=pin(ROOT/'docs/experiments/instinct-continuous-command/long/PREFIX.json')
known={829377162,876527864,1330121982,1478630124,167149835,185193671} | {2000546218, 1024380226, 2479027, 1982296229, 1999375583, 1469625851, 696110106, 741106095}


def seed(label, index):
    suffix=f'instinctrl-v60-closure/short/{label}/{index}'
    return int.from_bytes(sha256(suffix.encode()).digest()[:4],'big') % 2**31


origins=torch.tensor([[96.*lane,0.,0.] for lane in range(LANES)],dtype=torch.float64)
clock=torch.arange(total//STEP_MS+1,dtype=torch.float64)*STEP_MS/1000.
block=dict(id='continuous-replay-independent',task='pulse_hold')


def opportunities(seed_scene, stream, sweep):
    layouts,streams=load_lane_streams(stream,require_independent=True)
    _,_,vertices,faces=scene_geometry(origins,seed_scene,command_routes=True)
    triangles=np.asarray(vertices)[np.asarray(faces).reshape(-1,3)]
    plan=dict(schema_version='instinctrl-frozen-progress-opportunities/1',run_id=run_id,windows=[],stop_windows=[])
    diagnostics=[]
    restoration=[]
    for lane,(layout,lane_stream) in enumerate(zip(layouts,streams,strict=True)):
        origin=origins[lane].numpy()
        reset=reset_spec(block,seed=seed_scene,split='development',lane=lane,mount_position_m=(0.,0.,.05))
        vectors=np.array([lane_stream['events'][i]['velocity_w_m_s'] for i in sampled_events(lane_stream)[:-1]])
        speed=np.linalg.norm(vectors,axis=1)
        route=origin+np.asarray(reset['root_position_local_m'])+np.cumsum(
            np.concatenate((np.zeros((1,3)),.10*vectors)),axis=0)
        walls=triangles[(np.abs(triangles-origin).max(axis=(1,2))<50.)]
        clear=static_distance(route,walls)-.21
        center=None
        if layout=='street_gap':
            center=street_sweep_pose(origins[lane],clock,torch.tensor(sweep['phase_m'][lane-16],dtype=torch.float64),
                                     ramp_s=sweep['ramp_s'],profile=sweep['profile']).numpy()
            clear=np.minimum(clear,np.linalg.norm(route-center,axis=1)-.55)
        previous=np.vstack((np.zeros((1,3)),vectors[:-1]))
        nominal,_=projected_clearance(route[:-1],previous,vectors,
                                     center[:-1] if center is not None else None,walls,.21)
        hazards=hazard_responses(nominal,nominal,vectors,previous,sustained_opportunity=True)['events']
        for index,event in enumerate(hazards):
            anchor=event['first_feasible_request_step']
            if anchor is not None:
                end=hazards[index+1]['start_step'] if index+1<len(hazards) else len(vectors)
                restoration.append(dict(lane=lane,layout=layout,hazard_start=event['start_step'],anchor=anchor,end=end))
        ranges = ((12,75),(200,1180))
        for sequence,(lo,hi) in enumerate(ranges):
            good=next((start for start in range(lo,hi-20) if speed[start:start+20].min()>=.10
                       and np.linalg.norm(vectors[start:start+20]-vectors[start],axis=1).max()<=.06
                       and clear[start:start+21].min()>=.48),None)
            if good is None:
                raise ValueError(f'lane {lane} planned progress window {sequence}')
            plan['windows'].append(dict(id=f'lane{lane}-progress-{sequence}',layout=layout,lane=lane,
                                        start_step=good,end_step=good+20))
        stopped=0
        for start in np.flatnonzero((speed[:-1]>.03)&(speed[1:]==0.))+1:
            end=start+26
            if start<90 or end>len(speed) or speed[start:end].any() or clear[start:end+1].min()<.55:
                continue
            plan['stop_windows'].append(dict(id=f'lane{lane}-stop-{stopped}',lane=lane,
                                            start_step=int(start),end_step=int(end)))
            stopped+=1
            if stopped==1: break
        if stopped!=1: raise ValueError(f'lane {lane} planned stationary windows')
        diagnostics.append(dict(lane=lane,layout=layout,minimum_nominal_clearance_m=float(clear.min()),
                                maximum_planned_displacement_m=float(np.linalg.norm(route-route[0],axis=1).max())))
    if set(w['layout'] for w in restoration)!=set(layouts):
        raise ValueError('planned restoration coverage across all five layouts')
    return plan,diagnostics,restoration


refused=[]
for index in range(16):
    assert not stop_paths(),stop_paths()
    scene,ccg=seed('scene',index),seed('commands',index)
    assert len({scene,ccg,*known})==len(known)+2
    scratch=save(args.folder/f'prospective-stream-{index}.json',generate(prefix,ccg,total,
                 independent_lanes=True,limit_planned_position=True))
    sweep=authored_street_sweep(scene,ramped=True,observable=True)
    try:
        result,lanes,restoration=opportunities(scene,scratch,sweep)
    except ValueError as problem:
        refused.append(dict(index=index,scene_seed=scene,generator_seed=ccg,constraint=str(problem)))
        continue
    stream=save(args.folder/'STREAMS.json',json.loads(Path(scratch['path']).read_text()))
    scheduled=save(args.folder/'OPPORTUNITIES.json',result)
    anchors=save(args.folder/'RESTORATION-WINDOWS.json',dict(schema='instinctrl-v60-fresh-restoration/1',
        method='pre-flight nominal integrated Crazy commands and authored geometry; 0.21 m nominal envelope',
        restoration_windows=restoration))
    field=dict(scene_seed=scene,generator_seed=ccg,stream=stream,opportunities=scheduled,
               street_sweep=sweep,pulse_acceleration_m_s2=.5,restoration=anchors,index=index,
               selection_basis='first hash-derived pair satisfying authored geometry windows without an actor',
               refused_nominal_preflight=refused)
    save(args.folder/'SEEDS.json',field)
    save(args.folder/'PLANNING_CHECK.json',dict(**field,predicted_plan_only=True,scheduled_progress=len(result['windows']),
         scheduled_stops=len(result['stop_windows']),lanes=lanes))
    print({k:field[k] for k in ('scene_seed','generator_seed','index')},len(result['windows']),len(result['stop_windows']))
    break
else:
    raise SystemExit(json.dumps(refused))
