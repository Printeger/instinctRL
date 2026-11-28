"""Screen earlier full-ICS first issues for a shared lane 19 reserve trajectory."""
import argparse
import gzip
import json
from pathlib import Path
import runpy
import shutil

import numpy as np
import torch

from instinctlab.control.escape_passage import advance_command_response, observed_path_clearance
from instinctlab.control.escape_recovery import MOVER_PASSAGE_REPROOF_25
from instinctlab.control.measured_curve_deflection import fresh_motion_history
from instinctlab.control.observed_mover_preview import _rank_continuations
from instinctlab.control.pipeline import ControlCoreConfig
from instinctlab.control.registered_mover_entry import _piecewise_clearance, _registered_fit
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked, git
from training_workloads.unified_h1.escape_holdout_scene import scene_geometry


def inputs(refs):
    cfg = refs['protocol']
    origin = load(checked(refs['reopening_inputs']))
    for name, entry in origin.items():
        if name != 'v60_commit':
            checked(entry)
    assert git('rev-parse','HEAD')==origin['v60_commit'] and not git('branch','--show-current')
    later = load(checked(refs['shared_inputs']))
    parking_process = load(checked(later['split_inputs']))
    parking = load(checked(parking_process['common_inputs']))
    replay = runpy.run_path(str(checked(refs['closed_script'])))
    engine = runpy.run_path(str(checked(refs['shared_script'])))
    motion = runpy.run_path(str(checked(parking['parking_script'])))
    h = runpy.run_path(origin['fixture']['path'])['execute'].__globals__
    with gzip.open(origin['trace']['path'],'rt') as stream:
        initial = json.loads(next(stream))
        archive = [json.loads(next(stream)) for _ in range(cfg['score_end'])]
    assert initial==h['initial'] and [row['step'] for row in archive]==list(range(len(archive)))
    ray = h['ray_tensor']
    near = h['origins'][cfg['lane']].numpy()
    triangles=h['triangles']
    local=((triangles.max(axis=1)>=near-45.)&(triangles.min(axis=1)<=near+45.)).all(axis=1)
    _,_,vertices,idx=scene_geometry(h['origins'],cfg['scene_seed'],command_routes=True)
    solids=np.asarray(vertices)[np.asarray(idx).reshape(-1,3)]
    raw=np.memmap(checked(refs['dense']),'<f4','r',shape=(1200,21,32,7))
    return dict(archive=archive,initial=initial,scene=h,ray=ray,mesh=triangles[local],solids=solids,raw=raw,
                issued_bank=replay['issued_bank'],runtime=runpy.run_path(str(checked(refs['common_script']))),
                engine=engine,motion=motion)


def action_menu(data, cfg, anchor):
    lane = cfg['lane']
    archive = data['archive']
    h = data['scene']
    option_cfg = ControlCoreConfig(ics=h['ics'])
    option = h['ClosedControl'](option_cfg,data['ray'],MOVER_PASSAGE_REPROOF_25).escape_option
    banks = h['observer']()
    for row in archive[:anchor+1]:
        state=torch.tensor(row['actor_state'][lane]).reshape(1,52)
        banks=[advance_command_response(*obs,state,lag=lag) for obs,lag in zip(banks,h['lags'],strict=True)]
    frames=[h['reconstruct'](data['initial'],archive[k],archive[k-1],lane,h['rays'],data['mesh'])
            for k in range(anchor-3,anchor+1)]
    scan=torch.tensor(np.concatenate([frame for frame,_ in frames])).reshape(1,12,59,360)
    reg=_registered_fit(scan,state,data['ray'],entry=False)
    assert reg is not None and bool(fresh_motion_history(state.reshape(1,4,13))[0])
    fit,basis,spacing=reg
    coast=torch.stack([obs[1][0] for obs in banks]).numpy()
    plane=(coast@basis[:2])@basis.T
    previous=state[0,48:51]
    selected=torch.tensor([archive[anchor]['selected_action'][lane]])
    stock=option.issued(scan,state,selected,option_cfg)
    stock_v=stock.v_final_b.numpy()
    available=data['issued_bank']
    angles=np.arange(72)*np.pi/36.
    direction=np.stack((np.cos(angles),np.sin(angles)),axis=1)
    axes=direction@basis.T
    velocity=state.new_tensor(np.concatenate([s*axes for s in (.15,.225,.2999,.375,.4499)]))
    _,forth,normal=available(option,scan,state,option_cfg,velocity)
    step=forth-previous
    targets=previous+step*(.30/step.norm(dim=1,keepdim=True).clamp_min(.30))
    actions,first,valid=available(option,scan,state,option_cfg,targets)
    valid &= normal & ((first-previous).norm(dim=1).numpy()<=.30001)
    moving=_piecewise_clearance(fit,plane,previous.numpy(),first.numpy(),forth.numpy(),.60,1.,spacing)
    backup=_piecewise_clearance(fit,plane,previous.numpy(),first.numpy(),np.zeros_like(forth.numpy()),.10,.90,spacing)
    surface=torch.cat([observed_path_clearance(scan,data['ray'],state.new_tensor(coast[1:2]),
                                               batch[None],hold_s=.60)[0] for batch in forth.split(24)]).numpy()
    surface-=.055+.10*(first-forth).norm(dim=1).numpy()
    floor=np.minimum(np.minimum(moving,backup),surface)
    rank=_rank_continuations(fit,coast,previous.numpy(),forth.numpy())
    stock_rank=float(_rank_continuations(fit,coast,previous.numpy(),stock_v)[0])
    stock_floor=float(_piecewise_clearance(fit,plane,previous.numpy(),stock_v,stock_v,.60,1.,spacing)[0])
    allowed=np.flatnonzero(valid & (floor>=.39))
    q=archive[anchor]['attitude_wxyz'][lane]
    rotation=np.array([h['body_to_world'](q,v) for v in np.eye(3)])
    issued_world=first.numpy()@rotation
    requested=state[0,45:48].numpy()
    row=archive[anchor]
    audit=dict(native_beta=row['ics_beta'][lane],bare_stock_body=stock_v[0].tolist(),
               native_body=row['v_final_b'][lane],local_wall_stopping_hold_s=row['local_wall_stopping_hold_s'][lane],
               stock_short_floor_m=stock_floor,stock_rank_m=stock_rank,capture_error_m=max(x or 0. for _,x in frames),
               absolute_population=int(len(allowed)),rank_gain_count=int((rank[allowed]>=stock_rank+.08).sum()),
               nonzero_population=int((np.linalg.norm(issued_world[allowed],axis=1)>.01).sum()))
    assert len(allowed)>0
    chosen=[]
    for angle in np.arange(cfg['directions'])*2.*np.pi/cfg['directions']:
        axis=np.array([np.cos(angle),np.sin(angle)])
        far=allowed[np.argmax(issued_world[allowed,:2]@axis)]
        if far not in chosen:
            chosen.append(int(far))
    if np.argmax(np.where(valid,floor,-np.inf)) not in chosen:
        chosen.append(int(np.argmax(np.where(valid,floor,-np.inf))))
    return audit, [dict(menu_id=i,world=issued_world[i].tolist(),raw_action=actions[i].tolist(),
                        first_b=first[i].tolist(),planned_next_b=forth[i].tolist(),certificate_m=float(floor[i]),
                        rank_m=float(rank[i]),request_fraction=float(first[i].numpy()@requested/(requested@requested)))
                   for i in chosen]


def prepare(data,cfg,anchor):
    lane,end=cfg['lane'],cfg['score_end']
    prior=data['archive'][anchor-1]
    vehicle=np.array(prior['root_state_after_w'][lane])
    pending=np.array(prior['v_final_w'][lane])
    source=data['raw']
    joint=np.concatenate((source[anchor:end,:-1,lane].reshape(-1,7),source[end-1,-1:,lane])).astype(float)
    assert np.allclose(vehicle[:3],joint[0,:3],atol=2.e-5,rtol=0.)
    t=np.arange(len(joint))*.005
    models=[data['motion']['propagation'](t,end-anchor,lag,vehicle[:3],vehicle[7:10],pending) for lag in cfg['lags_s']]
    movers=data['motion']['mover_bank'](joint,cfg['cases'],t,anchor)
    names=list(cfg['cases'])
    groups=[[names.index(x) for x in group] for group in cfg['case_groups']]
    issued=np.array([row['raw_event'][lane]['velocity_w_m_s'] for row in data['archive'][anchor:end]])
    return dict(config=cfg | {'start':anchor},positive=issued,models=models,time=t,movers=movers,case_names=names,
                groups=groups,envelope=.34+joint[:,6],body=joint[:,6],solids=data['solids'],runtime=data['runtime'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    assert not stop_paths() and shutil.disk_usage(args.output.parent).free>=30*2**30+10*2**20
    file=Path(__file__).resolve()
    reg=file.with_name('INPUTS.json')
    refs=load(reg)
    cfg=refs['protocol']
    data=inputs(refs)
    engine=data['engine']
    summaries={}
    for anchor in cfg['anchors']:
        assert not stop_paths(),stop_paths()
        with torch.inference_mode():
            bank,first=action_menu(data,cfg,anchor)
        batch=prepare(data,cfg,anchor)
        profile=batch['config']
        branches={}
        for label, mode in refs['modes'].items():
            trials=[]
            for head in first:
                assert not stop_paths(),stop_paths()
                state=engine['compile_run'](batch,profile,mode,head)
                result=engine['solve'](state,batch,profile,mode,mode['seeds'])
                trials.append(dict(head=head,best=result))
            best=max((r for r in trials if r['best'] is not None),default=None,
                     key=lambda r:r['best']['clearance_m'])
            branches[label]=dict(attempts=len(trials),best=best,trials=trials)
        summaries[str(anchor)]=dict(menu=bank,modes=branches)
        save(args.output.with_name(args.output.stem+'-'+str(anchor)+'.json'),
             dict(registry=pin(reg),script=pin(file),anchor=anchor,results=summaries[str(anchor)]))
        print(json.dumps(dict(anchor=anchor,menu=bank,
               results={k:None if v['best'] is None else v['best']['best']['clearance_m'] for k,v in branches.items()})),
              flush=True)
    assert not stop_paths(),stop_paths()
    print(json.dumps(dict(raw=save(args.output,dict(registry=pin(reg),script=pin(file),anchors=summaries)))),flush=True)


if __name__=='__main__':
    main()
