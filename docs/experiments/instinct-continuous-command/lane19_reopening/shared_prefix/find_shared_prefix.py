"""Co-optimize identical issued prefixes with optional brief protective motion."""
import argparse
import json
from pathlib import Path
import runpy
import shutil

import numpy as np
from scipy.optimize import LinearConstraint, minimize
import torch

from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.replay_clearance import static_distance
from tools.r6_direct.source import checked
from training_workloads.unified_h1.escape_holdout_scene import scene_geometry


def slot_map(active, split):
    suffix = np.flatnonzero(active >= split)
    left = np.arange(len(active))
    right = left.copy()
    right[suffix] = np.arange(len(active), len(active) + len(suffix))
    return np.array([left, right]), len(active) + len(suffix)


def compile_run(data, cfg, mode, head):
    start, stop, end = (cfg[k] for k in ('start','optimization_end','score_end'))
    begin, resume = (x-start for x in cfg['zero_window'])
    stop_issued = begin+mode['extra_motion_ticks']
    active = np.r_[np.arange(1,stop_issued),np.arange(resume,cfg['return_to_ccg']-start)]
    mapping, slots = slot_map(active,mode['split_step']-start)
    issued = data['positive'].copy()
    issued[:active[-1]+1] = 0.
    issued[0]=head['world']
    anchor = begin*20
    end_neutral = resume*20
    deadline = anchor+200
    index = np.arange(len(data['time']))
    # Stop position can peak during protective motion; once physical
    # neutral begins, the rest of its path is a braking line segment.
    drift_samples = np.r_[np.arange(anchor,min(end_neutral,(stop_issued+1)*20)+1,20),end_neutral]
    fit = index <= (stop-start)*20
    A, b, limits = [], [], []
    models = []
    owners = np.eye(slots)[mapping]
    for group, owner in enumerate(owners):
        for k in range(1,len(active)):
            if active[k]==resume:
                if mode['restart_slew']:
                    A.append(owner[k])
                    b.append(np.zeros(2))
                    limits.append(cfg['slew_m_s']**2)
            else:
                A.append(owner[k]-owner[k-1])
                b.append(np.zeros(2))
                limits.append(cfg['slew_m_s']**2)
        A.append(owner[-1])
        b.append(-issued[active[-1]+1,:2])
        limits.append(cfg['slew_m_s']**2)
        position, speed, pcoef, vcoef = [], [], [], []
        for base,w,base_speed,sw in data['models']:
            position.append(base+w@issued)
            speed.append(base_speed+sw@issued)
            pcoef.append(w[:,active]@owner)
            vcoef.append(sw[:,active]@owner)
        position,speed,pcoef,vcoef=(np.stack(x) for x in (position,speed,pcoef,vcoef))
        for lag in range(len(position)):
            for instant in drift_samples:
                if instant==anchor:
                    continue
                d = position[lag,instant]-position[lag,anchor]
                A.append(pcoef[lag,instant]-pcoef[lag,anchor])
                b.append(d[:2])
                limits.append((cfg['drift_m']-cfg['drift_headroom_m'])**2-d[2]**2)
            d = speed[lag,deadline]
            A.append(vcoef[lag,deadline])
            b.append(d[:2])
            limits.append((cfg['stop_speed_m_s']-cfg['speed_headroom_m_s'])**2-d[2]**2)
        models.append(dict(position=position,speed=speed,pcoef=pcoef,vcoef=vcoef,
                           centers=data['movers'][data['groups'][group]]))
    A= np.vstack([np.eye(slots),owners[0,0],A])
    b= np.vstack([np.zeros((slots,2)),-issued[0,:2],b])
    limits=np.r_[[cfg['cap_m_s']**2]*slots,cfg['slew_m_s']**2-issued[0,2]**2,limits]
    return dict(active=active,slots=slots,mapping=mapping,fixed=issued,A=A,b=b,limits=limits,
                models=models,fit=fit,anchor=anchor,end_neutral=end_neutral,deadline=deadline)


def seed_vector(info,cfg,mode,settings,cap_step):
    active = info['active']
    shared_end=mode['split_step']-cfg['start']
    resume=cfg['zero_window'][1]-cfg['start']
    first=info['fixed'][0,:2]
    values=np.zeros((info['slots'],2))
    for group, ids in enumerate(info['mapping']):
        prior=first
        for local, step in enumerate(active):
            common = step < shared_end
            angle = np.deg2rad(settings[0 if common else group+1])
            objective=.36*np.array([np.cos(angle),np.sin(angle)])
            if step>=active[-1]-2:
                objective=info['fixed'][active[-1]+1,:2]
            if step==resume:
                prior=np.zeros(2)
                move=cfg['slew_m_s'] if mode['restart_slew'] else cfg['cap_m_s']
            else:
                move=cfg['slew_m_s']
            prior=cap_step(objective,prior,move)
            values[ids[local]]=prior
    return values


def exact(info,data,moves,walls=False):
    outputs=[]
    for i, model in enumerate(info['models']):
        position=model['position'].copy()
        motion=model['speed'].copy()
        position[:,:,:2]+=model['pcoef']@moves
        motion[:,:,:2]+=model['vcoef']@moves
        clearness=np.linalg.norm(position[:,None]-model['centers'],axis=3)-data['envelope']
        a,b,d=(info[k] for k in ('anchor','end_neutral','deadline'))
        drift=np.linalg.norm(position[:,a:b+1]-position[:,a:a+1],axis=2).max()
        speed=np.linalg.norm(motion[:,d:b+1],axis=2).max()
        static = min(float((static_distance(p,data['solids'])-data['body']).min()) for p in position) if walls else None
        idx=np.unravel_index(clearness.argmin(),clearness.shape)
        issue=info['fixed'].copy()
        issue[info['active'],:2]=moves[info['mapping'][i]]
        outputs.append(dict(position=position,clearance=clearness,
                     physical_min_m=float(clearness[idx]), lag_s=data['config']['lags_s'][idx[0]],
                     case=data['case_names'][data['groups'][i][idx[1]]], time_s=data['config']['start']*.1+float(data['time'][idx[2]]),
                     drift_m=float(drift),quiet_speed_m_s=float(speed),static_min_m=static,issues=issue))
    return outputs


def solve(info,data,cfg,mode,seeds):
    runtime=data['runtime']
    n=info['slots']
    A,b,limits=(info[k] for k in ('A','b','limits'))

    def residual(x):
        u=A@x[:-1].reshape(n,2)+b
        return limits-(u*u).sum(axis=1)

    def jacobian(x):
        u=A@x[:-1].reshape(n,2)+b
        return np.column_stack(((-2.*A[:,:,None]*u[:,None]).reshape(len(A),n*2),np.zeros(len(A))))

    merit=np.r_[np.zeros(2*n),-1.]
    fit=info['fit']
    best=None
    for seed in seeds:
        samples=[set() for _ in info['models']]
        x=np.r_[seed_vector(info,cfg,mode,seed,runtime['cap_step']).flatten(),-.25]
        for step in range(cfg['refinements']):
            outs=exact(info,data,x[:-1].reshape(n,2))
            constraints=[]
            for group, model in enumerate(info['models']):
                out=outs[group]
                samples[group].update(runtime['close_samples'](out['clearance'][:,:,fit],cfg['witnesses_per_case']))
                constraints.append(runtime['support'](model['position'][:,fit],model['pcoef'][:,fit],
                    sorted(samples[group]),out['position'][:,fit],model['centers'][:,fit],data['envelope'][fit],n))
            combined=LinearConstraint(np.vstack([c.A for c in constraints]),np.concatenate([c.lb for c in constraints]),np.inf)
            found=minimize(lambda y:-y[-1],x,jac=lambda _:merit,method='SLSQP',
                           constraints=[combined,dict(type='ineq',fun=residual,jac=jacobian)],
                           bounds=[(-cfg['cap_m_s'],cfg['cap_m_s'])]*(2*n)+[(-.6,.67)],
                           options=dict(ftol=2.e-10,maxiter=cfg['max_solver_iterations']))
            x=found.x
            controls=x[:-1].reshape(n,2)
            outs=exact(info,data,controls)
            score=min(o['physical_min_m'] for o in outs)
            pass_stop=all(o['drift_m']<=cfg['drift_m'] and o['quiet_speed_m_s']<=cfg['stop_speed_m_s'] for o in outs)
            if residual(x).min()>=-2.e-8 and pass_stop and (best is None or score>best['clearance_m']):
                complete=exact(info,data,controls,walls=True)
                before=mode['split_step']-cfg['start']
                assert np.array_equal(complete[0]['issues'][:before],complete[1]['issues'][:before])
                begin,end=(t-cfg['start'] for t in cfg['zero_window'])
                assert all(not out['issues'][begin+mode['extra_motion_ticks']:end].any() for out in complete)
                best=dict(clearance_m=min(min(o['static_min_m'],o['physical_min_m']) for o in complete),
                          mover_clearance_m=score, seed=seed, iteration=step, solver_success=bool(found.success),
                          group_outputs=[{k:(v.tolist() if k=='issues' else v)
                                          for k,v in out.items() if k not in ('position','clearance')} for out in complete])
            if best is not None and best['clearance_m']>=cfg['wanted_margin_m']:
                return best
    return best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    assert not stop_paths() and shutil.disk_usage(args.output.parent).free>=30*2**30+10*2**20
    here=Path(__file__).resolve()
    registry=here.with_name('INPUTS.json')
    inputs=load(registry)
    split=runpy.run_path(str(checked(inputs['split_script'])))
    data=split['load_inputs'](load(checked(inputs['split_inputs'])))
    cfg=data['config']|inputs['settings']
    original=load(checked(inputs['exact_menu']))
    drift=load(checked(inputs['drift_screen']))
    strict_ids={row['menu_id'] for row in drift['modes']['full_speed_restart']['rows'] if row['worst_upper_m']>=.15}
    assert len(original['menu'])==142 and len(strict_ids)==17
    cases=list(data['origin']['protocol']['cases'])
    geometry=load(checked(data['origin']['geometry']))
    _,_,vertex,faces=scene_geometry(torch.tensor(geometry['env_origins']),cfg['scene_seed'],command_routes=True)
    data.update(case_names=cases,groups=[[cases.index(c) for c in group] for group in inputs['case_groups']],
                body=data['envelope']-.34, solids=np.asarray(vertex)[np.asarray(faces).reshape(-1,3)])
    order=sorted(original['menu'],key=lambda row:(row['full_speed_restart']['worst_upper_m'],row['rank_m']),reverse=True)
    results={}
    for name,mode in inputs['modes'].items():
        candidates=[r for r in order if r['menu_id'] in strict_ids] if mode['strict_bank'] else order
        samples=[]
        for head in candidates:
            assert not stop_paths(),stop_paths()
            found=solve(compile_run(data,cfg,mode,head),data,cfg,mode,mode['seeds'])
            samples.append(dict(menu_id=head['menu_id'],best=found))
            if len(samples)%20==0:
                print(json.dumps(dict(mode=name,attempts=len(samples))),flush=True)
            if found is not None and found['clearance_m']>=cfg['wanted_margin_m']:
                break
        winners=[r for r in samples if r['best'] is not None]
        best=max(winners,key=lambda r:r['best']['clearance_m'],default=None)
        results[name]=dict(menu_population=len(candidates),tested=len(samples),winner=best,outcomes=samples)
        print(json.dumps(dict(mode=name,population=len(candidates),tested=len(samples),
                     best_floor_m=None if best is None else best['best']['clearance_m'])),flush=True)
        save(args.output.with_name(args.output.stem+'-'+name+'.json'),
             dict(inputs=pin(registry),script=pin(here),mode=name,result=results[name]))
    assert not stop_paths(),stop_paths()
    print(json.dumps(dict(raw=save(args.output,dict(inputs=pin(registry),script=pin(here),results=results)))),flush=True)


if __name__=='__main__':
    main()
