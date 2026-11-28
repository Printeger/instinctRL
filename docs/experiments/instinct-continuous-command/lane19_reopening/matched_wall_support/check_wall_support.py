"""Test fixed common-footprint wall halves after strict registration rejects."""
import argparse
import json
import math
from pathlib import Path
import runpy

import numpy as np
import torch

from instinctlab.control import registered_mover_entry as registered
from instinctlab.control.measured_curve_deflection import _Fit, fresh_motion_history
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.source import checked


def motion_fit(directions, offsets):
    distances=offsets[:, -1:] - offsets
    translation=np.linalg.lstsq(directions,distances,rcond=None)[0].T
    residual=np.linalg.norm(directions@translation.T-distances,axis=0).max()
    return translation,float(residual)


def matched_translation(scan,rays,rotation,basis,up,options):
    scans=scan.reshape(4,3,-1).numpy()
    planes=registered._observed_planes(scan.numpy(),rays.numpy(),rotation,basis,up)
    info=dict(plane_count=len(planes))
    if len(planes)<3:
        return None,info|{'reason':'insufficient_planes'}
    directions=np.stack([p[0] for p in planes])@basis
    min_singular=float(np.linalg.svd(directions,compute_uv=False)[-1])
    info['minimum_singular']=min_singular
    if min_singular<.35:
        return None,info|{'reason':'poor_conditioning'}
    global_reference,error=motion_fit(directions,np.stack([p[1] for p in planes]))
    info['original_pool_residual_m']=error
    allowed=registered._DISTANCE_ERROR
    if error>allowed:
        return None,info|{'reason':'poor_original_pool'}
    surfaces=[]
    mount=np.asarray(registered._MOUNT)
    for payload,pose in zip(scans,rotation,strict=True):
        points,normals=registered._surface_normals(payload,rays.numpy())
        surfaces.append(((points+mount)@pose,normals@pose,np.linalg.norm(points,axis=1)))
    aligned=global_reference@basis.T
    pooled,left,right=[],[],[]
    for number,(normal,signed,*_) in enumerate(planes):
        tangent=np.cross(up,normal)
        footprints=[]
        for i,(points,normals,length) in enumerate(surfaces):
            along=points@tangent+aligned[i]@tangent
            height=points@up
            offset=points@normal
            coherent=(normals@normal>=math.cos(.045)) & (length<=registered._RANGE_LIMIT) & (abs(offset-signed[i])<=.013)
            if coherent.sum()<24:
                return None,info|dict(reason='insufficient_wall_support',plane_index=number)
            footprints.append((along[coherent],height[coherent],offset[coherent]))
        low_x=max(f[0].min() for f in footprints)+options['erosion_m']
        high_x=min(f[0].max() for f in footprints)-options['erosion_m']
        low_z=max(f[1].min() for f in footprints)+options['erosion_m']
        high_z=min(f[1].max() for f in footprints)-options['erosion_m']
        gate=options['gap_m']/2.
        center=(low_x+high_x)/2.
        if high_x-low_x<.14 or high_z-low_z<.20:
            return None,info|dict(reason='footprint_does_not_overlap',plane_index=number,
                                  overlap_w_m=float(high_x-low_x),overlap_height_m=float(high_z-low_z))
        ticks=[]
        for x,z,d in footprints:
            common=(x>=low_x)&(x<=high_x)&(z>=low_z)&(z<=high_z)
            halves=(common&(x<=center-gate),common&(x>=center+gate))
            counts=[int(mask.sum()) for mask in halves]
            if min(counts)<options['minimum_half_returns'] or min(np.ptp(z[m]) for m in halves)<options['minimum_half_height_m']:
                return None,info|dict(reason='inadequate_disjoint_halves',plane_index=number,
                                      side_counts=counts,overlap_width_m=float(high_x-low_x))
            union=halves[0]|halves[1]
            med=float(np.median(d[union]))
            if float(np.sqrt(np.mean(np.square(d[union]-med))))>.006:
                return None,info|dict(reason='common_patch_nonplanar',plane_index=number)
            ticks.append(dict(pool=med,left=float(np.median(d[halves[0]])),right=float(np.median(d[halves[1]])),counts=counts))
        info.setdefault('patches',[]).append(dict(plane_index=number,normal=normal.tolist(),footprint=[float(v) for v in
                                              (low_x,high_x,low_z,high_z)],frames=ticks))
        for array,field in ((pooled,'pool'),(left,'left'),(right,'right')):
            array.append([t[field] for t in ticks])
    pooled,left,right=(np.asarray(x) for x in (pooled,left,right))
    median_fit,pool_res=motion_fit(directions,pooled)
    left_fit,left_res=motion_fit(directions,left)
    right_fit,right_res=motion_fit(directions,right)
    pool_shift=float(np.linalg.norm(median_fit-global_reference,axis=1).max())
    errors=[float(np.linalg.norm(x-median_fit,axis=1).max()) for x in (left_fit,right_fit)]
    info.update(pooled_residual_m=pool_res,half_residuals_m=[left_res,right_res],half_errors_m=errors,
                original_pool_difference_m=pool_shift)
    # Check the held-out half before permitting either pooled result to
    # pass. The same gate can be checked with a synthetic spatial offset.
    negative_right=right.copy()
    negative_right[0,0]+=options['mixed_surface_shift_m']
    mixed,mixed_res=motion_fit(directions,negative_right)
    mixed_delta=float(np.linalg.norm(mixed-median_fit,axis=1).max())
    info['mixed_surface_control']=dict(accepted=max(mixed_delta,mixed_res)<=allowed,
                                      half_error_m=mixed_delta,half_residual_m=mixed_res)
    if max(pool_shift,pool_res,left_res,right_res,*errors)>allowed:
        return None,info|{'reason':'common_support_still_disagrees'}
    return median_fit,info|{'reason':'accepted'}


def coordinates(scan,state,rays):
    shape=registered._return_shapes(scan,state,rays,entry=False,align_gravity=True)
    if shape is None:
        return None
    _,_,time,_=shape
    h=state.reshape(4,13)
    up=-h[-1,3:6].numpy()
    up/=np.linalg.norm(up)
    f=np.array([1.,0.,0.])-up*up[0]
    f/=np.linalg.norm(f)
    basis=np.stack((f,np.cross(up,f)),axis=1)
    rotation=registered._rotations(h,time)
    return shape,rotation,basis,up


def fallback_fit(scan,state,rays,options):
    original=registered._registered_fit(scan,state,rays,entry=False)
    if original is not None:
        return original,dict(reason='unchanged_v60',wall=None)
    if not bool(fresh_motion_history(state.reshape(1,4,13))[0]):
        return None,dict(reason='stale_history',wall=None)
    frame=coordinates(scan,state,rays)
    if frame is None:
        return None,dict(reason='missing_mover_shape',wall=None)
    shape,rotation,basis,up=frame
    translation,wall=matched_translation(scan,rays,rotation,basis,up,options)
    if translation is None:
        return None,dict(reason=wall['reason'],wall=wall)
    centers,radius,_,dt=shape
    mount=np.asarray(registered._MOUNT)
    points=(centers.numpy()+mount@rotation-mount)@basis+translation
    gaps=dt.numpy()
    velocity=np.diff(points,axis=0)/gaps[:,None]
    acceleration=np.diff(velocity,axis=0)/(.50*(gaps[1:]+gaps[:-1]))[:,None]
    if np.linalg.norm(velocity[-1])>1.42 or np.linalg.norm(acceleration,axis=1).max()>1.05:
        return None,dict(reason='mover_motion_limit',wall=wall)
    fit=_Fit(centers[-1].numpy(),float(radius),np.broadcast_to(velocity[-1]@basis.T,(3,3)),
             np.broadcast_to(acceleration,(3,2,2)))
    return (fit,basis,float(gaps[-1])),dict(reason='fixed_common_footprint',wall=wall)


def replay_samples(refs):
    raw=load(checked(refs['closed_pair']))
    base=load(checked(refs['readback_inputs']))
    cfg=load(checked(base['reserve_inputs']))
    source=runpy.run_path(str(checked(base['reserve_script'])))
    data=source['inputs'](cfg)
    h,rows=data['scene'],data['archive']
    start, lane=refs['start'],cfg['protocol']['lane']
    r=np.array([h['body_to_world'](rows[start]['attitude_wxyz'][lane],e) for e in np.eye(3)])
    visible=[h['reconstruct'](data['initial'],rows[t],rows[t-1],lane,h['rays'],data['mesh'])[0]
             for t in range(start-3,start+1)]
    solid=data['raw'][start:refs['through']+1,:-1,lane].reshape(-1,7).astype(float)
    movers=cfg['protocol']['cases']
    centers=data['motion']['mover_bank'](solid,movers,np.arange(len(solid))*.005,start)
    for mode,arms in raw['outcomes'].items():
        for case,arm in arms.items():
            history=np.array(rows[start]['actor_state'][lane],dtype=np.float32)
            frames=list(visible)
            records={v['step']:v for v in arm['records']}
            trajectory=centers[list(movers).index(case)]
            for step in range(start+1,refs['through']+1):
                assert not stop_paths(),stop_paths()
                current=np.array(rows[step]['actor_state'][lane][-1],dtype=np.float32)
                current[:3]=0.
                current[3:6]=r@np.array([0.,0.,-1.])
                current[6:9]=r@np.array(records[step]['request_w'])
                current[9:12]=r@np.array(records[step-1]['issued_w'])
                history=np.vstack((history[1:],current))
                frames=frames[1:]+[h['render'](np.array(records[step]['position_before']),r,trajectory[(step-start)*20],data['mesh'])]
                if step>=refs['from']:
                    yield mode,case,step,torch.tensor(np.concatenate(frames)).reshape(1,12,59,360),torch.tensor(history).reshape(1,52),data['ray']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    file=Path(__file__).resolve()
    registry=file.with_name('INPUTS.json')
    refs=load(registry)
    scores=[]
    with torch.inference_mode():
        for mode,case,k,scan,state,rays in replay_samples(refs):
            original=registered._registered_fit(scan,state,rays,entry=False)
            frame=coordinates(scan,state,rays)
            score=dict(mode=mode,case=case,step=k,original_fit=original is not None)
            if frame:
                _,rotation,basis,up=frame
                _,score['matched_support']=matched_translation(scan,rays,rotation,basis,up,refs['options'])
            elif frame is None:
                score['matched_support']=dict(reason='missing_mover_shape')
            scores.append(score)
    assert not stop_paths(),stop_paths()
    print(json.dumps(dict(raw=save(args.output,dict(registry=pin(registry),script=pin(file),scores=scores)),
        scores=[(r['mode'],r['case'],r['step'],r['original_fit'],r['matched_support']['reason']) for r in scores])),flush=True)


if __name__=='__main__':
    main()
