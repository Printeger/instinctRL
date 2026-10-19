"""Visualization ONLY: render hash-bound original R5 poses in Isaac Sim.

Never imports a trainer, optimizer, environment stepping loop or policy. The
frozen checkpoint is verified as the provenance of the original raw trajectory.
USD stage edits and presentation materials exist only in this new replay stage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def pin(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(Path(args.inputs).read_text())
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(args.inputs).resolve().parents[4]
    start = time.time()
    def check():
        for p in [root/'STOP', root/'coordination/STOP', root/'coordination/commander/STOP',
                  root/'coordination/executor/STOP', root/'coordination/research/STOP',
                  Path(args.inputs).parent/'STOP', root.parent/'STOP',
                  root.parent/'instinctRL_coordination/STOP',
                  root.parent/'instinctRL_coordination/commander/STOP',
                  root.parent/'instinctRL_coordination/executor/STOP']:
            if p.exists():
                raise RuntimeError(f'STOP: {p}')
        if time.time()-start > cfg['limits']['wall_seconds']:
            raise TimeoutError('Visualization wall budget')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > cfg['limits']['max_output_bytes']:
            raise RuntimeError('Visualization output budget')
    check()
    for record in cfg['sources']+cfg['assets_and_source']+[v[k] for v in cfg['clips'] for k in ('raw','evaluation','checkpoint')]:
        assert pin(record['path'])['sha256'] == record['sha256'], record['path']
    source = Path(cfg['source_snapshot'])
    sys.path.insert(0, str(source/'isaaclab-training'))
    sys.path.insert(0, str(source))
    from training_workloads.unified_h1.asset_closure import validate_asset_closure
    asset_root = root/'runtime_assets/Assets/Isaac/5.1'
    closure = validate_asset_closure(asset_root, 'f64a0561477efcc612761f6b3034f9612f2d3cb42832138e86ff7a4c18e1ef52')
    os.environ['ISAACSIM_ASSET_ROOT'] = str(asset_root)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('render timeout')))
    signal.alarm(cfg['limits']['wall_seconds'])
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 960, 'height': 640,
                         'renderer': 'RayTracedLighting', 'anti_aliasing': 2,
                         'multi_gpu': False, 'active_gpu': 0, 'hide_ui': True})
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    from pxr import UsdGeom, UsdLux, Gf
    from instinctlab.task.scene_h1 import combined_scene_triangles
    from training_workloads.unified_h1.evaluation import sensor_origin_surface_distance
    torch.set_num_threads(2)
    omni.timeline.get_timeline_interface().stop()
    rep.orchestrator.set_capture_on_play(False)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    light = UsdLux.DomeLight.Define(stage, '/World/ReplayLight')
    light.CreateIntensityAttr(300)
    light.CreateColorAttr((.85,.91,1.))
    sun = UsdLux.DistantLight.Define(stage, '/World/ReplaySun')
    sun.CreateIntensityAttr(800)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set((25,-25,-35))
    points, indices, identities = combined_scene_triangles(torch.zeros((1,3)))
    for i,name in enumerate(('Floor','XWall','YWall')):
        mesh = UsdGeom.Mesh.Define(stage, '/World/'+name)
        mesh.CreatePointsAttr(points[4*i:4*i+4])
        mesh.CreateFaceVertexCountsAttr([3,3])
        mesh.CreateFaceVertexIndicesAttr([v-4*i for v in indices[6*i:6*i+6]])
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([[(.22,.26,.30),(.39,.46,.54),(.44,.50,.57)][i]])
    drone = UsdGeom.Xform.Define(stage, '/World/ReplayDrone')
    translate = drone.AddTranslateOp()
    orient = drone.AddOrientOp()
    asset = stage.DefinePrim('/World/ReplayDrone/Asset')
    asset.GetReferences().AddReference(str(source/'isaaclab-training/instinctlab/assets/taslab_uav/taslab_uav.usd'))
    def line(name, pts, color, width=.012):
        c=UsdGeom.BasisCurves.Define(stage, '/World/'+name)
        c.CreateTypeAttr('linear'); c.CreateCurveVertexCountsAttr([len(pts)])
        c.CreatePointsAttr(pts);c.CreateWidthsAttr([width]);c.SetWidthsInterpolation('constant')
        c.CreateDisplayColorAttr([color]); return c
    for j in range(-6,7):
        line(f'GridX{j+6}', [(j,-6,.005),(j,6,.005)], (.35,.40,.45),.006)
        line(f'GridY{j+6}', [(-6,j,.005),(6,j,.005)], (.35,.40,.45),.006)
    line('SafeX',[(5.2,-6,2.05),(5.2,6,2.05)],(1.,.18,.06),.018)
    line('SafeY',[(-6,5.2,2.05),(6,5.2,2.05)],(1.,.18,.06),.018)
    distance_line = line('Distance',[(0,0,2),(6,0,2)],(1.,.8,.05),.012)
    trail = line('Trail',[(0,0,2),(0,0,2)],(.1,.8,1),.008)
    camera = UsdGeom.Camera.Define(stage, '/World/ReplayCamera')
    camera.CreateFocalLengthAttr(28)
    camera.CreateClippingRangeAttr((.05,200.))
    cam_transform = camera.AddTransformOp()
    product = rep.create.render_product(str(camera.GetPath()), tuple(cfg['resolution']))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([product])
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,18); small=ImageFont.truetype(font_path,15)
    videos=[]
    clips=cfg['clips'] if not args.preview else [cfg['clips'][4]]
    for clip in clips:
        check()
        records=[json.loads(s) for s in Path(clip['raw']['path']).read_text().splitlines()]
        initial,rows=records[0],records[1:]
        lane=clip['lane']; origin=np.array(initial['env_origins_w_m'][lane])
        path=out/(clip['name']+'.mp4')
        writer=imageio.get_writer(str(path),fps=cfg['fps'],codec='libx264',quality=8,
                                 macro_block_size=1,ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart'])
        telemetry=[];history=[]
        try:
            for i,row in enumerate(rows):
                check()
                state=row['root_state_w'][lane];pos=np.array(state[:3])-origin
                sensor=np.array(row['sensor_positions_w_m'][lane])-origin
                translate.Set(Gf.Vec3d(*pos))
                orient.Set(Gf.Quatf(state[3],Gf.Vec3f(*state[4:7])))
                if clip['case']=='W':
                    eye,target=((2.,-3.,3.8),(5.,0.,2.)) if lane==0 else ((-3.,2.,3.8),(0.,5.,2.))
                else:
                    eye=pos+np.array([-2.6,-3.4,1.9]);target=pos+np.array([.3,.25,0.])
                cam_transform.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
                d,names=sensor_origin_surface_distance(torch.from_numpy(sensor[None, :]),torch.zeros((1,3),dtype=torch.float64))
                endpoint=sensor.copy();axis={'floor':2,'x_wall':0,'y_wall':1}[names[0]]
                endpoint[axis]=0 if axis==2 else 6
                # Clamp to the same finite rectangle used by frozen evaluation.
                if axis==2:endpoint[:2]=np.clip(endpoint[:2],-42,42)
                else:
                    endpoint[1-axis]=np.clip(endpoint[1-axis],-6,6);endpoint[2]=np.clip(endpoint[2],0,12)
                distance_line.GetPointsAttr().Set([tuple(sensor),tuple(endpoint)])
                history.append(tuple(pos)); trail.GetPointsAttr().Set(history if len(history)>1 else history*2)
                trail.GetCurveVertexCountsAttr().Set([max(2,len(history))])
                # Zero delta time: renderer only. Raw pose is not integrated again.
                rep.orchestrator.step(rt_subframes=4 if i==0 else 1,delta_time=0.,pause_timeline=True)
                data=rgb.get_data()
                assert data.size and data.shape[:2]==(640,960), data.shape
                frame=Image.fromarray(data[:,:,:3].astype(np.uint8))
                draw=ImageDraw.Draw(frame);draw.rectangle((0,0,960,96),fill=(14,21,30));draw.rectangle((0,490,960,640),fill=(14,21,30))
                def scalar(v):
                    while isinstance(v,list):v=v[0]
                    return float(v)
                vel=row['actual_velocity_b'][lane];command=row['command_b'][lane]
                q=scalar(row['q'][lane]);beta=scalar(row['ics_beta'][lane])
                draw.text((16,8),f'InstinctRL R5 | {clip["case"]} | {clip["label"]} | lane {lane}',font=font,fill='white')
                draw.text((16,36),'ISAAC SIM 3D | Original evaluation trajectory replay | No new dynamics',font=small,fill='#90cdf4')
                draw.text((16,61),f'ICS clearance margin = {clip["margin_m"]:.2f} m | checkpoint {clip["checkpoint"]["sha256"][:16]}...',font=small,fill='white')
                draw.text((16,500),f't(state)={(i+1)*.1:5.1f}s   t(control)={i*.1:5.1f}s   q={q:.0f}   ICS beta={beta:.3f}',font=font,fill='white')
                draw.text((16,528),f'Actual v_body: [{vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f}] m/s',font=font,fill='#7de3ff')
                draw.text((16,555),f'Target v_body: [{command[0]:+.3f}, {command[1]:+.3f}, {command[2]:+.3f}] m/s',font=font,fill='white')
                draw.text((16,582),f'Sensor geometry distance: {float(d[0]):.3f} m ({names[0]}) | Orange: 0.80 m wall safety line',font=small,fill='#ffbb78')
                draw.text((16,611),'Fixed u512 + margin already passed. Controller improvement is NOT isolated RL gain.',font=small,fill='#ffd86b')
                writer.append_data(np.asarray(frame))
                telemetry.append(dict(frame=i,raw_step=row['step'],state_time_s=(i+1)*.1,control_time_s=i*.1,position_local_m=pos.tolist(),quaternion_wxyz=state[3:7],actual_velocity_b=vel,target_velocity_b=command,distance_m=float(d[0]),surface=names[0],q=q,ics_beta=beta))
                if i in (0,20,len(rows)-1):frame.save(out/f'{clip["name"]}_frame{i:03d}.png')
                if args.preview and i>=20:break
        finally:writer.close()
        sidecar=out/(clip['name']+'.json')
        sidecar.write_text(json.dumps(dict(clip=clip,raw_seed=initial['seed'],original_origin_w_m=origin.tolist(),frames=telemetry),indent=2)+'\n')
        videos.append(dict(**clip,video=pin(path),telemetry=pin(sidecar),frames=len(telemetry),duration_s=len(telemetry)/cfg['fps']))
        (out/'RUN_MANIFEST.json').write_text(json.dumps(dict(status='IN_PROGRESS',videos=videos),indent=2)+'\n')
        print('VIDEO_COMPLETE',clip['name'],len(telemetry),flush=True)
    manifest=dict(status='COMPLETE',mode=cfg['mode'],source_commit=args.source_commit,entrypoint=pin(__file__),inputs=pin(args.inputs),asset_closure=closure,pid=os.getpid(),started_unix=start,finished_unix=time.time(),videos=videos,physics_steps=0,policy_inference_steps=0,optimizer_updates=0,checkpoint_writes=0,pose_source='Original raw, translated by original environment origin only; no pose interpolation; rotor phases unavailable and left at asset default.',scene='Exact frozen combined_scene_triangles and TASLAB USD; presentation materials, lighting, camera, grid and annotation lines added only to replay stage.',timing=cfg['timing'])
    stage.GetRootLayer().Export(str(out/'replay_stage.usda'))
    (out/'RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rgb.detach([product.path]);product.destroy()
    signal.alarm(30)
    app.close()


if __name__=='__main__':main()
