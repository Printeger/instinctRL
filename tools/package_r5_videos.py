"""CPU-only MP4 composition and decoded-frame/telemetry verification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import imageio_ffmpeg


def pin(p):
    p=Path(p).resolve()
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)


def main():
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
    run=Path(args.run).resolve();out=run.parent
    manifest=json.loads((run/'RUN_MANIFEST.json').read_text())
    assert manifest['status']=='COMPLETE'
    checks=[]
    for video in manifest['videos']:
        assert pin(video['video']['path'])['sha256']==video['video']['sha256']
        side=json.loads(Path(video['telemetry']['path']).read_text());lane=video['lane']
        raw=[json.loads(x) for x in Path(video['raw']['path']).read_text().splitlines()]
        assert pin(video['raw']['path'])['sha256']==video['raw']['sha256']
        assert len(side['frames'])==len(raw)-1
        for frame,row in zip(side['frames'],raw[1:]):
            assert frame['raw_step']==row['step']
            for a,b in [('actual_velocity_b','actual_velocity_b'),('target_velocity_b','command_b')]:assert frame[a]==row[b][lane]
            assert frame['quaternion_wxyz']==row['root_state_w'][lane][3:7]
            for a,b in [('q','q'),('ics_beta','ics_beta')]:
                v=row[b][lane]
                while isinstance(v,list):v=v[0]
                assert frame[a]==v
        checks.append(decode(video['video']['path'],video['frames']))
    comparisons=[]
    for wall in ['x_wall','y_wall']:
        inputs=[run/f'W_{wall}_{tag}.mp4' for tag in ['u512_m000','u512_m015','u640_m015']]
        output=out/f'W_{wall}_comparison.mp4'
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-n','-v','error']
        for file in inputs:command+=['-i',str(file)]
        command+=['-filter_complex','[0:v][1:v][2:v]hstack=inputs=3[v]','-map','[v]','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
        subprocess.run(command,check=True)
        comparisons.append(dict(video=pin(output),inputs=[pin(f) for f in inputs],layout='Left: original u512 margin0; middle: same u512 margin0.15; right: u640 margin0.15. Synchronized original record indices.',verification=decode(output,200)))
    cfg=json.loads(Path(manifest['inputs']['path']).read_text())
    for record in cfg['sources']+cfg['assets_and_source']+[v[k] for v in cfg['clips'] for k in ['raw','checkpoint','evaluation']]:assert pin(record['path'])['sha256']==record['sha256']
    result=dict(schema='r5-video-manifest-v1',status='COMPLETE',render_type='Isaac Sim 5.1 RTX 3D rendering of original evaluated trajectories',not_a_new_physics_validation=True,run=pin(run/'RUN_MANIFEST.json'),replay_inputs=manifest['inputs'],render_source_commit=manifest['source_commit'],packaging_entrypoint=pin(__file__),videos=manifest['videos'],comparisons=comparisons,verification=dict(decoded=checks,all_frozen_inputs_unchanged=True,telemetry_exact_raw_match=True),selection=cfg['selection'],timing=cfg['timing'],attribution=cfg['attribution'],limitations=['No new dynamics or inference. Checkpoint is hash-verified provenance of recorded original policy execution.','Only representative lanes, not all evaluation trajectories.','Rotor joint phases were not logged; rotor meshes use asset default.','Replay materials, lighting, camera and safety lines are visualization additions; actor inputs unchanged.'])
    (out/'VIDEO_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE',single_videos=len(checks),comparisons=len(comparisons))))


def decode(path,expected):
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=0;shape=None
    while True:
        ok,frame=cap.read()
        if not ok:break
        count+=1;shape=list(frame.shape[:2]);assert frame.std()>5
    cap.release();assert count==expected,(str(path),count,expected);assert fps==10
    return dict(path=str(path),decoded_frames=count,fps=fps,shape=shape)


if __name__=='__main__':main()
