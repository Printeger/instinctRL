"""Bounded CPU extraction of selected R6 replay trajectories; no rendering.

Controller must reserve output before calling this worker. Existing exact files
are reusable; differing outputs or aliased destinations are never overwritten.
"""
import argparse
import json
from pathlib import Path
from rlflow.store import canonical
from tools.r6_recovery import checked, file_pin, RecoveryError
from tools.r6_decision_packet import persist_exact
from tools import r6_video_trajectory as trajectory

CAP=16*1024**2


def prepare(selection_pin,output):
    selection=json.loads(checked(selection_pin).read_text());output=Path(output)
    if (selection.get('schema_version')!='r6-video-selection/1' or selection.get('status')!='VIDEO_SOURCES_SELECTED_NOT_RENDERED'
        or selection.get('fps')!=10 or selection.get('physics_reintegrated')is not False
        or selection.get('dispatch_authorized')is not False or not 1<=len(selection.get('clips',[]))<=14):
        raise RecoveryError('VIDEO_EXTRACTION_SELECTION')
    if not output.is_absolute() or output.resolve()!=output or output.is_symlink():raise RecoveryError('VIDEO_EXTRACTION_PATH')
    inputs=[selection_pin]+[p for c in selection['clips'] for p in c['raw'].values()]
    for p in inputs:
        path=checked(p)
        if output==path or output in path.parents:raise RecoveryError('VIDEO_EXTRACTION_INPUT_OVERLAP')
    allowed={f'clip-{i:02d}.json' for i in range(len(selection['clips']))}|{'result.json'}
    if output.exists() and any(p.name not in allowed for p in output.iterdir()):raise RecoveryError('VIDEO_EXTRACTION_UNCLAIMED_OUTPUT')
    rows=[];frames=0
    def save(name,value):
        data=canonical(value)+b'\n'
        if output.exists() and any(p.is_symlink() or not p.is_file() for p in output.iterdir()):raise RecoveryError('VIDEO_EXTRACTION_OUTPUT_ENTRY')
        prior=sum(p.stat().st_size for p in output.iterdir() if p.name!=name) if output.exists() else 0
        if prior+len(data)>CAP:raise RecoveryError('VIDEO_EXTRACTION_CAP')
        path=output/name;persist_exact(path,data);return file_pin(path)
    for i,clip in enumerate(selection['clips']):
        decoded=trajectory.decode(clip);frames+=len(decoded['frames'])
        rows.append(dict(index=i,trajectory=save(f'clip-{i:02d}.json',decoded),frame_count=len(decoded['frames'])))
    if frames!=selection['total_frames']:raise RecoveryError('VIDEO_EXTRACTION_TOTAL_FRAMES')
    for p in inputs:checked(p)
    result=dict(status='RECORDED_TRAJECTORIES_EXTRACTED_NOT_RENDERED',selection=selection_pin,trajectories=rows,
        total_frames=frames,fps=10,extractor=file_pin(__file__),decoder=file_pin(trajectory.__file__),
        physics_reintegrated=False,scientific_acceptance=False,dispatch_authorized=False)
    save('result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('selection',type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('output',type=Path);args=p.parse_args()
    result=prepare(dict(path=str(args.selection.resolve()),sha256=args.sha256),args.output.absolute())
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
