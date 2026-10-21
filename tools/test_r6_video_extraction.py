"""Complete CPU batch worker on real compressed audit fixtures; no rendering."""
import json,subprocess,sys
from pathlib import Path
import pytest
from tools.test_r6_video_trajectory import trajectory
from tools import prepare_r6_video_trajectories as m
from tools.r6_recovery import file_pin,RecoveryError

@pytest.fixture
def selected(trajectory,tmp_path):
    _,_,write=trajectory;clip=write()
    value=dict(schema_version='r6-video-selection/1',status='VIDEO_SOURCES_SELECTED_NOT_RENDERED',fps=10,
        physics_reintegrated=False,dispatch_authorized=False,clips=[clip],total_frames=200)
    p=tmp_path/'selection.json';p.write_text(json.dumps(value))
    return file_pin(p),tmp_path/'extracted'


def test_exact_batch_replays_without_overwriting_originals(selected):
    pin,out=selected;before=m.checked(pin).read_bytes();result=m.prepare(pin,out)
    assert m.prepare(pin,out)==result and m.checked(pin).read_bytes()==before
    decoded=json.loads(m.checked(result['trajectories'][0]['trajectory']).read_text())
    assert decoded['frames'][-1]['reset_after_state']
    assert not result['physics_reintegrated'] and not result['dispatch_authorized']


def test_actual_isolated_cpu_worker_entry(selected):
    pin,out=selected;root=Path(__file__).resolve().parents[1]
    program="import sys;sys.path[:0]=[sys.argv[1],sys.argv[1]+'/tools/rlflow'];from tools.prepare_r6_video_trajectories import main;sys.argv=sys.argv[2:];main()"
    result=subprocess.run([sys.executable,'-I','-B','-c',program,str(root),'worker',pin['path'],'--sha256',pin['sha256'],str(out)],
        env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout);assert value['total_frames']==200 and len(value['trajectories'])==1


@pytest.mark.parametrize('fault',['overlap','drift','size','extra','symlink','partial_count'])
def test_worker_rejects_unsafe_or_changed_outputs(selected,monkeypatch,fault):
    pin,out=selected
    if fault=='overlap':out=Path(pin['path']).parent
    elif fault=='drift':
        value=m.prepare(pin,out);m.checked(value['trajectories'][0]['trajectory']).write_text('{}')
    elif fault=='size':monkeypatch.setattr(m,'CAP',10)
    elif fault=='extra':out.mkdir();(out/'unclaimed').write_text('keep')
    elif fault=='symlink':out.symlink_to(Path(pin['path']).parent,target_is_directory=True)
    else:
        path=m.checked(pin);v=json.loads(path.read_text());v['total_frames']=199;path.write_text(json.dumps(v));pin=file_pin(path)
    with pytest.raises(RecoveryError):m.prepare(pin,out)
