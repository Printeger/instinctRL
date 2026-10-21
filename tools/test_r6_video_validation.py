"""Actual synthetic MP4/trajectory validation; no Isaac rendering or science claim."""
import json
import subprocess
import sys
from pathlib import Path
import pytest
from tools import validate_r6_videos as m
from tools.test_r6_video_trajectory import trajectory
from tools.test_r6_video_presentation import mp4
from tools.prepare_r6_video_trajectories import prepare
from tools.r6_recovery import file_pin, RecoveryError

@pytest.fixture
def rendered(trajectory,tmp_path):
    def save(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return file_pin(p)
    _,_,write=trajectory;clip=write();dummy=save('dummy.json',{})
    clip.update(assignment_index=0,checkpoint=dummy,source_spec=dummy,domain='target',motion_pass=False,selection_reasons=['representative_failure'])
    clip['assignment']['model_role']='baseline'
    selected=save('selected.json',dict(schema_version='r6-video-selection/1',status='VIDEO_SOURCES_SELECTED_NOT_RENDERED',clips=[clip],fps=10,total_frames=200,physics_reintegrated=False,dispatch_authorized=False,unavailable_categories=[dict(reason='NO_SUCH_OUTCOME')]))
    prepare(selected,tmp_path/'extracted');extraction=file_pin(tmp_path/'extracted/result.json')
    clips=m.presentation.load(extraction);media=tmp_path/'media';media.mkdir()
    telemetry=save('media/telemetry.json',dict(trajectory=clips[0]['trajectory'],frames=clips[0]['frames']))
    video=mp4(media/'synthetic.mp4',200)
    snapshot=tmp_path/'snapshot';sources=[]
    for name in ('isaaclab-training/instinctlab/task/scene_h1.py','isaaclab-training/instinctlab/assets/taslab_uav/taslab_uav.usd'):
        p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('CPU scene fixture');sources.append(file_pin(p))
    renderer=snapshot/'render.py';renderer.write_text('CPU frozen renderer identity fixture')
    manifest=save('media/RUN_MANIFEST.json',dict(schema_version='r6-video-render/1',status='RENDERED_PENDING_CPU_VALIDATION',extraction=extraction,renderer=file_pin(renderer),scene_sources=sources,videos=[dict(index=0,trajectory=clips[0]['trajectory'],video=video,telemetry=telemetry)],physics_steps=0,policy_inference_steps=0))
    inventory=save('inventory.json',dict(output_inventory_complete=True,closure=dummy,spec=dummy,files=[dict(**pin,bytes=Path(pin['path']).stat().st_size) for pin in (manifest,video,telemetry)]))
    q=dict(render_manifest=manifest,extraction=extraction,renderer=file_pin(renderer),source_snapshot=str(snapshot),inventory=inventory,closure=dummy,postrun=dummy,spec=dummy)
    return save('inputs.json',q),tmp_path/'validated',save


def test_cpu_full_decode_and_frozen_identity_and_viewing_manifest(rendered):
    pin,out,_=rendered;result=m.prepare(pin,out)
    assert m.prepare(pin,out)==result and result['total_frames']==200 and not result['scientific_acceptance']
    value=json.loads(m.checked(result['manifest']).read_text())
    assert value['videos'][0]['motion_pass']is False and value['unavailable_categories']
    sample=value['visual_samples'][0]
    assert sample['frames']==[0,199] and sample['video']==value['videos'][0]['video']
    import cv2
    assert cv2.imread(str(m.checked(sample['image']))).shape==(352,960,3)
    assert value['groups'][0]['comparison']=='SINGLE_REGISTERED_TRAJECTORY'
    page=m.checked(result['viewer']).read_text();assert 'Promise.all' in page and 'currentTime=t' in page and 'synthetic.mp4' in page


def test_actual_isolated_cpu_validation_program(rendered):
    pin,out,_=rendered;root=Path(__file__).resolve().parents[1]
    code="import sys;sys.path[:0]=[sys.argv[1],sys.argv[1]+'/tools/rlflow'];from tools.validate_r6_videos import main;sys.argv=sys.argv[2:];main()"
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(root),'worker',pin['path'],'--sha256',pin['sha256'],str(out)],capture_output=True,text=True,timeout=60,env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['total_frames']==200


@pytest.mark.parametrize('fault',['inventory','renderer','scene','overlap','extra','extra_visual','modified_video'])
def test_rejects_unbound_or_changed_media(rendered,fault):
    pin,out,save=rendered;q=json.loads(m.checked(pin).read_text())
    if fault=='inventory':
        v=json.loads(m.checked(q['inventory']).read_text());v['files'].pop();q['inventory']=save('inventory2.json',v)
    elif fault=='renderer':q['renderer']=q['spec']
    elif fault=='scene':m.checked(json.loads(m.checked(q['render_manifest']).read_text())['scene_sources'][0]).write_text('drift')
    elif fault=='overlap':out=Path(pin['path']).parent
    elif fault=='extra':out.mkdir();(out/'other').touch()
    elif fault=='extra_visual':out.mkdir();(out/'visual-13.jpg').touch()
    else:
        v=json.loads(m.checked(q['render_manifest']).read_text());m.checked(v['videos'][0]['video']).write_bytes(b'corrupt')
    pin=save('inputs2.json',q)
    with pytest.raises(RecoveryError):m.prepare(pin,out)


def test_only_same_random_realization_is_grouped(monkeypatch):
    from copy import deepcopy
    base=dict(assignment_index=0,assignment=dict(model_role='baseline',evaluation_seed=17),checkpoint={},block_id='wall',lane=2,domain='target',motion_pass=False,selection_reasons=[],source_spec={},raw={})
    clips=[]
    for i,seed in enumerate((17,17,18)):
        c=deepcopy(base);c['assignment_index']=i;c['assignment']['evaluation_seed']=seed
        clips.append(dict(decoded=dict(clip=c)))
    monkeypatch.setattr(m.presentation,'load',lambda _:clips)
    rows,groups=m.viewing_rows({},dict(videos=[dict(index=i,video={},telemetry={},frames=200) for i in range(3)]))
    assert [g['videos'] for g in groups]==[[0,1],[2]]
