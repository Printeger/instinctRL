"""Actual detached local worker/receipt/cleanup on a fake CPU extraction program."""
import json,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from rlflow.jobs import identity
from rlflow.store import sha
from tools import r6_video_extraction as m
from tools.r6_recovery import Journal,file_pin,RecoveryError

FREE=500*1024**3
SCRIPT='''import json,sys,time,hashlib
from pathlib import Path
root=Path(__file__).parent
with (root/'calls').open('a') as f:f.write('call\\n')
{body}
selection=Path(sys.argv[1]);q=json.loads(selection.read_text());out=Path(sys.argv[-1]);out.mkdir(parents=True)
def pin(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
rows=[]
for i,c in enumerate(q['clips']):
 p=out/f'clip-{{i:02d}}.json';p.write_text(json.dumps(dict(clip=c,frames=[{{}}]*c['frame_count'],physics_reintegrated=False)))
 rows.append(dict(index=i,trajectory=pin(p),frame_count=c['frame_count']))
r=dict(status='RECORDED_TRAJECTORIES_EXTRACTED_NOT_RENDERED',selection=pin(selection),trajectories=rows,total_frames=q['total_frames'],fps=10,physics_reintegrated=False,scientific_acceptance=False,dispatch_authorized=False)
(out/'result.json').write_text(json.dumps(r))
'''

@pytest.fixture
def worker(tmp_path,monkeypatch):
    j=Journal(tmp_path/'batch');budget=json.loads((m.ROOT/'tools/fixtures/r6_budget.json').read_text())
    store=SimpleNamespace(root=tmp_path/'store')
    with j.writer():
        j.initialize(budget);q=dict(store_root=str(store.root));event=dict(goal_id=budget['goal_id'],event_id='CONTROLLER',job_id='CONTROLLER',input_sha256=sha(q),role='r6-controller')
        controller,_=j.intent(event,kind='cpu',request=q,resources={'output_actual_plus_unreleased_reservations_bytes':1},purpose='development',free_disk_bytes=FREE)
        s=j.read();s['events'][controller]['cursor']={'final_collection':{'status':'FINAL_BATCH_COLLECTED_REQUIRES_INDEPENDENT_AUDIT','collection':{'batch':{}},'completed_jobs':[]}};j.save(s)
    raw=tmp_path/'raw';raw.write_text('fixture')
    selected=dict(schema_version='r6-video-selection/1',status='VIDEO_SOURCES_SELECTED_NOT_RENDERED',clips=[{'frame_count':200,'raw':{'fixture':file_pin(raw)}}],total_frames=200,fps=10,physics_reintegrated=False,dispatch_authorized=False)
    monkeypatch.setattr(m.selection,'prepare',Mock(return_value=selected))
    b=SimpleNamespace(j=j,store=store,controller=controller,program=tmp_path/'fake.py',root=tmp_path,proc=None)
    yield b
    if b.proc is not None and b.proc.poll() is None:
        with b.j.writer():m.cancel_bounded(b.j,b.key,'TEST_CLEANUP',b.proc,directory_kind='cpu-video',status_prefix='CPU_VIDEO')
        b.proc.wait(timeout=5)


def register(b,body='time.sleep(.03)'):
    b.program.write_text(SCRIPT.format(body=body))
    with b.j.writer():b.key=m.register(b.j,b.controller,b.store,free_disk_bytes=FREE,python=file_pin(sys.executable),program=file_pin(b.program),timeout_seconds=10)


def handoff(b):
    with b.j.writer():return m.handoff(b.j,b.key,b.store,poll_seconds=.01)


def calls(b):return (b.root/'calls').read_text().splitlines() if (b.root/'calls').exists() else []


def claim(b):
    with b.j.writer():
        s=b.j.read();s['events'][b.key]['dispatch_claim']=dict(controller=identity(os.getpid()),worker=file_pin(m.__file__),deadline_monotonic=time.monotonic()+40,deadline_wall=time.time()+40);b.j.save(s)


def test_charge_once_and_durable_receipt_replay(worker):
    b=worker;register(b);before=b.j.read()['budget'];out=handoff(b)
    assert out['status']=='VIDEO_TRAJECTORIES_READY_NOT_RENDERED' and not out['dispatch_authorized']
    assert handoff(b)==out and calls(b)==['call'] and b.j.read()['budget']==before
    assert b.j.read()['events'][b.key]['purpose']=='confirmation_delivery'
    assert not any(e['kind']=='model' for e in b.j.read()['events'].values())


def test_restart_reconnects_exact_live_worker(worker):
    b=worker;register(b,'time.sleep(.5)');claim(b)
    b.proc=subprocess.Popen([sys.executable,'-B','-m','tools.r6_video_extraction',str(b.j.directory),b.key],cwd=m.ROOT,env=m.environment(),start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    deadline=time.time()+5
    while not calls(b):
        assert b.proc.poll() is None and time.time()<deadline;time.sleep(.01)
    b.j=Journal(b.j.directory);handoff(b);b.proc.wait(timeout=5);assert calls(b)==['call']


@pytest.mark.parametrize('fault',['claim','partial','stop','failure','raw','reservation'])
def test_ambiguous_failed_or_changed_event_never_retries(worker,fault):
    b=worker;register(b,'raise SystemExit(9)' if fault=='failure' else 'pass')
    if fault in ('claim','partial'):
        claim(b)
        if fault=='partial':
            p=b.j.directory/'cpu-video'/b.key;p.mkdir(parents=True);(p/'receipt.json').write_text('{')
    elif fault=='stop':(b.j.directory/'STOP').touch()
    elif fault=='raw':(b.root/'raw').write_text('changed')
    elif fault=='reservation':
        with b.j.writer():
            s=b.j.read();s['budget']['reservations']=[r for r in s['budget']['reservations'] if r['identity']['role']!=m.ROLE];b.j.save(s)
    for _ in range(2):
        with pytest.raises(RecoveryError):handoff(b)
    assert calls(b)==(['call'] if fault=='failure' else [])


def test_actual_tensor_extractor_through_registered_worker(worker):
    b=worker;directory=b.root/'real-recorded-fixture';directory.mkdir()
    code="""import json,sys
from pathlib import Path
from tools.test_r6_video_trajectory import trajectory
records,traces,write=trajectory.__wrapped__(Path(sys.argv[1]));clip=write()
print(json.dumps(dict(schema_version='r6-video-selection/1',status='VIDEO_SOURCES_SELECTED_NOT_RENDERED',clips=[clip],total_frames=200,fps=10,physics_reintegrated=False,dispatch_authorized=False)))
"""
    result=subprocess.run([str(m.PYTHON),'-B','-c',code,str(directory)],cwd=m.ROOT,env=m.environment(),capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    m.selection.prepare.return_value=json.loads(result.stdout)
    with b.j.writer():b.key=m.register(b.j,b.controller,b.store,free_disk_bytes=FREE,timeout_seconds=60)
    out=handoff(b);data=json.loads(m.checked(out['result']).read_text())
    assert data['total_frames']==200 and data['extractor']==file_pin(m.extractor.__file__)
    trajectory=json.loads(m.checked(data['trajectories'][0]['trajectory']).read_text())
    assert trajectory['frames'][-1]['reset_after_state'] and not trajectory['physics_reintegrated']
    assert handoff(b)==out


def test_stop_cancels_actual_video_extraction_worker(worker):
    from rlflow.jobs import owns
    b=worker;register(b,'time.sleep(30)');claim(b)
    b.proc=subprocess.Popen([sys.executable,'-B','-m','tools.r6_video_extraction',str(b.j.directory),b.key],cwd=m.ROOT,env=m.environment(),start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    deadline=time.time()+5
    while not calls(b):
        assert b.proc.poll()is None and time.time()<deadline;time.sleep(.01)
    (b.j.directory/'STOP').touch()
    with pytest.raises(RecoveryError):handoff(b)
    b.proc.wait(timeout=5)
    event=b.j.read()['events'][b.key]
    assert event['controller_cancel']['status']=='NO_KNOWN_OWNED_SURVIVORS'
    assert not any(owns(p) for p in event['controller_cancel']['known'])
    assert calls(b)==['call']
