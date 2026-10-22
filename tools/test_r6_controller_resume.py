"""Accepted fake CLI review plus real fresh Python child; controller work is stubbed."""
import copy,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from rlflow.jobs import identity
from tools import r6_controller_resume as resume,r6_controller as controller
from tools.test_r6_controller_revision import revision,run as review,FREE
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import RecoveryError


@pytest.fixture
def ready(revision):
    b=revision;b.accepted=review(b)
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.15)'])
    b.dead=identity(child.pid);child.wait(timeout=5)
    with b.j.writer():
        state=b.j.read();state['events'][b.controller]['controller_process']=b.dead;b.j.save(state)
    return b


def run(b):
    with b.j.writer():
        return resume.run(b.j,b.controller,b.accepted['event_key'],b.store,free_disk_bytes=FREE)


def test_accepted_revision_resumes_original_inputs_once(ready,monkeypatch):
    b=ready;before=copy.deepcopy(b.j.read()['events'][b.controller]);store=b.store.replay()
    worker=Mock(return_value={'status':'CPU_BOUNDARY'});monkeypatch.setattr(controller,'run',worker)
    out=run(b);budget=b.j.read()['budget'];assert run(b)==out and b.j.read()['budget']==budget
    assert worker.call_count==1 and worker.call_args.args[1]==before['request']['inputs']
    assert b.j.read()['events'][b.controller]==before and b.store.replay()==store
    assert out['process']!=b.dead and out['controller_result']['status']=='CPU_BOUNDARY'
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'


@pytest.mark.parametrize('fault',['live_process','same_process','cursor','store','code','commit','stop','unaccepted'])
def test_unreviewed_or_unsafe_resume_refuses_before_claim(ready,monkeypatch,fault):
    b=ready
    if fault in ('live_process','same_process','cursor','unaccepted'):
        with b.j.writer():
            state=b.j.read();owner=state['events'][b.controller]
            if fault=='same_process':owner['controller_process']=identity(os.getpid())
            if fault=='live_process':owner['controller_process']=identity(os.getppid())
            if fault=='cursor':owner['cursor']['phase']='OTHER'
            if fault=='unaccepted':owner['implementation_revisions']=[]
            b.j.save(state)
    if fault=='store':b.store=SimpleNamespace(root=b.store.root,replay=lambda:{'changed':True})
    if fault=='code':monkeypatch.setattr(controller,'implementation',lambda:{'unknown':True})
    if fault=='commit':monkeypatch.setattr(resume.revision,'committed_implementation',lambda c:{'wrong':True})
    if fault=='stop':(b.j.directory/'STOP').touch()
    worker=Mock(side_effect=AssertionError('no resume'));monkeypatch.setattr(controller,'run',worker)
    before=b.j.read()
    with pytest.raises(RecoveryError):run(b)
    assert b.j.read()==before;worker.assert_not_called()


def test_crash_after_reservation_before_claim_can_recover(ready,monkeypatch):
    b=ready;original=b.j.intent
    def crash(*a,**kw):original(*a,**kw);raise RuntimeError('reserved only')
    monkeypatch.setattr(b.j,'intent',crash)
    with pytest.raises(RuntimeError,match='reserved only'):run(b)
    budget=b.j.read()['budget'];monkeypatch.setattr(b.j,'intent',original)
    worker=Mock(return_value={'status':'CPU_BOUNDARY'});monkeypatch.setattr(controller,'run',worker)
    run(b);assert b.j.read()['budget']==budget and worker.call_count==1


def test_failed_controller_keeps_claim_and_never_blindly_repeats(ready,monkeypatch):
    b=ready;worker=Mock(side_effect=RuntimeError('controller fixture failure'));monkeypatch.setattr(controller,'run',worker)
    cursor=copy.deepcopy(b.j.read()['events'][b.controller]['cursor'])
    with pytest.raises(RuntimeError,match='controller fixture failure'):run(b)
    budget=b.j.read()['budget']
    with pytest.raises(RecoveryError,match='CLAIMED_NO_BLIND_RETRY'):run(b)
    assert worker.call_count==1 and b.j.read()['budget']==budget
    assert b.j.read()['events'][b.controller]['cursor']==cursor
    failed=[e for e in b.j.read()['events'].values() if e['identity'].get('role')=='controller-resume']
    assert failed[0]['status']=='CONTROLLER_RESUME_FAILED' and failed[0]['resume_error']['type']=='RuntimeError'


def test_real_child_loads_reviewed_state_and_registers_same_controller(ready):
    b=ready
    config=b.write('resume-child.json',dict(directory=str(b.j.directory),controller=b.controller,
        revision=b.accepted['event_key'],store_root=str(b.store.root),store=b.store.replay(),after=b.after))
    script='''import json,sys
from pathlib import Path
from types import SimpleNamespace
from tools import r6_controller as c,r6_controller_revision as r,r6_controller_resume as resume
from tools.r6_recovery import Journal
v=json.loads(Path(sys.argv[1]).read_text());j=Journal(v['directory'])
# Exact committed-source/runtime identities and the scientific work are fixtures.
# Original independent review receipts, worker cleanup and fresh PID are real.
c.implementation=lambda:v['after']
r.committed_implementation=lambda commit:v['after']
r.snapshot.git=lambda root,*args:('b'*40+'\\n').encode()
def boundary(j,pin,store,**kw):
 key,_=c.register(j,pin,store,**kw)
 return dict(status='CPU_FRESH_CHILD_BOUNDARY',controller_key=key)
c.run=boundary
with j.writer():out=resume.run(j,v['controller'],v['revision'],SimpleNamespace(root=Path(v['store_root']),replay=lambda:v['store']),free_disk_bytes=500*1024**3)
print(json.dumps(out))
'''
    process=subprocess.run([sys.executable,'-B','-c',script,config['path']],cwd=controller.ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.:tools:tools/rlflow'),capture_output=True,text=True,timeout=30)
    assert process.returncode==0,process.stderr
    out=json.loads(process.stdout);owner=b.j.read()['events'][b.controller]
    assert out['process']['pid']!=os.getpid() and out['process']!=b.dead
    assert owner['controller_process']==out['process']
    assert owner['cursor']==b.old['cursor'] and owner['identity']==b.old['identity']
    assert out['controller_result']['controller_key']==b.controller


@pytest.mark.parametrize('review_only',[False,True])
def test_revision_cli_releases_writer_before_launching_resume(ready,monkeypatch,review_only):
    from tools import r6_controller_revision as r
    from tools.r6_recovery import Journal
    b=ready;database=b.store.root/'runtime/research.db';database.parent.mkdir(exist_ok=True);database.touch()
    monkeypatch.setattr('rlflow.store.Store',lambda root:b.store)
    monkeypatch.setattr(r,'prepare',Mock(return_value=b.accepted))
    calls=[]
    def launched(argv,**kwargs):
        # A competing real writer acquisition must succeed before child launch.
        with Journal(b.j.directory).writer():pass
        calls.append((argv,kwargs));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',launched)
    argv=['revision',str(b.j.directory),b.controller,b.proposal['path'],'--proposal-sha256',b.proposal['sha256'],'--store',str(b.store.root)]
    if review_only:argv.append('--review-only')
    monkeypatch.setattr(sys,'argv',argv)
    if review_only:r.main();assert calls==[]
    else:
        with pytest.raises(SystemExit) as exit:r.main()
        assert exit.value.code==0 and len(calls)==1
        assert calls[0][0]==[sys.executable,'-B','-m','tools.r6_controller_resume',str(b.j.directory),b.controller,b.accepted['event_key'],'--store',str(b.store.root)]
