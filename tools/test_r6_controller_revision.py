"""Real Journal/fake-CLI revision receipts; commit identity mocked except Git test."""
import copy,json,sys,subprocess
from pathlib import Path
import pytest
from tools import r6_controller_revision as r,r6_controller as c
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import file_pin,RecoveryError
from tools.r6_batch import model_handoff
from rlflow.store import sha
FREE=500*1024**3
SCRIPT='''import json,sys,hashlib
from pathlib import Path
p=json.loads(sys.stdin.read().splitlines()[-1]);reject=Path(__file__).with_suffix('.reject').exists()
with Path(__file__).with_suffix('.count').open('a') as f:f.write('call\\n')
r=dict(decision='MUST_FIX' if reject else 'PASS_R6_CONTROLLER_REVISION',summary='CPU substitute, no real review',evidence_pins=[p['proposal']],changed_files=[],must_fix=['fixture reject'] if reject else [],reviewer='r6-independent-controller-revision',packet_sha256=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),remaining_budget={'automatic_sessions_remaining':p['caps']['automatic_model_sessions']-p['budget']['automatic_model_sessions']-1})
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(r)}}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))
'''

@pytest.fixture
def revision(case,monkeypatch):
    b=case;b.before={'control_sha256':'a'*64,'schemas_sha256':'a'*64};b.after={'control_sha256':'b'*64,'schemas_sha256':'b'*64}
    monkeypatch.setattr(c,'implementation',lambda:b.before)
    b.initial=b.write('initial.json',dict(plan_pin=b.context,master_pin=b.context,review_executable=file_pin(b.builder)))
    b.request=b.write('controller.json',dict(initial_request=b.initial,profile_context=b.context))
    with b.j.writer():b.controller,_=c.register(b.j,b.request,b.store,free_disk_bytes=FREE)
    b.old=copy.deepcopy(b.j.read()['events'][b.controller]);monkeypatch.setattr(c,'implementation',lambda:b.after)
    b.proposal=b.write('revision.json',dict(from_commit='a'*40,to_commit='b'*40,reason='Repair fixture controller',evidence_pins=[b.context]))
    monkeypatch.setattr(r,'committed_implementation',lambda commit:b.before if commit=='a'*40 else b.after)
    monkeypatch.setattr(r.snapshot,'git',lambda root,*args:('main' if args[0]=='symbolic-ref' else 'a'*40 if args[0]=='merge-base' else 'b'*40).encode()+b'\n')
    b.reviewer=b.root/'revision-reviewer';b.reviewer.write_text('#!'+sys.executable+'\n'+SCRIPT);b.reviewer.chmod(0o700)
    monkeypatch.setattr(r,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
    return b


def run(b):
    with b.j.writer():return r.prepare(b.j,b.controller,b.proposal,b.store,executable=file_pin(b.reviewer),free_disk_bytes=FREE)


def test_review_adopts_source_without_changing_identity_cursor_or_store(revision):
    b=revision;store=b.store.replay();out=run(b);s=b.j.read();e=s['events'][b.controller]
    for field in ('identity','request','cursor'):assert e[field]==b.old[field]
    assert e['implementation_revisions']==[out['event_key']] and b.store.replay()==store
    budget=s['budget'];assert run(b)==out and b.j.read()['budget']==budget
    with b.j.writer():assert c.register(b.j,b.request,b.store,free_disk_bytes=FREE)[0]==b.controller
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'


def test_rejection_does_not_adopt_or_call_again(revision):
    b=revision;b.reviewer.with_suffix('.reject').touch()
    for _ in range(2):
        with pytest.raises(RecoveryError,match='REJECTED'):run(b)
    assert not b.j.read()['events'][b.controller].get('implementation_revisions')
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'


def test_uncommitted_source_rejected_before_model(revision,monkeypatch):
    b=revision;monkeypatch.setattr(r,'committed_implementation',lambda commit:b.before)
    with pytest.raises(RecoveryError,match='UNCOMMITTED_SOURCE'):run(b)
    assert not b.reviewer.with_suffix('.count').exists()


def test_stop_prevents_revision(revision):
    b=revision;(b.j.directory/'STOP').touch()
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert not b.reviewer.with_suffix('.count').exists()


def test_post_review_cursor_drift_preserves_original_revision_failure(revision,monkeypatch):
    b=revision;original=r.model_handoff
    def mutate(j,k):
        original(j,k);s=j.read();s['events'][b.controller]['cursor']['phase']='OTHER';j.save(s)
    monkeypatch.setattr(r,'model_handoff',mutate)
    with pytest.raises(RecoveryError,match='POST_REVIEW_DRIFT'):run(b)
    assert not b.j.read()['events'][b.controller].get('implementation_revisions')
    with pytest.raises(RecoveryError,match='STATE_DRIFT'):run(b)
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'


def test_intent_to_packet_save_crash_recovers_exact_hashed_prompt(revision,monkeypatch):
    b=revision;original=b.j.intent;crashed=False
    def crash(*args,**kwargs):
        nonlocal crashed
        result=original(*args,**kwargs)
        if not crashed and kwargs['kind']=='model':
            crashed=True;raise RuntimeError('fixture crash')
        return result
    monkeypatch.setattr(b.j,'intent',crash)
    with pytest.raises(RuntimeError):run(b)
    budget=b.j.read()['budget'];out=run(b)
    assert out['status']=='CONTROLLER_REVISION_APPLIED' and b.j.read()['budget']==budget
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'


def test_tampered_packet_cannot_authorize_resume(revision):
    b=revision;out=run(b)
    with b.j.writer():
        s=b.j.read();s['events'][out['event_key']]['revision_packet']={'tampered':True};b.j.save(s)
        with pytest.raises(RecoveryError,match='PACKET_DRIFT'):c.register(b.j,b.request,b.store,free_disk_bytes=FREE)


def test_commit_identity_uses_git_blobs_not_changed_worktree(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir()
    def git(*args):return subprocess.check_output(['git','-C',str(root),*args]).decode().strip()
    git('init','-q','-b','main');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
    paths=['tools/control.py','training_workloads/unified_h1/config.py','docs/experiments/r6-robustness-auto/prompts/source-preparation.schema.json','docs/experiments/r6-robustness-auto/prompts/other.schema.json']
    for p in paths:q=root/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_text('{}')
    git('add','.');git('commit','-qm','fixture');commit=git('rev-parse','HEAD');monkeypatch.setattr(r,'ROOT',root)
    value=r.committed_implementation(commit);(root/paths[0]).write_text('modified')
    assert r.committed_implementation(commit)==value
    git('add','.');git('commit','-qm','changed');assert r.committed_implementation(git('rev-parse','HEAD'))!=value


def test_receipt_recorded_before_adoption_crash_reuses_review(revision,monkeypatch):
    b=revision;original=r.model_handoff;once=True
    def crash(j,k):
        nonlocal once
        result=original(j,k)
        if once:once=False;raise RuntimeError('fixture post-receipt crash')
        return result
    monkeypatch.setattr(r,'model_handoff',crash)
    with pytest.raises(RuntimeError):run(b)
    assert not b.j.read()['events'][b.controller].get('implementation_revisions')
    budget=b.j.read()['budget'];run(b)
    assert b.j.read()['budget']==budget and b.reviewer.with_suffix('.count').read_text()=='call\n'


def test_later_unreviewed_source_still_rejected(revision,monkeypatch):
    b=revision;run(b);monkeypatch.setattr(c,'implementation',lambda:{'unreviewed':True})
    with b.j.writer():
        with pytest.raises(RecoveryError,match='INPUT_DRIFT'):c.register(b.j,b.request,b.store,free_disk_bytes=FREE)


def test_stale_budget_before_dispatch_rejected(revision,monkeypatch):
    b=revision;original=b.j.intent
    def changed(*args,**kwargs):
        result=original(*args,**kwargs)
        if kwargs['kind']=='model':
            s=b.j.read();s['budget']['used']['automatic_model_sessions']+=1;b.j.save(s)
        return result
    monkeypatch.setattr(b.j,'intent',changed)
    with pytest.raises(RecoveryError,match='STALE_BUDGET'):run(b)
    assert not b.reviewer.with_suffix('.count').exists()
