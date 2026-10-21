"""Real recorded NEEDS_REPAIR decision is preserved; no new profile/live action."""
import json,copy
import pytest
from tools import r6_repair_input as r,r6_controller as c
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import RecoveryError,file_pin,Journal
FREE=500*1024**3

@pytest.fixture
def repair(case):
    b=case;b.initial=b.write('repair-initial.json',dict(plan_pin=b.context,master_pin=b.context,review_executable=file_pin(b.builder)))
    b.request=b.write('repair-controller.json',dict(initial_request=b.initial,profile_context=b.context))
    with b.j.writer():
        b.controller,_=c.register(b.j,b.request,b.store,free_disk_bytes=FREE)
        s=b.j.read();s['events'][b.controller]['cursor'].update(phase='REPAIR_AND_REVIEW',job_key=b.key,parent_key=b.parent)
        b.original=copy.deepcopy(s['events'][b.key]);b.parent_original=copy.deepcopy(s['events'][b.parent]);b.j.save(s)
    return b


def run(b):
    with b.j.writer():return r.prepare(b.j,b.controller,b.store,free_disk_bytes=FREE)


def test_input_preserves_actual_failed_job_and_review_and_charges_once(repair):
    b=repair;store=b.store.replay();out=run(b);budget=b.j.read()['budget'];value=json.loads(r.checked(out['repair_input']).read_text())
    assert value['description']['decision']['decision']['decision']=='NEEDS_REPAIR'
    assert value['description']['closed_job_evidence']['closure']==b.original['closure']
    assert value['dispatch_authorized'] is False and value['goal_complete'] is False
    b.j=Journal(b.j.directory);assert run(b)==out and b.j.read()['budget']==budget
    assert b.store.replay()==store and b.j.read()['events'][b.key]==b.original
    assert b.j.read()['events'][b.parent]==b.parent_original
    assert b.j.read()['events'][out['event_key']]['purpose']=='repair'
    assert not b.builder.with_suffix('.count').exists()


def test_before_output_crash_recovers_reserved_input(repair,monkeypatch):
    b=repair;original=r.persist_exact
    monkeypatch.setattr(r,'persist_exact',lambda *a:(_ for _ in ()).throw(RuntimeError('fixture crash')))
    with pytest.raises(RuntimeError):run(b)
    budget=b.j.read()['budget'];monkeypatch.setattr(r,'persist_exact',original);run(b)
    assert b.j.read()['budget']==budget


def test_stop_does_not_create_repair_or_worker(repair):
    b=repair;(b.j.directory/'STOP').touch();before=b.j.read()
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert b.j.read()==before


def test_nonrepair_decision_cannot_create_repair(repair,monkeypatch):
    b=repair;monkeypatch.setattr(r,'decision_record',lambda *a:dict(decision=dict(decision='PROPOSE_CANDIDATE',must_fix=[])))
    with pytest.raises(RecoveryError,match='DECISION'):run(b)


def test_tampered_failed_receipt_blocks_repair(repair):
    b=repair;p=r.checked(b.parent_original['receipt']);p.write_text('{}')
    with pytest.raises(RecoveryError,match='PIN_DRIFT'):run(b)
