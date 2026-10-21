"""Composition boundaries with real persistent journal, no model or job launch."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools import r6_continue_job as c
from tools.r6_recovery import Journal,file_pin,RecoveryError
from rlflow.store import sha

@pytest.fixture
def case(tmp_path,monkeypatch):
    j=Journal(tmp_path/'batch')
    with j.writer():j.initialize(json.loads((Path(__file__).parent/'fixtures/r6_budget.json').read_text()))
    def pin(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return file_pin(p)
    plan=pin('plan.json',{});exe=pin('exe.json',{});review_pin=pin('review.json',{'decision':'PASS_R6_NEXT_JOB'})
    request={'executable':exe};ident=dict(goal_id=j.read()['goal_id'],event_id='NEXT_JOB_REVIEW:parent:'+plan['sha256'],job_id='parent-job',input_sha256=sha(request),role='next-job-review');key=sha(ident)
    def register(*args,**kwargs):
        j.intent(ident,kind='model',request=request,resources={'automatic_model_sessions':1},purpose='development',free_disk_bytes=500*1024**3)
        return key
    def handoff(*args,**kwargs):
        s=j.read();s['events'][key]['receipt']=pin('receipt.json',{'last_message':review_pin});j.save(s)
    reg=Mock(side_effect=register);hand=Mock(side_effect=handoff);iss=Mock();advance=Mock(return_value={'status':'DECISION_PROPOSAL_RETURNED'})
    monkeypatch.setattr(c.next_job,'register_review',reg);monkeypatch.setattr(c,'model_handoff',hand)
    monkeypatch.setattr(c,'issue',iss);monkeypatch.setattr(c.next_job,'advance',advance)
    return SimpleNamespace(j=j,plan=plan,exe=exe,review_pin=review_pin,key=key,reg=reg,hand=hand,iss=iss,advance=advance,pin=pin)


def run(b):
    with b.j.writer():return c.continue_job(b.j,'parent',b.plan,b.plan,b.exe,object(),free_disk_bytes=500*1024**3)


def test_registered_review_crash_resumes_handoff_without_registering_again(case):
    b=case;b.hand.side_effect=RuntimeError('worker still in flight')
    with pytest.raises(RuntimeError):run(b)
    b.j=Journal(b.j.directory)
    # Fixture closure still points at same on-disk journal, as a resumed worker would.
    b.hand.side_effect=lambda *a,**kw:None
    with b.j.writer():
        s=b.j.read();s['events'][b.key]['receipt']=b.pin('receipt.json',{'last_message':b.review_pin});b.j.save(s)
    run(b);assert b.reg.call_count==1 and b.hand.call_count==2 and b.iss.call_count==1


def test_existing_edge_goes_directly_to_exact_runner_even_under_stop(case):
    b=case
    with b.j.writer():
        s=b.j.read();s['events']['edge']={'identity':{'role':c.next_job.ROLE},'request':dict(parent_key='parent',plan=b.plan,review_key=b.key,review=b.review_pin)};b.j.save(s)
    (b.j.directory/'STOP').touch();run(b)
    b.reg.assert_not_called();b.hand.assert_not_called();b.iss.assert_not_called();b.advance.assert_called_once()


def test_issuance_failure_prevents_dispatch(case):
    b=case;b.iss.side_effect=RecoveryError('review or budget rejected')
    with pytest.raises(RecoveryError):run(b)
    b.advance.assert_not_called()


def test_review_failure_prevents_issuance(case):
    b=case;b.hand.side_effect=RecoveryError('failed receipt')
    with pytest.raises(RecoveryError):run(b)
    b.iss.assert_not_called();b.advance.assert_not_called()


def test_recovery_refuses_changed_review_executable(case):
    b=case;b.hand.side_effect=RuntimeError('crash')
    with pytest.raises(RuntimeError):run(b)
    b.exe=b.pin('changed-exe.json',{'new':True})
    with pytest.raises(RecoveryError,match='EXECUTABLE_DRIFT'):run(b)
    assert b.hand.call_count==1
