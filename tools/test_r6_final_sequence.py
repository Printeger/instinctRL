"""Real sequence Journal; issuance, physical jobs and collection are explicit substitutes."""
from unittest.mock import Mock
import pytest
from tools import r6_final_sequence as s
from tools.test_r6_final_plans_review import review_ready, batch, prepared, ready, case, run as review_run, FREE
from tools.r6_recovery import RecoveryError

@pytest.fixture
def sequence(review_ready,monkeypatch):
    b=review_ready;b.reviewed=review_run(b);b.calls=[];b.job_records={};b.fail_at=None
    monkeypatch.setattr(s.issuer,'issue',Mock())
    def run(journal,**kw):
        i=next(i for i,r in enumerate(b.aggregate['assignments']) if r['materialization']['artifacts']['spec']==kw['spec_pin'])
        if i==b.fail_at:raise RuntimeError('interrupted between assignments')
        b.calls.append(i);key='fixture-job-'+str(i)
        b.job_records[key]={'identity':{'job_id':b.aggregate['assignments'][i]['materialization']['job_id']}}
        return dict(status='FINAL_JOB_ANALYSIS_RETURNED',event_key=key)
    monkeypatch.setattr(s.runner,'run',run)
    monkeypatch.setattr(b.j,'canonical_job',lambda k:b.job_records[k])
    b.collect=Mock(return_value={'status':'fixture-complete','scientific_acceptance':False})
    monkeypatch.setattr(s.collection,'collect',b.collect)
    return b


def run(b):
    with b.j.writer():return s.advance(b.j,b.aggregate_pin,b.reviewed['event_key'],b.reviewed['review'],{},b.store,free_disk_bytes=FREE)


def test_sequence_runs_all_in_order_and_does_not_repeat_completed(sequence):
    b=sequence;out=run(b);budget=b.j.read()['budget'];assert run(b)==out
    assert b.calls==list(range(len(b.aggregate['assignments'])))
    assert b.j.read()['budget']==budget and not out['scientific_acceptance']
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'


def test_restart_preserves_completed_prefix(sequence):
    b=sequence;b.fail_at=1
    with pytest.raises(RuntimeError):run(b)
    assert b.calls==[0]
    b.fail_at=None;run(b)
    assert b.calls==list(range(len(b.aggregate['assignments'])))


def test_stop_prevents_next_assignment_after_interruption(sequence):
    b=sequence;b.fail_at=1
    with pytest.raises(RuntimeError):run(b)
    (b.j.directory/'STOP').touch();b.fail_at=None
    # Real issuer owns the pre-issuance guard; preserve it in this substitute.
    s.issuer.issue.side_effect=lambda *a,**kw:s.profile.guard(b.j)
    with pytest.raises(RecoveryError):run(b)
    assert b.calls==[0]
