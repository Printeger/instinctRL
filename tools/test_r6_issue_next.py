"""Real Store approval transactions; parent review/runtime/delegation are CPU fixtures."""
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools.test_r6_submit_real_contract import bound
from tools import r6_issue_next as m
from tools.r6_recovery import file_pin,RecoveryError
from tools.r6_budget import BudgetError
from rlflow.store import Conflict,sha
from rlflow import policy


@pytest.fixture
def ready(bound,monkeypatch):
    b=bound;spec=dict(b.spec,id='R6-ISSUANCE-CPU')
    sp=b.store.root.parent/'new-spec.json';sp.write_text(json.dumps(spec))
    b.plan=dict(spec_pin=file_pin(sp),purpose='development')
    pp=b.store.root.parent/'plan.json';pp.write_text(json.dumps(b.plan));b.pp=file_pin(pp)
    b.jid='J-'+sha(dict(id=spec['id'],spec=spec))[:16];b.newspec=spec
    b.review=Mock(return_value=(b.plan,spec,b.jid));monkeypatch.setattr(m,'reviewed_plan',b.review)
    monkeypatch.setattr(m,'delegation',Mock())
    monkeypatch.setattr('tools.r6_storage_allocation.for_spec',lambda *a,**kw:None)
    b.args=dict(parent_key='parent',plan_pin=b.pp,review_key='review',review_pin=b.pp,
        master_pin=b.pp,store=b.store,free_disk_bytes=500*1024**3)
    return b


def issue(b):
    with b.journal.writer():return m.issue(b.journal,**b.args)


def test_exact_approval_replays_without_duplicate_store_writes(ready):
    b=ready;r=issue(b);state=b.store.replay();budget=b.journal.read()['budget']
    assert issue(b)==r and b.store.replay()==state and b.journal.read()['budget']==budget
    assert r['status']=='EXACT_NEXT_JOB_AUTHORIZED_NOT_SUBMITTED'
    assert b.jid not in state['jobs'];b.popen.assert_not_called()


def test_crash_after_actual_approval_recovers_existing_transaction(ready,monkeypatch):
    b=ready;original=policy.decide
    def crash(*a,**kw):original(*a,**kw);raise RuntimeError('crash after Store commit')
    monkeypatch.setattr(policy,'decide',crash)
    with pytest.raises(RuntimeError,match='Store commit'):issue(b)
    state=b.store.replay();budget=b.journal.read()['budget']
    monkeypatch.setattr(policy,'decide',Mock(side_effect=AssertionError('must not approve twice')))
    issue(b);assert state==b.store.replay() and budget==b.journal.read()['budget']


def test_stop_or_missing_condition_never_issues(ready):
    b=ready;before=b.store.replay();(b.journal.directory/'STOP').touch()
    with pytest.raises(RecoveryError,match='STOP'):issue(b)
    assert b.store.replay()==before
    (b.journal.directory/'STOP').unlink()
    b.store.transact('FIXTURE_REJECT_EVIDENCE','bad',lambda s:s['evidence']['cpu'].update(accepted=False))
    before=b.store.replay()
    with pytest.raises(RecoveryError,match='CONDITION'):issue(b)
    assert b.store.replay()==before


def test_budget_cannot_issue_unaffordable_job(ready):
    b=ready
    with b.journal.writer():
        s=b.journal.read();s['budget']['used']['live_starts']=s['budget']['caps']['live_starts'];b.journal.save(s)
    before=b.store.replay()
    with pytest.raises(BudgetError):issue(b)
    assert b.store.replay()==before


def test_review_failure_prevents_store_changes(ready):
    b=ready;b.review.side_effect=RecoveryError('review failed');before=b.store.replay()
    with pytest.raises(RecoveryError,match='review failed'):issue(b)
    assert b.store.replay()==before


def test_revoked_approval_is_never_reissued(ready):
    b=ready;r=issue(b)
    b.store.transact('FIXTURE_REVOKE','revoke',lambda s:s['approvals'][r['approval_id']].update(revoked=True))
    before=b.store.replay()
    with pytest.raises(Conflict,match='no unique exact'):issue(b)
    assert b.store.replay()==before


def test_stop_between_show_and_approve_preserves_unapproved_proposal(ready,monkeypatch):
    b=ready;original=policy.show_proposal
    def stop(*args):
        result=original(*args);(b.journal.directory/'STOP').touch();return result
    monkeypatch.setattr(policy,'show_proposal',stop);before=len(b.store.replay()['approvals'])
    with pytest.raises(RecoveryError,match='STOP'):issue(b)
    assert len(b.store.replay()['approvals'])==before
    assert any(p['content'].get('spec')==b.newspec for p in b.store.replay()['proposals'].values())


def test_changed_plan_cannot_rebind_same_parent(ready):
    b=ready;issue(b);before=b.store.replay()
    path=Path(b.pp['path']).with_name('different-plan.json');path.write_text('{"different":true}')
    b.args['plan_pin']=file_pin(path)
    with pytest.raises(RecoveryError,match='PARENT_ALREADY_BOUND'):issue(b)
    assert b.store.replay()==before
