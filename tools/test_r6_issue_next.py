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
from tools.r6_storage_allocation import for_spec as real_for_spec


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


@pytest.mark.parametrize('dimension', ['live_starts','training_transitions','training_checkpoint_writes'])
def test_budget_cannot_issue_unaffordable_job(ready,dimension):
    b=ready
    if dimension.startswith('training_') and b.newspec['h1']['mode']!='research-r6-training':
        return
    with b.journal.writer():
        s=b.journal.read();s['budget']['used'][dimension]=s['budget']['caps'][dimension];b.journal.save(s)
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


def test_real_prepaid_storage_issuance_then_next_job_registration(ready,monkeypatch,tmp_path):
    from tools.test_r6_storage_allocation import case as storage_case,register,spec as storage_spec
    from tools import r6_storage_allocation as allocation,r6_next_job
    b=ready;storage=storage_case.__wrapped__(tmp_path/'storage',monkeypatch)
    owner=register(storage);value,pin=storage_spec(storage,monkeypatch)
    params=json.loads(Path(value['pins']['authorization']['path']).read_text())['parameters']
    b.observed['authorization']['parameters'].update(params)
    authpath=tmp_path/'merged-auth.json';authpath.write_text(json.dumps(b.observed['authorization']))
    b.newspec['id']='R6-FIXTURE';b.newspec['pins']=dict(b.newspec['pins'],authorization=file_pin(authpath))
    b.newspec['h1']=dict(b.newspec['h1'],attempt=str(storage.target.parent))
    b.newspec['argv']=list(b.newspec['argv']);b.newspec['argv'][5]=str(authpath)
    b.newspec['argv'][7]=str(storage.target.parent)
    storage.target.mkdir(parents=True,exist_ok=True)
    sp=tmp_path/'merged-spec.json';sp.write_text(json.dumps(b.newspec));b.plan['spec_pin']=file_pin(sp)
    pp=tmp_path/'merged-plan.json';pp.write_text(json.dumps(b.plan));b.args['plan_pin']=file_pin(pp)
    b.jid='J-'+sha(dict(id=b.newspec['id'],spec=b.newspec))[:16]
    b.journal=storage.j
    with b.journal.writer():
        s=b.journal.read();s['budget']['capacity_pass']=True;b.journal.save(s)
    b.review.return_value=(b.plan,b.newspec,b.jid)
    monkeypatch.setattr(allocation,'for_spec',real_for_spec)
    monkeypatch.setattr(r6_next_job,'reviewed_plan',b.review)
    result=issue(b)
    assert 'storage_binding' not in b.journal.read()['events'][owner]
    with b.journal.writer():
        edge=r6_next_job.register(b.journal,b.args['parent_key'],b.args['plan_pin'],
            b.args['review_key'],b.args['review_pin'],b.store,free_disk_bytes=500*1024**3)
    assert b.journal.read()['events'][edge]['identity']['job_id']==result['job_id']
    assert b.jid not in b.store.replay()['jobs']
    from tools.r6_register_job import register as register_job
    from tools.r6_budget import totals
    before=totals(b.journal.read()['budget'])[allocation.DISK]
    with b.journal.writer():
        job=register_job(b.journal,b.plan['spec_pin'],b.store,free_disk_bytes=500*1024**3)
        assert register_job(b.journal,b.plan['spec_pin'],b.store,free_disk_bytes=500*1024**3)==job
    state=b.journal.read()
    assert state['events'][owner]['storage_binding']['job_id']==b.jid
    assert state['events'][job]['resources'][allocation.DISK]==0
    assert totals(state['budget'])[allocation.DISK]==before
