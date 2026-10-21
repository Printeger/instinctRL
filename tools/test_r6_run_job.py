"""One local entry point, real temporary Store/Journal and synthetic closure.

Resource queries and downstream analysis/model handoff are replaced; this is
not evidence of a deployed controller or of full scientific execution.
"""
from unittest.mock import Mock
import pytest
from tools import r6_run_job as runner
from tools.r6_recovery import Journal, RecoveryError, file_pin
from tools.test_r6_registered_cycle import unregistered
from tools.test_r6_submit_close import cycle
from tools.test_r6_submit_store import bound


@pytest.fixture
def local(unregistered,monkeypatch):
    b=unregistered
    b.observe=Mock(return_value=dict(status='OBSERVATION_PASS',event_key=b.queue_key))
    def downstream(journal,key,store,*args,**kwargs):
        assert journal.canonical_job(key)['closure']
        with pytest.raises(BlockingIOError):
            with Journal(journal.directory).writer():pass
        return {'proposal_only':True}
    b.downstream=Mock(side_effect=downstream)
    monkeypatch.setattr(runner,'wait_for_observation',b.observe)
    monkeypatch.setattr(runner,'finish_and_decide',b.downstream)
    return b


def invoke(b):
    with b.journal.writer():
        return runner.run(b.journal,b.spec_pin,b.store,file_pin(b.script),
                          file_pin(b.script),free_disk_bytes=500*1024**3)


def test_one_writer_reaches_downstream_and_restart_skips_resource_wait(local):
    b=local;first=invoke(b)
    assert first['closure']['native_status']=='FAILED'
    assert first['result']=={'proposal_only':True}
    assert not first['scientific_acceptance']
    b.observe.side_effect=AssertionError('claimed job must not requeue')
    (b.journal.directory/'STOP').touch()
    assert invoke(b)==first
    assert b.submit.call_count==1 and b.downstream.call_count==2
    b.popen.assert_not_called()


def test_crash_after_closure_resumes_downstream_without_resubmit(local):
    b=local;downstream=b.downstream.side_effect
    b.downstream.side_effect=RuntimeError('CPU crash after close')
    with pytest.raises(RuntimeError,match='after close'):invoke(b)
    b.downstream.side_effect=downstream
    b.observe.side_effect=AssertionError('must reconnect original claim')
    assert invoke(b)['status']=='DECISION_PROPOSAL_RETURNED'
    assert b.submit.call_count==1


def test_ambiguous_submission_never_requeues_or_retries(local):
    b=local;b.submit.side_effect=RuntimeError('ambiguous submit')
    with pytest.raises(RuntimeError,match='ambiguous submit'):invoke(b)
    b.observe.side_effect=AssertionError('must not requeue')
    with pytest.raises(RecoveryError,match='AMBIGUOUS_NO_RETRY'):invoke(b)
    assert b.submit.call_count==1
    b.downstream.assert_not_called()


@pytest.mark.parametrize('gate',['stop','capacity','approval'])
def test_preflight_rejects_before_resource_queries(local,gate):
    b=local
    if gate=='stop':(b.journal.directory/'STOP').touch()
    elif gate=='capacity':
        with b.journal.writer():
            state=b.journal.read();state['budget']['capacity_pass']=False;b.journal.save(state)
    else:
        b.store.transact('FIXTURE_REVOKE','revoke',lambda s:s['approvals'][b.approval].update(revoked=True))
    with pytest.raises((RecoveryError,ValueError)):invoke(b)
    b.observe.assert_not_called();b.submit.assert_not_called();b.downstream.assert_not_called()


def test_failed_observation_does_not_reserve_or_submit(local):
    b=local;before=b.journal.read()
    b.observe.return_value=dict(status='INVALID_WINDOW',event_key=b.queue_key)
    assert invoke(b)['status']=='RESOURCE_OBSERVATION_REJECTED'
    assert b.journal.read()==before
    b.submit.assert_not_called();b.downstream.assert_not_called()


def test_expired_unclaimed_reservation_is_not_replaced(local,monkeypatch):
    from tools.r6_register_job import register
    b=local
    with b.journal.writer():
        key=register(b.journal,b.spec_pin,b.store,free_disk_bytes=500*1024**3)
    before=b.journal.read()
    monkeypatch.setattr(runner.time,'time',lambda:before['events'][key]['request']['wait_deadline_unix']+1)
    with pytest.raises(RecoveryError,match='EXPIRED_RESERVATION'):invoke(b)
    assert b.journal.read()==before
    b.observe.assert_not_called();b.submit.assert_not_called()
