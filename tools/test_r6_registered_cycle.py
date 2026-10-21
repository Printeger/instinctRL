"""Actual reservation/Store/closure composition; synthetic terminal job evidence."""
import pytest
from tools.r6_register_job import register_and_close
from tools.r6_recovery import RecoveryError
from tools.test_r6_submit_close import cycle
from tools.test_r6_submit_store import bound


@pytest.fixture
def unregistered(cycle):
    b=cycle
    with b.journal.writer():
        state=b.journal.read();entry=state['events'].pop(b.key)
        b.spec_pin=entry['request']['spec']
        state['budget']['reservations']=[r for r in state['budget']['reservations']
                                       if r['identity']!=entry['identity']]
        b.journal.save(state)
    finish=b.submit.side_effect
    def terminal(store,spec,*,admission_guard):
        b.key=next(k for k,e in b.journal.read()['events'].items()
                   if e['kind']=='job' and e['identity']['job_id']==b.jid)
        return finish(store,spec,admission_guard=admission_guard)
    b.submit.side_effect=terminal
    return b


def invoke(b):
    with b.journal.writer():
        return register_and_close(b.journal,b.spec_pin,b.store,queue_key=b.queue_key,
                                  free_disk_bytes=500*1024**3)


def test_one_writer_reserves_submits_and_closes_preserving_native_failure(unregistered):
    b=unregistered;result=invoke(b)
    assert result['event_key']==b.key
    assert result['closure']['native_status']=='FAILED'
    assert result['closure']['native_exit_code']==-9
    assert not result['closure']['scientific_acceptance']
    assert b.journal.read()['current_job'] is None
    assert b.submit.call_count==1
    b.popen.assert_not_called()


def test_unavailable_window_does_not_spend_live_reservation(unregistered,monkeypatch):
    b=unregistered;before=b.journal.read()
    from tools import r6_resource_queue as queue
    clock=dict(queue.now());clock['monotonic_ns']+=9*10**9
    monkeypatch.setattr(queue,'now',lambda:clock)
    with pytest.raises(RecoveryError,match='QUEUE_STALE'):invoke(b)
    assert b.journal.read()==before
    b.submit.assert_not_called()


def test_closed_recovery_ignores_new_resource_window_and_preserves_later_phase(unregistered,monkeypatch):
    b=unregistered;first=invoke(b)
    with b.journal.writer():
        state=b.journal.read();state['status']='MODEL_DECISION_RECORDED';b.journal.save(state)
    before=b.journal.read()
    def forbidden(*args):raise AssertionError('Recovery must not demand a fresh idle window')
    monkeypatch.setattr('tools.r6_submit.queue_ready',forbidden)
    (b.journal.directory/'STOP').touch()
    assert invoke(b)==first and b.journal.read()==before
    assert b.submit.call_count==1


def test_crash_after_reservation_reuses_event_and_deadline(unregistered,monkeypatch):
    b=unregistered
    from tools import r6_submit
    original=r6_submit.submit_and_close
    def crash(*args,**kwargs):raise RuntimeError('CPU crash after reservation')
    monkeypatch.setattr(r6_submit,'submit_and_close',crash)
    with pytest.raises(RuntimeError,match='after reservation'):invoke(b)
    state=b.journal.read();key=next(k for k,e in state['events'].items() if e['kind']=='job')
    request=state['events'][key]['request']
    reservations=state['budget']['reservations']
    monkeypatch.setattr(r6_submit,'submit_and_close',original)
    result=invoke(b)
    assert result['event_key']==key
    assert b.journal.read()['events'][key]['request']==request
    after=b.journal.read()['budget']['reservations']
    assert len(after)==len(reservations)
    for old,new in zip(reservations,after):
        for field in ('event_key','identity','resources','purpose'):
            assert old[field]==new[field]
    # Terminal accounting adds actual observations without releasing the charge.
    job_reservation=next(r for r in after if r['identity']['job_id']==b.jid and r['identity']['role']=='job')
    assert job_reservation['observations']
    assert b.submit.call_count==1
