"""Real temporary Store and real-mode control contracts; no child launch.

The imported fixture replaces runtime inspection and GPU observations explicitly.
"""
from unittest.mock import patch
import pytest
from rlflow.store import Conflict
from tools.r6_recovery import RecoveryError
from tools.r6_register_job import register
from tools.test_r6_submit_real_contract import bound, MINIMUM


@pytest.fixture
def fresh(bound):
    b=bound
    with b.journal.writer():
        state=b.journal.read();entry=state['events'].pop(b.key)
        b.spec_pin=entry['request']['spec']
        state['budget']['reservations']=[r for r in state['budget']['reservations'] if r['identity']!=entry['identity']]
        b.journal.save(state)
    return b


def invoke(b,**kwargs):
    with b.journal.writer():return register(b.journal,b.spec_pin,b.store,free_disk_bytes=500*1024**3,**kwargs)


def test_real_contract_reservation_matches_full_minimum_and_never_submits(fresh):
    b=fresh;key=invoke(b);state=b.journal.read();entry=state['events'][key]
    assert entry['resources']==MINIMUM
    assert entry['identity']['job_id']==b.jid
    assert not b.store.replay()['jobs']
    b.popen.assert_not_called()


def test_replay_keeps_original_deadline_and_state_even_after_expiry(fresh):
    b=fresh;key=invoke(b);before=b.journal.read()
    deadline=before['events'][key]['request']['wait_deadline_unix']
    with patch('tools.r6_register_job.time.time',return_value=deadline+1000):
        assert invoke(b)==key
    assert b.journal.read()==before


@pytest.mark.parametrize('mode',['capacity','stop','model'])
def test_registration_gates_prevent_reservation(fresh,mode):
    b=fresh
    with b.journal.writer():
        state=b.journal.read()
        if mode=='capacity':state['budget']['capacity_pass']=False
        if mode=='model':state['current_model_event']='OTHER'
        b.journal.save(state)
    if mode=='stop':(b.journal.directory/'STOP').touch()
    before=b.journal.read()
    with pytest.raises(RecoveryError):invoke(b)
    assert b.journal.read()==before
    b.popen.assert_not_called()


def test_revoked_store_authority_blocks_new_registration(fresh):
    b=fresh
    b.store.transact('FIXTURE_REVOKE','revoke',lambda s:[a.update(revoked=True) for a in s['approvals'].values()])
    before=b.journal.read()
    with pytest.raises(Conflict,match='no unique'):invoke(b)
    assert b.journal.read()==before


def test_replay_cannot_change_purpose_or_replace_missing_reservation(fresh):
    b=fresh;key=invoke(b)
    with pytest.raises(RecoveryError,match='REPLAY_DRIFT'):invoke(b,purpose='confirmation_delivery')
    with b.journal.writer():
        state=b.journal.read();event=state['events'][key]['identity']
        state['budget']['reservations']=[r for r in state['budget']['reservations'] if r['identity']!=event]
        b.journal.save(state)
    before=b.journal.read()
    with pytest.raises(RecoveryError,match='RESERVATION_DRIFT'):invoke(b)
    assert b.journal.read()==before


def test_registered_event_can_enter_existing_submit_bridge(fresh,monkeypatch):
    from tools.r6_submit import submit_once
    b=fresh;key=invoke(b)
    monkeypatch.setattr('rlflow.releases.freeze',lambda root:b.store.root)
    with b.journal.writer(),pytest.raises(OSError,match='CPU stop before Popen'):
        submit_once(b.journal,key,b.store,queue_key=b.queue_key)
    with b.journal.writer():result=submit_once(b.journal,key,b.store,queue_key=b.queue_key)
    assert result['job_id']==b.jid and b.popen.call_count==1
