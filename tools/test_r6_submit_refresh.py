"""Slow-validation recovery with real queue records and synthetic CPU clocks."""
import copy
from unittest.mock import Mock
import pytest
from rlflow import experiments
from tools import r6_resource_queue as queue
from tools.r6_recovery import RecoveryError
from tools.r6_submit import submit_and_close
from tools.test_r6_submit_close import cycle
from tools.test_r6_submit_store import bound


@pytest.fixture
def slow(cycle,monkeypatch):
    b=cycle;clock=queue.now()
    monkeypatch.setattr(queue,'now',lambda:dict(clock))
    original=experiments.authorized
    def authorize(*args):
        result=original(*args);clock['monotonic_ns']+=9*10**9;return result
    monkeypatch.setattr(experiments,'authorized',authorize)
    def refresh():
        # Existing collector's backoff and full window, modeled with CPU time.
        old=b.journal.read()['events'][b.queue_key]
        clock['monotonic_ns']=old['finished_clock']['monotonic_ns']+60*10**9
        key=queue.begin(b.journal,old['request']['spec'],ordinal=1,free_disk_bytes=500*1024**3)
        for index in range(13):
            if index:clock['monotonic_ns']+=5*10**9
            sample=copy.deepcopy(old['samples'][index]);sample['monotonic_ns']=clock['monotonic_ns']
            queue.append_sample(b.journal,key,index,sample)
        queue.finish(b.journal,key)
        return key
    b.refresh=Mock(side_effect=refresh)
    return b


def invoke(b,queue_key=None):
    with b.journal.writer():
        return submit_and_close(b.journal,b.key,b.store,
            queue_key=queue_key or b.queue_key,refresh_observation=b.refresh)


def test_slow_validation_refreshes_before_claim_and_recovery_never_refreshes(slow):
    b=slow;old=copy.deepcopy(b.journal.read()['events'][b.queue_key])
    result=invoke(b);claim=b.journal.read()['events'][b.key]['submission_claim']
    assert claim['queue_key']!=b.queue_key
    assert b.journal.read()['events'][b.queue_key]['window_record']==old['window_record']
    assert b.journal.read()['events'][b.queue_key]['queue_status']=='OBSERVATION_PASS'
    b.refresh.assert_called_once()
    b.refresh.side_effect=AssertionError('claimed recovery must not refresh')
    assert invoke(b,claim['queue_key'])==result
    assert b.submit.call_count==1


def test_refresh_cannot_return_old_stale_record(slow):
    b=slow;b.refresh.side_effect=lambda:b.queue_key
    with pytest.raises(RecoveryError,match='QUEUE_STALE'):invoke(b)
    assert not b.journal.read()['events'][b.key].get('submission_claim')
    b.submit.assert_not_called()


@pytest.mark.parametrize('stop_or_expire',['stop','expire'])
def test_final_guards_rechecked_after_refresh(slow,monkeypatch,stop_or_expire):
    b=slow;original=b.refresh.side_effect
    def refresh():
        key=original()
        if stop_or_expire=='stop':(b.journal.directory/'STOP').touch()
        else:
            deadline=b.journal.read()['events'][b.key]['request']['wait_deadline_unix']
            monkeypatch.setattr('tools.r6_submit.time.time',lambda:deadline+1)
        return key
    b.refresh.side_effect=refresh
    with pytest.raises(RecoveryError,match='STOP|DEADLINE'):invoke(b)
    assert not b.journal.read()['events'][b.key].get('submission_claim')
    b.submit.assert_not_called()
