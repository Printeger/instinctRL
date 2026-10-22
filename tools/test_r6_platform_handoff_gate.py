"""Ordinary local gate with real socket RPC and writer lock; synthetic platform lifecycle."""
import os
import pytest
from rlflow.jobs import identity
from tools import r6_platform_handoff_gate as gate
from tools.r6_recovery import Journal,RecoveryError
from tools.test_r6_platform_rpc import api,traced,delivery,batch


@pytest.fixture
def waiting(api):
    b=api
    with b.j.writer():
        state=b.j.read();owner=state['events'][b.controller]
        owner.update(cursor=dict(phase='FIRST_JOB',completed_jobs=[]),controller_process=identity(os.getpid()))
        b.j.save(state)
    return b


def test_wait_holds_single_writer_until_external_idle_without_model_or_goal_mutation(waiting,monkeypatch):
    b=waiting;b.server.settings['goal']='active';budget=b.j.read()['budget'];cursor=b.j.read()['events'][b.controller]['cursor']
    observations=[]
    def externally_paused(_):
        observations.append(b.j.read()['events'][b.controller]['platform_handoff_gate']['pending_reason'])
        with pytest.raises(BlockingIOError):
            with Journal(b.j.directory).writer():pass
        b.server.settings['goal']='paused'
    monkeypatch.setattr(gate.time,'sleep',externally_paused)
    with b.j.writer():result=gate.wait(b.j,b.controller,timeout_seconds=10,poll_seconds=.01)
    assert result['status']=='READY' and result['checks']==2
    assert observations==['PLATFORM_RPC_GOAL_AUTO_CONTINUATION_NOT_STOPPED']
    assert result['last_rpc']['responses'][4]['result']['goal']['status']=='paused'
    assert b.j.read()['budget']==budget and b.j.read()['events'][b.controller]['cursor']==cursor
    assert set(b.server.methods)<={'initialize','initialized','thread/goal/get','thread/read','thread/turns/list'}


def test_timeout_is_durable_and_restart_does_not_extend_it(waiting):
    b=waiting;b.server.settings['goal']='active'
    with b.j.writer(),pytest.raises(RecoveryError,match='TIMED_OUT'):gate.wait(b.j,b.controller,timeout_seconds=1,poll_seconds=1)
    first=b.j.read()['events'][b.controller]['platform_handoff_gate'];count=len(b.server.methods)
    with b.j.writer(),pytest.raises(RecoveryError,match='TIMED_OUT'):gate.wait(b.j,b.controller,timeout_seconds=1,poll_seconds=1)
    assert len(b.server.methods)==count
    assert b.j.read()['events'][b.controller]['platform_handoff_gate']['deadline_unix']==first['deadline_unix']


def test_stop_during_wait_never_reaches_ready(waiting,monkeypatch):
    b=waiting;b.server.settings['goal']='active'
    monkeypatch.setattr(gate.time,'sleep',lambda _:(b.j.directory/'STOP').touch())
    with b.j.writer(),pytest.raises(RecoveryError,match='STOP'):gate.wait(b.j,b.controller,timeout_seconds=10)
    assert b.j.read()['events'][b.controller]['platform_handoff_gate']['status']=='WAITING'


def test_protocol_or_ownership_error_is_not_treated_as_temporary_busy(waiting):
    b=waiting;b.server.settings['server_error']=True
    with b.j.writer(),pytest.raises(RecoveryError,match='SERVER_ERROR'):gate.wait(b.j,b.controller,timeout_seconds=10)
    assert b.server.methods==['initialize']
    assert b.j.read()['events'][b.controller]['platform_handoff_gate']['status']=='WAITING'


def test_existing_job_recovery_cannot_reset_first_deployment_gate(waiting):
    b=waiting
    with b.j.writer():
        state=b.j.read();state['events'][b.controller]['cursor']['completed_jobs']=['fixture'];b.j.save(state)
        with pytest.raises(RecoveryError,match='FIRST_CONTROLLER_REQUIRED'):gate.wait(b.j,b.controller)
    assert not b.server.methods
