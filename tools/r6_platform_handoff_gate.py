"""Bounded local startup wait for an externally paused, idle root thread.

This runs in the ordinary controller process under its existing writer lock.
It neither pauses the Goal nor starts a model while awaiting handoff.
"""
import os
import time
from rlflow.jobs import identity
from tools.r6_recovery import RecoveryError
from tools import r6_platform_rpc as rpc,r6_platform_wait as platform
from tools.r6_profile_preparation import guard

PENDING={'PLATFORM_RPC_GOAL_AUTO_CONTINUATION_NOT_STOPPED','PLATFORM_RPC_ROOT_NOT_IDLE',
    'PLATFORM_WAIT_ROOT_TURN_ACTIVE_NO_DISPATCH','PLATFORM_WAIT_HANDOFF_NOT_SETTLED'}


def wait(journal,controller_key,*,timeout_seconds=600,poll_seconds=15):
    guard(journal)
    if type(timeout_seconds) is not int or not 1<=timeout_seconds<=600 or not 0<poll_seconds<=60:
        raise RecoveryError('PLATFORM_HANDOFF_BOUND')
    state=journal.read();owner=state['events'][controller_key];process=identity(os.getpid())
    if (owner['identity']['role']!='r6-controller' or owner.get('controller_process')!=process
            or owner['cursor']['phase']!='FIRST_JOB' or owner['cursor']['completed_jobs']):
        raise RecoveryError('PLATFORM_HANDOFF_FIRST_CONTROLLER_REQUIRED')
    if not rpc.enabled(state):raise RecoveryError('PLATFORM_HANDOFF_RPC_REQUIRED')
    previous=owner.get('platform_handoff_gate')
    if previous:
        deadline=previous['deadline_unix']
        if previous['timeout_seconds']!=timeout_seconds:raise RecoveryError('PLATFORM_HANDOFF_BOUND_DRIFT')
    else:
        deadline=time.time()+timeout_seconds
        owner['platform_handoff_gate']=dict(started_at=time.time(),deadline_unix=deadline,
            timeout_seconds=timeout_seconds,checks=0,status='WAITING',initial_process=process)
        journal.save(state)
    if deadline-time.time()>timeout_seconds:raise RecoveryError('PLATFORM_HANDOFF_CLOCK_BACKWARDS')
    monotonic_deadline=time.monotonic()+max(0,deadline-time.time())
    while True:
        guard(journal)
        if time.time()>=deadline or time.monotonic()>=monotonic_deadline:
            state=journal.read();state['events'][controller_key]['platform_handoff_gate'].update(status='TIMED_OUT',finished_at=time.time())
            state['events'][controller_key]['status']='PLATFORM_HANDOFF_TIMED_OUT'
            journal.save(state);raise RecoveryError('PLATFORM_HANDOFF_TIMED_OUT')
        record=None;pending=None
        try:
            record=rpc.observe(journal.read());rpc.idle(journal.read(),record)
            record=platform.preflight(journal)
        except RecoveryError as error:
            if str(error) not in PENDING:raise
            pending=str(error)
        guard(journal);state=journal.read();owner=state['events'][controller_key];value=owner['platform_handoff_gate']
        value.update(checks=value['checks']+1,last_checked_at=time.time(),process=process,last_rpc=record,
            pending_reason=pending,status='WAITING' if pending else 'READY')
        owner['status']='WAIT_PLATFORM_HANDOFF' if pending else 'PLATFORM_HANDOFF_READY'
        if not pending:
            if time.time()>=deadline or time.monotonic()>=monotonic_deadline:
                value['status']='TIMED_OUT';owner['status']='PLATFORM_HANDOFF_TIMED_OUT';journal.save(state);raise RecoveryError('PLATFORM_HANDOFF_TIMED_OUT')
            value['finished_at']=time.time();journal.save(state);return value
        journal.save(state)
        time.sleep(min(poll_seconds,max(0,min(deadline-time.time(),monotonic_deadline-time.monotonic()))))
