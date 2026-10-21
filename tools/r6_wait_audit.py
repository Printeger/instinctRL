"""Bounded per-controller local wait evidence; never asserts platform inactivity."""
import time
from rlflow.jobs import identity
from rlflow.store import sha
from tools.r6_recovery import RecoveryError
import os


def fingerprint(state):
    return dict(model_events_sha256=sha({k:e['identity'] for k,e in state['events'].items() if e['kind']=='model'}),
        platform_wake_accounting=state.get('platform_automatic_wake_accounting'))


def record(journal,key,*,finish=False):
    """Called only after the existing no_model_workers guard on every iteration."""
    state=journal.read();entry=journal.canonical_job(key,state=state)
    audit=entry.get('local_wait_audit')
    if audit and audit.get('complete'):return
    # Do not manufacture wait evidence for a previously closed historical job.
    if not audit and entry.get('closure'):return
    now=time.time();controller=identity(os.getpid());current=fingerprint(state)
    if audit is None:
        audit=dict(schema_version='r6-local-wait-audit/1',event_key=key,job_id=entry['identity']['job_id'],
            fingerprint=current,segments=[],complete=False,scope='REGISTERED_LOCAL_MODEL_WORKERS_ONLY',
            platform_inactivity_proven=False)
        entry['local_wait_audit']=audit
    if audit['fingerprint']!=current:raise RecoveryError('WAIT_AUDIT_NEW_MODEL_OR_PLATFORM_WAKE')
    if not audit['segments'] or audit['segments'][-1]['controller']!=controller:
        if len(audit['segments'])>=64:raise RecoveryError('WAIT_AUDIT_SEGMENT_CAP')
        if audit['segments']:audit['segments'][-1]['interrupted']=True
        audit['segments'].append(dict(controller=controller,started_at=now,last_checked_at=now,
            checks=0,registered_model_workers_present=False,interrupted=False))
    segment=audit['segments'][-1]
    if now<segment['last_checked_at']:raise RecoveryError('WAIT_AUDIT_CLOCK_BACKWARDS')
    segment['checks']+=1;segment['last_checked_at']=now
    if finish:
        segment['finished_at']=now;audit['complete']=True
        audit['unobserved_restart_gap']=any(s['interrupted'] for s in audit['segments'])
    journal.save(state)
