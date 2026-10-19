"""Close an already submitted exact R6 job using existing rlflow evidence.

This module cannot submit, issue permissions, refund reservations, launch a model
or accept scientific results. The batch writer waits locally and preserves native
failure facts. Resource admission and new-job dispatch remain separate gates.
"""
from pathlib import Path
import json
import math
import os
import time

from rlflow import experiments, policy
from rlflow.jobs import identity, owns
from rlflow.store import atomic, canonical, sha
from tools.r6_recovery import RecoveryError, checked, file_pin

TERMINAL = {'FAILED', 'STOPPED', 'SUCCEEDED'}


def read(path):
    return json.loads(Path(path).read_text())


def exact_job(state, entry, spec):
    jid = 'J-' + sha({'id': spec['id'], 'spec': spec})[:16]
    if entry['identity']['job_id'] != jid:
        raise RecoveryError('JOB_EVENT_IDENTITY_MISMATCH')
    job = state['jobs'].get(jid)
    if job is None:
        raise RecoveryError('JOB_MISSING_NO_RESUBMISSION')
    if job.get('id') != jid or job.get('spec') != spec:
        raise RecoveryError('JOB_SPEC_DRIFT')
    return job


def no_model_workers(journal):
    state = journal.read()
    if state.get('current_model_event') is not None:
        raise RecoveryError('MODEL_HANDOFF_STILL_ACTIVE')
    for key, event in state['events'].items():
        if event['kind'] != 'model':
            continue
        if owns(event.get('process')):
            raise RecoveryError('MODEL_WORKER_ALIVE_DURING_JOB')
        call = journal.directory / 'calls' / key
        for filename in ('worker.json', 'child.json'):
            path = call / filename
            if path.exists() and owns(read(path).get('identity')):
                raise RecoveryError('MODEL_PROCESS_ALIVE_DURING_JOB')
        # An unconsumed dispatch can hide an as-yet-unbound child. Never infer
        # absence from a missing PID/receipt alone.
        if event.get('dispatch_claim') and not event.get('receipt'):
            raise RecoveryError('MODEL_DISPATCH_UNRESOLVED')
        if event.get('receipt'):
            receipt = read(checked(event['receipt']))
            if owns(receipt.get('process')) or owns(receipt.get('child_identity')):
                raise RecoveryError('MODEL_RECEIPT_PROCESS_ALIVE_DURING_JOB')
            cleanup = receipt.get('cleanup', {})
            if (cleanup.get('status') != 'VERIFIED_NO_OWNED_DESCENDANTS'
                    or any(owns(i) for i in cleanup.get('observed_identities', []))):
                raise RecoveryError('MODEL_CLEANUP_UNCONFIRMED')
    return sum(e['kind'] == 'model' for e in state['events'].values())


def cleanup_evidence(directory, job):
    names = ('receipt.json', 'lifeguard-ack.json', 'descendant-cleanup.json',
             'lifeguard-ready.json', 'process-identity.json', 'descendant-ledger.json')
    if directory.is_symlink() or any((directory / name).is_symlink() for name in names):
        raise RecoveryError('JOB_EVIDENCE_SYMLINK')
    pins = {name: file_pin(directory / name) for name in names}
    records = {name: read(checked(pin)) for name, pin in pins.items()}
    receipt = records['receipt.json']
    if (receipt.get('job_id') != job['id'] or receipt.get('spec_sha256') != sha(job['spec'])
            or receipt.get('status') != job['status'] or job.get('receipt') != receipt):
        raise RecoveryError('NATIVE_RECEIPT_BINDING')
    ack = records['lifeguard-ack.json']; clean = records['descendant-cleanup.json']
    if (ack.get('owner') != job.get('identity') or not isinstance(ack.get('owner'), dict)
            or ack.get('descendant_cleanup') != clean or ack.get('no_survivors') is not True
            or clean.get('quiescent') is not True or clean.get('timed_out') is not False
            or clean.get('survivors') != []
            or type(clean.get('stable_scans')) is not int
            or type(clean.get('required_stable_scans')) is not int
            or clean['stable_scans'] < max(2, clean['required_stable_scans'])):
        raise RecoveryError('NORMAL_ACK_REQUIRED')
    if records['descendant-ledger.json'].get('owner') != job['identity']:
        raise RecoveryError('DESCENDANT_LEDGER_OWNER')
    identities = [job['identity'], records['lifeguard-ready.json']['identity'],
                  records['process-identity.json'], *clean['discovered'],
                  *records['descendant-ledger.json']['descendants']]
    if any(owns(i) for i in identities):
        raise RecoveryError('OWNED_PROCESS_STILL_ALIVE')
    # Re-read after the liveness scan; never bind evidence that changed mid-check.
    for pin in pins.values():
        checked(pin)
    return pins, receipt


def close_terminal_job(journal, key, store):
    """Idempotent CPU close record, not a resource refund or scientific verdict."""
    journal.require_writer()
    state = journal.read(); entry = state['events'][key]
    if entry['kind'] != 'job' or entry['identity']['input_sha256'] != sha(entry['request']):
        raise RecoveryError('JOB_EVENT_REQUIRED')
    spec = read(checked(entry['request']['spec']))
    job = exact_job(store.replay(), entry, spec)
    if job['status'] not in TERMINAL:
        raise RecoveryError('JOB_NOT_TERMINAL')
    pins, receipt = cleanup_evidence(store.root / 'jobs' / job['id'], job)
    violations = []
    observations = {}
    for field, resource in (('elapsed_seconds', 'live_wall_seconds'), ('allocated_seconds', 'allocated_gpu_seconds')):
        value = receipt.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            violations.append('UNKNOWN_OR_INVALID_' + field)
        else:
            observations[resource] = value
            if value > entry['resources'].get(resource, -1):
                violations.append('RESERVATION_EXCEEDED_' + resource)
    result = dict(schema_version='r6-job-close/1', event_key=key,
        request_sha256=sha(entry['request']), job_id=job['id'], spec_sha256=sha(spec),
        native_status=receipt['status'], native_exit_code=receipt.get('exit_code'),
        evidence=pins, observed=observations, h1_evidence=job.get('h1_evidence'),
        reservation_retained=entry['resources'], budget_violations=violations,
        scientific_acceptance=False, output_inventory_complete=False,
        postrun_complete=False, status='CLOSED_REQUIRES_CPU_ANALYSIS_AND_POSTRUN')
    path = journal.directory / 'jobs' / key / 'close.json'
    if entry.get('closure'):
        if read(checked(entry['closure'])) != result:
            raise RecoveryError('JOB_CLOSURE_DRIFT')
        return result
    if path.exists():
        if read(path) != result:
            raise RecoveryError('JOB_CLOSURE_DRIFT')
    else:
        atomic(path, canonical(result) + b'\n')
    entry.update(closure=file_pin(path), status='JOB_CLOSED')
    state.update(current_job=None, status='JOB_CLOSED_PENDING_ANALYSIS')
    journal.save(state)
    return result


def job_handoff(journal, key, store, *, poll_seconds=15):
    """Wait locally for one existing job; STOP requests only its owned cancel.

    Deadline is part of the pre-registered request, not extended on restart.
    A missing job or dead handle without a conclusive receipt is reconciled once,
    then reported as ambiguous. Neither path submits or reruns anything.
    """
    journal.require_writer()
    if type(poll_seconds) not in (int, float) or not 0 < poll_seconds <= 60:
        raise RecoveryError('LOCAL_POLL_INTERVAL')
    state = journal.read(); entry = state['events'][key]; request = entry['request']
    if (entry['kind'] != 'job' or set(request) != {'spec', 'wait_deadline_unix'}
            or entry['identity']['input_sha256'] != sha(request)):
        raise RecoveryError('JOB_REQUEST_IDENTITY')
    deadline = request['wait_deadline_unix']
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise RecoveryError('JOB_DEADLINE')
    spec = read(checked(request['spec']))
    for resource in ('live_wall_seconds', 'allocated_gpu_seconds'):
        if entry['resources'].get(resource, -1) < spec['resources']['wall_seconds'] + 180:
            raise RecoveryError('JOB_CLEANUP_RESERVATION_REQUIRED')
    if deadline > time.time() + spec['resources']['wall_seconds']:
        raise RecoveryError('JOB_DEADLINE_EXCEEDS_BOUND')
    job = exact_job(store.replay(), entry, spec)
    if state.get('current_job') not in (None, job['id']):
        raise RecoveryError('OTHER_JOB_HANDOFF_ACTIVE')
    model_count = no_model_workers(journal)
    directory = store.root / 'jobs' / job['id']
    state.update(current_job=job['id'], controller_identity=identity(os.getpid()), status='WAIT_EXACT_JOB')
    journal.save(state)
    while True:
        if no_model_workers(journal) != model_count:
            raise RecoveryError('NEW_MODEL_EVENT_DURING_JOB_WAIT')
        store_state = store.replay()
        job = exact_job(store_state, journal.read()['events'][key], spec)
        state = journal.read(); entry = state['events'][key]
        root = Path(__file__).resolve().parents[1]
        extra_stops = [journal.directory.parent/'STOP', root/'STOP',
            root/'coordination/STOP', root/'coordination/commander/STOP', root/'coordination/executor/STOP']
        stop = bool(journal.stopped() or policy.stopped(store.root)
            or any(p.exists() for p in extra_stops) or store_state.get('paused'))
        expired = time.time() >= deadline
        if stop or expired:
            entry['stop_seen'] = bool(entry.get('stop_seen') or stop)
            entry['deadline_seen'] = bool(entry.get('deadline_seen') or expired)
            journal.save(state)
            if job['status'] not in TERMINAL:
                if entry.get('cancel_claim'):
                    if not (directory / 'cancel').exists():
                        raise RecoveryError('CANCEL_AMBIGUOUS_NO_RETRY')
                else:
                    entry['cancel_claim'] = dict(job_id=job['id'], identity=job.get('identity'))
                    journal.save(state)
                    experiments.cancel(store, job['id'])
        if (directory / 'receipt.json').exists() or not owns(job.get('identity')):
            experiments.reconcile(store, job['id'])
            job = exact_job(store.replay(), journal.read()['events'][key], spec)
        if job['status'] in TERMINAL:
            result = close_terminal_job(journal, key, store)
            state = journal.read(); entry = state['events'][key]
            state['current_job'] = None
            state['status'] = ('STOPPED_AFTER_OWNED_CLEANUP' if entry.get('stop_seen')
                else 'JOB_BOUND_EXCEEDED' if entry.get('deadline_seen') or result['budget_violations']
                else 'CPU_ANALYZE')
            journal.save(state)
            return result
        if not owns(job.get('identity')):
            raise RecoveryError('JOB_HANDLE_MISSING_NO_RETRY')
        if time.time() >= deadline + 180:
            raise RecoveryError('OWNED_CLEANUP_TIMEOUT')
        state = journal.read(); entry = state['events'][key]
        entry['local_wait_observations'] = entry.get('local_wait_observations', 0) + 1
        entry['last_local_wait'] = dict(time=time.time(), job_id=job['id'], process=job['identity'],
            model_events=model_count, model_workers_present=False)
        journal.save(state)
        time.sleep(poll_seconds)
