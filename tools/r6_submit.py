"""Single-writer bridge from a reserved R6 event to existing rlflow submit.

No permission issuance, budget reservation, retry, settlement or science decision.
A persisted claim with no authoritative Store job is ambiguous, never retried.
"""
import json
import argparse
import math
import os
import time
from pathlib import Path

from rlflow import experiments
from rlflow.jobs import identity
from rlflow.store import atomic, canonical, sha
from tools.r6_recovery import RecoveryError, checked, file_pin
from tools.r6_job_handoff import no_model_workers
from tools.r6_postrun import stop_paths
from tools import r6_resource_queue as queue


def require(condition, reason):
    if not condition:raise RecoveryError('SUBMIT_'+reason)


def queue_ready(journal, queue_key, spec_pin):
    state=journal.read();entry=queue.validate(journal,queue_key,state)
    require(entry['request']['spec']==spec_pin, 'QUEUE_SPEC_DRIFT')
    require(queue.observation_fresh(journal,queue_key),'QUEUE_STALE')
    return entry['window_record']


def submit_once(journal, key, store, *, queue_key, refresh_observation=None):
    journal.require_writer()
    state=journal.read();entry=journal.canonical_job(key,state=state);request=entry['request']
    require(set(request)=={'spec','wait_deadline_unix'} and sha(request)==entry['identity']['input_sha256']
            and sha(entry['identity'])==key and entry['identity']['goal_id']==state['goal_id'], 'EVENT_IDENTITY')
    spec=json.loads(checked(request['spec']).read_text())
    jid='J-'+sha({'id':spec['id'],'spec':spec})[:16]
    require(jid==entry['identity']['job_id'], 'JOB_IDENTITY')
    root=str(store.root.resolve())
    claim=entry.get('submission_claim');job=store.replay()['jobs'].get(jid)
    if claim is None:
        journal.guard();require(not stop_paths(journal), 'STOP')
        require(job is None, 'EXISTING_JOB_WITHOUT_CLAIM')
        require(entry.get('status')=='INTENT' and entry.get('process') is None
                and not entry.get('submission') and not (journal.directory/'jobs'/key).exists(),
                'UNCLAIMED_SIDE_EFFECT')
        require(state.get('current_job') is None, 'OTHER_JOB_ACTIVE')
        no_model_workers(journal)
        require(not any(e['kind']=='cpu' and (
            (e.get('dispatch_claim') and not e.get('receipt'))
            or any(not c.get('receipt') for c in e.get('sample_claims',{}).values()))
            for e in state['events'].values()), 'CPU_WORKER_UNRESOLVED')
        require(state['budget']['capacity_pass'] is True, 'CAPACITY_NOT_ACCEPTED')
        # Event resources are not themselves a charge. Require the exact
        # persisted reservation before any new Store effect; checking only
        # ledger totals would permit a missing or reduced reservation.
        ledger=state['budget'];event=entry['identity']
        reservation_key=json.dumps(tuple(event[k] for k in sorted(event)),separators=(',', ':'))
        reservations=[r for r in ledger['reservations']
                      if r['identity']==event or r['event_key']==reservation_key]
        require(ledger['goal_id']==state['goal_id'] and len(reservations)==1
                and reservations[0]['identity']==event
                and reservations[0]['event_key']==reservation_key
                and reservations[0]['resources']==entry['resources']
                and reservations[0]['purpose']==entry['purpose'], 'RESERVATION_BINDING')
        if spec.get('backend')=='real':
            require(spec.get('h1',{}).get('mode')=='research-r6-evaluation', 'R6_MODE_REQUIRED')
            auth=json.loads(checked(spec['pins']['authorization']).read_text())
            reserved=entry['resources']
            minimum={'live_starts':1,'simulator_initializations':1,
                'live_wall_seconds':spec['resources']['wall_seconds']+180,
                'allocated_gpu_seconds':spec['resources']['wall_seconds']+180,
                'nontraining_transitions':spec['resources']['transitions'],
                'output_actual_plus_unreleased_reservations_bytes':auth['parameters']['output_cap_bytes']}
            require(all(type(reserved.get(k)) in (int,float) and math.isfinite(reserved[k])
                        and reserved[k]>=v for k,v in minimum.items()), 'RESERVATION_INCOMPLETE')
        else:
            require(spec.get('backend')=='runner_cpu' and state['goal_id']=='R6-CPU-TEST-FIXTURE', 'CPU_FIXTURE_BOUNDARY')
        deadline=request['wait_deadline_unix']
        require(type(deadline) in (int,float) and math.isfinite(deadline)
                and 0<deadline-time.time()<=spec['resources']['wall_seconds'], 'DEADLINE')
        journal.require_dispatch_budget()
        experiments.validate(spec,store.root)
        approval=experiments.authorized(store.replay(),spec,store.root)
        try:
            window=queue_ready(journal,queue_key,request['spec'])
        except RecoveryError as error:
            if str(error)!='SUBMIT_QUEUE_STALE' or refresh_observation is None:raise
            # Complete expensive input validation first. Re-observation uses
            # the existing bounded collector, never a relaxed freshness limit.
            queue_key=refresh_observation()
            window=queue_ready(journal,queue_key,request['spec'])
        # Last STOP/budget check immediately before recording possible effects.
        journal.require_dispatch_budget();require(not stop_paths(journal), 'STOP')
        require(0<deadline-time.time()<=spec['resources']['wall_seconds'], 'DEADLINE')
        claim=dict(store_root=root,job_id=jid,spec_sha256=sha(spec),queue_key=queue_key,
            queue_window=window,approval_id=approval['approval_id'],controller=identity(os.getpid()),
            implementation=file_pin(Path(__file__)))
        state=journal.read();state['events'][key]['submission_claim']=claim
        state['status']='EXACT_JOB_SUBMISSION_CLAIMED';journal.save(state)
        journal.guard();require(not stop_paths(journal), 'STOP')
        experiments.submit(store,spec)
        job=store.replay()['jobs'].get(jid)
    else:
        require(claim['store_root']==root and claim['job_id']==jid
                and claim['spec_sha256']==sha(spec) and claim['queue_key']==queue_key, 'CLAIM_DRIFT')
        require(claim['implementation']==file_pin(Path(__file__)), 'IMPLEMENTATION_DRIFT')
        checked(claim['queue_window'])
        # Recovery never calls submit, even if the original call raised before
        # writing a Store job. That ambiguity requires explicit reconciliation.
    require(job is not None, 'AMBIGUOUS_NO_RETRY')
    require(job['id']==jid and job['spec']==spec and job['approval_id']==claim['approval_id'], 'STORE_BINDING')
    result=dict(schema_version='r6-exact-job-submission/1',event_key=key,
        store_root=root,job_id=jid,spec_sha256=sha(spec),approval_id=claim['approval_id'],
        queue_window=claim['queue_window'],job_observed_in_store=True,automatic_resubmission=False,
        scientific_acceptance=False)
    path=journal.directory/'jobs'/key/'submission.json'
    if entry.get('submission'):
        require(json.loads(checked(entry['submission']).read_text())==result, 'RECEIPT_DRIFT')
        return result
    if path.exists():require(json.loads(path.read_text())==result, 'RECEIPT_DRIFT')
    else:atomic(path,canonical(result)+b'\n')
    state=journal.read();require(state.get('current_job') in (None,jid), 'OTHER_JOB_ACTIVE')
    state['events'][key]['submission']=file_pin(path)
    state.update(current_job=jid,status='WAIT_EXACT_JOB');journal.save(state)
    return result


def submit_and_close(journal, key, store, *, queue_key, refresh_observation=None):
    """One writer owns submit and the existing model-free local wait/close.

Replay of a closed step verifies its original evidence without moving later
analysis/decision state backwards. Unknown submission effects never retry.
"""
    journal.require_writer()
    entry=journal.canonical_job(key)
    spec=json.loads(checked(entry['request']['spec']).read_text())
    for name in ('live_wall_seconds','allocated_gpu_seconds'):
        value=entry['resources'].get(name)
        require(type(value) in (int,float) and math.isfinite(value)
                and value>=spec['resources']['wall_seconds']+180, 'CLOSE_RESERVATION_REQUIRED')
    submission=submit_once(journal,key,store,queue_key=queue_key,
                           refresh_observation=refresh_observation)
    entry=journal.canonical_job(key)
    from tools.r6_job_handoff import job_handoff, exact_job, cleanup_evidence, TERMINAL
    if entry.get('closure'):
        closure=json.loads(checked(entry['closure']).read_text())
        job=exact_job(store.replay(),entry,spec)
        require(job['status'] in TERMINAL, 'CLOSED_JOB_NOT_TERMINAL')
        pins,receipt=cleanup_evidence(store.root/'jobs'/job['id'],job)
        expected=dict(event_key=key,request_sha256=sha(entry['request']),job_id=job['id'],
            spec_sha256=sha(spec),evidence=pins,native_status=receipt['status'],
            native_exit_code=receipt.get('exit_code'),reservation_retained=entry['resources'],
            h1_evidence=job.get('h1_evidence'),scientific_acceptance=False)
        require(all(closure.get(k)==v for k,v in expected.items()), 'CLOSURE_BINDING')
    else:
        closure=job_handoff(journal,key,store)
    return dict(submission=submission,closure=closure)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__, epilog=(
        'Submits or recovers one registered job; never calls a model. '
        '--wait-close retains the writer lock through local wait and closure. '
        'Use tools.r6_batch DIRECTORY EVENT_KEY --job-store STORE for local wait/close.'))
    parser.add_argument('directory',type=Path)
    parser.add_argument('event_key')
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--queue-key',required=True)
    parser.add_argument('--wait-close',action='store_true')
    args=parser.parse_args(argv)
    from tools.r6_recovery import Journal
    from rlflow.store import Store
    journal=Journal(args.directory.resolve())
    with journal.writer():
        # A mistyped Store path must not create a new authority database.
        root=args.store.resolve(strict=True)
        require((root/'runtime/research.db').is_file(), 'STORE_DATABASE_REQUIRED')
        action=submit_and_close if args.wait_close else submit_once
        result=action(journal,args.event_key,Store(root),queue_key=args.queue_key)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
