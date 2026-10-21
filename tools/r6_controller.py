"""Persist the R6 first-job/decision/preparation/next-job handoffs in one writer.

This composes existing bounded runners; it is not a scheduler or an approval.
Source compatibility uses its own exact review and frozen validation. Final
delivery still persists an incomplete boundary pending its delivery workflow.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from rlflow.jobs import identity
from rlflow.store import sha
from tools.r6_recovery import Journal, RecoveryError, checked, file_pin
from tools.r6_postrun import stop_paths
from tools.r6_job_handoff import no_model_workers
from tools.r6_decision import record as decision_record
from tools.r6_source_preparation import control_identity
from tools import r6_start_baseline as first
from tools import r6_continue_job as next_entry
from tools import r6_profile_preparation as profile
from tools import r6_source_preparation as source
from tools import r6_materialize_job as materializer
from tools import r6_execution_compatibility as compatibility
from tools import r6_training_preparation as training
from tools import r6_candidate_preparation as candidate
from tools import r6_confirmation_inputs as confirmation_inputs
from tools import r6_confirmation_preparation as confirmation
from tools import r6_final_models as final_models
from tools import r6_final_freeze_review as final_freeze_review
from tools import r6_holdout_publication as holdout_publication
from tools import r6_final_inputs as final_inputs
from tools import r6_final_preparation as final_preparation
from tools import r6_final_storage as final_storage
from tools import r6_final_plans_review as final_plans_review
from tools import r6_final_sequence as final_sequence
from tools import r6_delivery_evidence as delivery_evidence
from tools.r6_decision_packet import persist_exact
from rlflow.store import canonical

ROLE='r6-controller'
CAP=256*1024
ROOT=Path(__file__).resolve().parents[1]
TERMINAL_BOUNDARIES={'FINAL_DELIVERY_REQUIRED','FINAL_MEDIA_REQUIRED'}


def require(condition,reason):
    if not condition:raise RecoveryError('CONTROLLER_'+reason)


def implementation():
    return dict(control_sha256=control_identity(),schemas_sha256=sha({
        str(p.relative_to(ROOT)):file_pin(p)['sha256']
        for p in (ROOT/'docs/experiments/r6-robustness-auto/prompts').glob('*.schema.json')}))


def register(journal,request_pin,store,*,free_disk_bytes):
    journal.require_writer();state=journal.read()
    args=json.loads(checked(request_pin).read_text())
    require(set(args)=={'initial_request','profile_context'},'REQUEST_FIELDS')
    initial=json.loads(checked(args['initial_request']).read_text())
    require(set(initial)=={'plan_pin','master_pin','review_executable'},'INITIAL_FIELDS')
    for pin in initial.values():checked(pin)
    checked(args['profile_context'])
    request=dict(inputs=request_pin,implementation=implementation(),store_root=str(store.root.resolve()))
    event=dict(goal_id=state['goal_id'],event_id='R6_CONTROLLER',job_id='R6_CONTROLLER',
               input_sha256=sha(request),role=ROLE)
    previous=[(k,e) for k,e in state['events'].items() if e['identity']['role']==ROLE]
    require(len(previous)<=1,'ALIAS')
    if previous:
        key,entry=previous[0]
        require(entry['identity']==event and sha(event)==key and entry['request']==request,'INPUT_DRIFT')
    else:
        journal.guard();require(not stop_paths(journal),'STOP');no_model_workers(journal)
        require(state.get('current_job') is None,'ACTIVE_JOB')
        key,_=journal.intent(event,kind='cpu',request=request,
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
            purpose='development',free_disk_bytes=free_disk_bytes)
    state=journal.read();entry=state['events'][key]
    rows=[r for r in state['budget']['reservations'] if r['identity']==event]
    require(entry['kind']=='cpu' and entry['resources']=={'output_actual_plus_unreleased_reservations_bytes':CAP}
        and entry['purpose']=='development' and len(rows)==1 and rows[0]['resources']==entry['resources']
        and rows[0]['purpose']==entry['purpose']
        and rows[0]['event_key']==json.dumps(tuple(event[k] for k in sorted(event)),separators=(',',':')),
        'RESERVATION')
    # Intent is sufficient to recover a crash before the first cursor save.
    if 'cursor' not in entry:
        entry['cursor']=dict(phase='FIRST_JOB',context=args['profile_context'],completed_jobs=[])
    entry['controller_process']=identity(os.getpid());journal.save(state)
    return key,initial


def save(journal,key,cursor,*,status='RUNNING',error=None):
    state=journal.read();entry=state['events'][key]
    require(len(json.dumps(cursor).encode())<CAP-65536,'CURSOR_CAP')
    entry.update(cursor=cursor,status=status)
    if error is None:entry.pop('controller_error',None)
    else:entry['controller_error']=dict(type=type(error).__name__,message=str(error)[:4096])
    journal.save(state)


def parent_for_job(journal,job_key,store):
    state=journal.read();job=journal.canonical_job(job_key,state=state)
    matches=[k for k,e in state['events'].items() if e['kind']=='model'
        and e['identity']['role']=='post-analysis-decision'
        and e['identity']['job_id']==job['identity']['job_id']]
    require(len(matches)==1,'EXACT_JOB_DECISION')
    parent=matches[0]
    require(state['events'][parent].get('decision_input',{}).get('job_event_key')==job_key,'PARENT_JOB')
    return parent,decision_record(journal,parent,store)


def run(journal,request_pin,store,*,free_disk_bytes):
    """Run until a concrete external/unsupported boundary, never pure-wait models.

    Existing runners own live/model waits and claimed-action STOP cleanup.
    A crash preserves the current phase, whose existing event is replayed on
    restart. No new attempt is inferred from a missing result or process.
    """
    key,initial=register(journal,request_pin,store,free_disk_bytes=free_disk_bytes)
    while True:
        cursor=journal.read()['events'][key]['cursor'];phase=cursor['phase']
        free=min(free_disk_bytes,shutil.disk_usage(journal.directory).free)
        try:
            if phase=='FINAL_DELIVERY_REQUIRED' and 'final_collection' in cursor:
                outcome=delivery_evidence.prepare(journal,key,store,free_disk_bytes=free)
                require(outcome['status']=='DELIVERY_EVIDENCE_PREPARED_NOT_ACCEPTED','DELIVERY_EVIDENCE_STATUS')
                cursor.update(phase='FINAL_MEDIA_REQUIRED',delivery_evidence=outcome)
                save(journal,key,cursor)
                continue
            if phase in TERMINAL_BOUNDARIES:
                save(journal,key,cursor,status=phase)
                return dict(status=phase,event_key=key,goal_complete=False)
            if phase in {'FIRST_JOB','NEXT_JOB'}:
                if phase=='FIRST_JOB':
                    outcome=first.start(journal,store=store,free_disk_bytes=free,**initial)
                else:
                    outcome=next_entry.continue_job(journal,cursor['parent_key'],cursor['plan'],
                        initial['master_pin'],initial['review_executable'],store,free_disk_bytes=free)
                if outcome['status']=='RESOURCE_OBSERVATION_REJECTED':
                    save(journal,key,cursor,status='RESOURCE_OBSERVATION_REJECTED')
                    return dict(status='RESOURCE_OBSERVATION_REJECTED',event_key=key,goal_complete=False)
                require(outcome['status']=='DECISION_PROPOSAL_RETURNED','JOB_STATUS')
                job_key=outcome['event_key'] if phase=='FIRST_JOB' else outcome['child_job_event_key']
                journal.canonical_job(job_key)
                require(job_key not in cursor['completed_jobs'],'REPEATED_JOB')
                cursor['completed_jobs'].append(job_key)
                for name in ('source_preparation','snapshot_receipt','materialization_inputs'):
                    cursor.pop(name,None)
                cursor.update(phase='DECIDE',job_key=job_key)
            elif phase=='DECIDE':
                journal.guard();require(not stop_paths(journal),'STOP')
                parent,result=parent_for_job(journal,cursor['job_key'],store)
                decision=result['decision']['decision']
                cursor['parent_key']=parent
                if 'confirmation' in cursor:
                    sequence=cursor['confirmation'];index=len(sequence['results'])
                    require(index<3 and decision in {'PROPOSE_CONFIRMATION','REQUEST_FINAL_REVIEW'},
                        'CONFIRMATION_REQUIRES_REPAIR')
                    evidence=confirmation.collect(journal,cursor['job_key'],sequence['contexts'][index],store)
                    require(all(r['job_event_key']!=evidence['job_event_key'] for r in sequence['results']),
                        'CONFIRMATION_REPEATED_JOB')
                    sequence['results'].append(evidence)
                    if len(sequence['results'])==3:cursor['phase']='FINAL_MODEL_PREPARATION'
                    else:cursor.update(phase='PROFILE',context=sequence['contexts'][index+1])
                elif decision=='PROPOSE_CONFIRMATION':cursor['phase']='CONFIRMATION_INPUTS'
                elif decision=='PROPOSE_BASELINE_VALIDATION':
                    cursor.update(phase='FINAL_MODEL_PREPARATION',baseline_validation=dict(
                        parent_key=parent,job_key=cursor['job_key'],context=cursor['context']))
                elif decision=='REQUEST_FINAL_REVIEW':cursor['phase']='FINAL_DELIVERY_REQUIRED'
                elif decision=='PROPOSE_CANDIDATE':cursor['phase']='SOURCE'
                else:cursor['phase']='PROFILE'
            elif phase=='CONFIRMATION_INPUTS':
                outcome=confirmation_inputs.prepare(journal,cursor['parent_key'],cursor['context'],store,
                    executable=initial['review_executable'],free_disk_bytes=free)
                require(outcome['status']=='CONFIRMATION_INPUTS_PREPARED_NOT_REVIEWED','CONFIRMATION_INPUT_STATUS')
                cursor.update(phase='CONFIRMATION_PREPARATION',confirmation_inputs=outcome['inputs'])
            elif phase=='FINAL_MODEL_PREPARATION':
                outcome=final_models.prepare(journal,key,store,free_disk_bytes=free)
                require(outcome['status']=='FINAL_MODELS_BOUND_REVIEW_REQUIRED','FINAL_MODELS_STATUS')
                cursor.update(phase='FINAL_MODEL_REVIEW',final_models=outcome)
            elif phase=='FINAL_MODEL_REVIEW':
                outcome=final_freeze_review.prepare(journal,cursor['final_models']['freeze'],store,
                    executable=initial['review_executable'],free_disk_bytes=free)
                require(outcome['status']=='FINAL_MODEL_FREEZE_REVIEWED_NOT_RELEASED','FINAL_FREEZE_REVIEW_STATUS')
                cursor.update(phase='HOLDOUT_PREPARATION',final_freeze=outcome)
            elif phase=='HOLDOUT_PREPARATION':
                request={name:cursor['final_freeze'][name] for name in ('freeze','review')}
                outcome=holdout_publication.prepare(journal,request,free_disk_bytes=free)
                require(outcome['status']=='HOLDOUT_RELEASED_FINAL_BATCH_INPUTS_READY','HOLDOUT_STATUS')
                cursor.update(phase='FINAL_EVALUATION_PREPARATION_REQUIRED',final_batch=outcome)
            elif phase=='FINAL_EVALUATION_PREPARATION_REQUIRED':
                outcome=final_inputs.prepare(journal,key,cursor['final_batch']['batch'],store,free_disk_bytes=free)
                require(outcome['status']=='FINAL_INPUTS_AND_SNAPSHOT_READY_NOT_REVIEWED','FINAL_INPUTS_STATUS')
                cursor.update(phase='FINAL_PREPARATION',final_inputs=outcome)
            elif phase=='FINAL_PREPARATION':
                inputs=cursor['final_inputs']
                outcome=final_preparation.prepare(journal,inputs['snapshot'],inputs['inputs'],store,
                    executable=initial['review_executable'],free_disk_bytes=free)
                require(outcome['status']=='FINAL_BATCH_CONTEXTS_READY_NOT_RESERVED','FINAL_PREPARATION_STATUS')
                pin=journal.read()['events'][outcome['event_key']]['result']
                require(json.loads(checked(pin).read_text())==outcome,'FINAL_PREPARATION_PIN')
                cursor.update(phase='FINAL_MATERIALIZE',final_preparation=pin)
            elif phase=='FINAL_MATERIALIZE':
                outcome=final_storage.materialize_all(journal,cursor['final_preparation'],store,free_disk_bytes=free)
                if outcome['status']=='WAITING_FOR_ADMIN_PROVISION':
                    save(journal,key,cursor,status='WAITING_FOR_ADMIN_PROVISION')
                    return dict(status='WAITING_FOR_ADMIN_PROVISION',event_key=key,missing=outcome['missing'],goal_complete=False)
                require(outcome['status']=='FINAL_BATCH_PLANS_READY_NOT_AUTHORIZED','FINAL_MATERIALIZE_STATUS')
                path=journal.directory/'controllers'/key/'final-materializations.json'
                data=canonical(outcome)+b'\n';require(len(data)<65536,'FINAL_MATERIALIZATIONS_CAP')
                profile.guard(journal);persist_exact(path,data)
                cursor.update(phase='FINAL_PLANS_REVIEW',final_materializations=file_pin(path))
            elif phase=='FINAL_PLANS_REVIEW':
                outcome=final_plans_review.prepare(journal,cursor['final_materializations'],store,
                    executable=initial['review_executable'],free_disk_bytes=free)
                require(outcome['status']=='FINAL_BATCH_PLANS_REVIEWED_NOT_AUTHORIZED','FINAL_PLANS_REVIEW_STATUS')
                cursor.update(phase='FINAL_SEQUENCE',final_plans_review=outcome)
            elif phase=='FINAL_SEQUENCE':
                reviewed=cursor['final_plans_review']
                outcome=final_sequence.advance(journal,cursor['final_materializations'],reviewed['event_key'],
                    reviewed['review'],initial['master_pin'],store,free_disk_bytes=free)
                if outcome['status']=='RESOURCE_OBSERVATION_REJECTED':
                    save(journal,key,cursor,status=outcome['status'])
                    return dict(status=outcome['status'],event_key=key,goal_complete=False)
                require(outcome['status']=='FINAL_BATCH_COLLECTED_REQUIRES_INDEPENDENT_AUDIT','FINAL_SEQUENCE_STATUS')
                cursor.update(phase='FINAL_DELIVERY_REQUIRED',final_collection=outcome)
            elif phase=='CONFIRMATION_PREPARATION':
                outcome=confirmation.prepare(journal,cursor['parent_key'],cursor['confirmation_inputs'],store,
                    executable=initial['review_executable'],free_disk_bytes=free)
                require(outcome['status']=='THREE_CONFIRMATION_CONTEXTS_READY_NOT_RESERVED'
                    and len(outcome['contexts'])==3,'CONFIRMATION_PREPARATION_STATUS')
                cursor.update(phase='PROFILE',context=outcome['contexts'][0],
                    confirmation=dict(preparation=outcome,contexts=outcome['contexts'],results=[]))
            elif phase=='PROFILE':
                outcome=profile.prepare_profile(journal,cursor['parent_key'],cursor['context'],store,
                    executable=initial['review_executable'],free_disk_bytes=free)
                if outcome['status']=='SOURCE_PREPARATION_REQUIRED':
                    require('confirmation' not in cursor,'FROZEN_CONFIRMATION_SOURCE_CHANGE')
                    require('source_preparation' not in cursor,'SOURCE_CHANGE_DID_NOT_RESOLVE')
                    cursor['phase']='SOURCE'
                else:
                    require(outcome['status']=='PROFILE_AND_STORAGE_PLANS_READY_NOT_PROVISIONED','PROFILE_STATUS')
                    cursor.update(phase='MATERIALIZE',materialization_inputs=outcome['inputs'])
            elif phase=='SOURCE':
                # The source helper freezes a charged snapshot, but cannot grant
                # compatibility to changed source or mutate this controller.
                outcome=source.prepare_source(journal,cursor['parent_key'],store,
                    executable=initial['review_executable'],free_disk_bytes=free)
                snapshot=outcome['snapshot'];snapshot_event=journal.read()['events'][snapshot['event_key']]
                snapshot_pin=snapshot_event['receipt']
                require(json.loads(checked(snapshot_pin).read_text())==snapshot,'SNAPSHOT_RESULT')
                cursor.update(phase='CANDIDATE_PREPARATION' if 'candidate_design' in outcome['preparation'] else 'TRAINING_PREPARATION' if 'training_design' in outcome['preparation'] else 'COMPATIBILITY',
                    source_preparation=outcome,snapshot_receipt=snapshot_pin)
            elif phase=='TRAINING_PREPARATION':
                outcome=training.prepare(journal,cursor['parent_key'],cursor['snapshot_receipt'],cursor['context'],
                    cursor['source_preparation']['preparation'],store,executable=initial['review_executable'],free_disk_bytes=free)
                require(outcome['status']=='TRAINING_REVIEWS_ACCEPTED_CONTEXT_READY','TRAINING_PREPARATION_STATUS')
                cursor.update(phase='PROFILE',context=outcome['context'])
            elif phase=='CANDIDATE_PREPARATION':
                outcome=candidate.prepare(journal,cursor['parent_key'],cursor['snapshot_receipt'],cursor['context'],
                    cursor['source_preparation']['preparation'],store,executable=initial['review_executable'],free_disk_bytes=free)
                require(outcome['status']=='CANDIDATE_REVIEWS_ACCEPTED_CONTEXT_READY','CANDIDATE_PREPARATION_STATUS')
                cursor.update(phase='PROFILE',context=outcome['context'])
            elif phase=='COMPATIBILITY':
                outcome=compatibility.prepare(journal,cursor['parent_key'],cursor['snapshot_receipt'],
                    cursor['context'],store,executable=initial['review_executable'],free_disk_bytes=free)
                require(outcome['status']=='EXACT_COMPATIBILITY_ACCEPTED_CONTEXT_READY','COMPATIBILITY_STATUS')
                cursor.update(phase='PROFILE',context=outcome['context'])
            elif phase=='MATERIALIZE':
                outcome=materializer.materialize(journal,cursor['materialization_inputs'],store,free_disk_bytes=free)
                if outcome['status']=='WAITING_FOR_ADMIN_PROVISION':
                    save(journal,key,cursor,status='WAITING_FOR_ADMIN_PROVISION')
                    return dict(status='WAITING_FOR_ADMIN_PROVISION',event_key=key,
                        materialization_event_key=outcome['event_key'],missing=outcome['missing'],goal_complete=False)
                require(outcome['status']=='EXACT_INPUTS_MATERIALIZED_NOT_AUTHORIZED','MATERIALIZATION_STATUS')
                cursor.update(phase='NEXT_JOB',plan=outcome['artifacts']['plan'])
            else:raise RecoveryError('CONTROLLER_UNKNOWN_PHASE')
            save(journal,key,cursor)
        except Exception as error:
            # Preserve this exact phase; do not create a fresh worker/attempt.
            original=journal.read()['events'][key]['cursor']
            save(journal,key,original,status='INCOMPLETE_EXCEPTION',error=error)
            raise


def main():
    from rlflow.store import Store
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('request',type=Path)
    parser.add_argument('--request-sha256',required=True);parser.add_argument('--store',required=True,type=Path)
    args=parser.parse_args();root=args.store.resolve(strict=True)
    require((root/'runtime/research.db').is_file(),'STORE_REQUIRED')
    journal=Journal(args.directory.resolve())
    with journal.writer():
        result=run(journal,dict(path=str(args.request.resolve()),sha256=args.request_sha256),
                   Store(root),free_disk_bytes=shutil.disk_usage(journal.directory).free)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
