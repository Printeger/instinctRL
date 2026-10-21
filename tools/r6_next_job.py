"""Consume one reviewed, already Store-authorized next-job plan.

Adds a durable edge to the existing exact-job runner, not a new scheduler.
Does not issue Store authority or turn a scientific proposal into approval.
"""
import json
import hashlib
from pathlib import Path

from rlflow import experiments
from rlflow.store import sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_batch import verified_decision
from tools.r6_decision import record as record_decision
from tools.r6_job_handoff import no_model_workers
from tools.r6_postrun import stop_paths
from tools.r6_budget import admissible
from tools.r6_run_job import run as run_job

ROLE='next-job-handoff'
CAP=65536
SCHEMA=Path(__file__).resolve().parents[1]/'docs/experiments/r6-robustness-auto/prompts/next-job-review.schema.json'
FIELDS={'spec_pin','query_executable','decision_executable','queue_review','sampler_review','purpose','sandbox'}


def require(value,reason):
    if not value:raise RecoveryError('NEXT_JOB_'+reason)


def reviewed_plan(journal,parent_key,plan_pin,review_key,review_pin,store):
    parent=record_decision(journal,parent_key,store)
    state=journal.read();decision_pin=state['events'][parent_key]['decision_record']
    require(parent['decision']['decision']!='REQUEST_FINAL_REVIEW','FINAL_REVIEW_IS_NOT_JOB')
    plan=json.loads(checked(plan_pin).read_text())
    require(set(plan)==FIELDS and plan['purpose'] in {'development','confirmation_delivery','repair'}
        and plan['sandbox'] in {'read-only','danger-full-access'},'PLAN_FIELDS')
    review=state['events'][review_key];identity=review['identity']
    require(review['kind']=='model' and identity['role']=='next-job-review'
        and identity['job_id']==state['events'][parent_key]['identity']['job_id']
        and identity['goal_id']==state['goal_id'] and identity['input_sha256']==sha(review['request'])
        and sha(identity)==review_key,'REVIEW_IDENTITY')
    require(review['request'].get('schema')==file_pin(SCHEMA)
        and review['request'].get('sandbox')=='read-only','REVIEW_CONTRACT')
    receipt=json.loads(checked(review['receipt']).read_text())
    require(review['status']=='RECEIPT_RECORDED' and receipt['event_key']==review_key
        and receipt['request_sha256']==sha(review['request']) and receipt['process']==review.get('process')
        and receipt['last_message']['sha256']==review_pin['sha256'],'REVIEW_RECEIPT')
    rows=[r for r in state['budget']['reservations'] if r['identity']==identity]
    require(len(rows)==1 and rows[0]['resources']==review['resources']
        and rows[0]['purpose']==review['purpose'] and review['resources'].get('automatic_model_sessions')==1,
        'REVIEW_RESERVATION')
    result=verified_decision(journal,review_key)
    require(result==json.loads(checked(review_pin).read_text())
        and result.get('decision')=='PASS_R6_NEXT_JOB' and result.get('must_fix')==[]
        and result.get('parent_decision_sha256')==decision_pin['sha256']
        and result.get('plan_sha256')==plan_pin['sha256'],'REVIEW_BINDING')
    require(result.get('changed_files')==[] and decision_pin in result.get('evidence_pins',[])
        and plan_pin in result.get('evidence_pins',[]),'REVIEW_EVIDENCE')
    for pin in result['evidence_pins']:checked(pin)
    spec=json.loads(checked(plan['spec_pin']).read_text())
    child='J-'+sha({'id':spec['id'],'spec':spec})[:16]
    require(child!=state['events'][parent_key]['identity']['job_id'],'SELF_JOB')
    return plan,spec,child


def register(journal,parent_key,plan_pin,review_key,review_pin,store,*,free_disk_bytes):
    journal.require_writer()
    request=dict(parent_key=parent_key,plan=plan_pin,review_key=review_key,review=review_pin)
    state=journal.read()
    prior=[(k,e) for k,e in state['events'].items()
           if e['identity']['role']==ROLE and e['request'].get('parent_key')==parent_key]
    require(len(prior)<=1,'PARENT_ALIAS')
    if prior:
        key,entry=prior[0]
        require(entry['request']==request,'PARENT_ALREADY_BOUND')
        expected=dict(goal_id=state['goal_id'],event_id='NEXT_JOB:'+parent_key,
            job_id=entry['identity']['job_id'],input_sha256=sha(request),role=ROLE)
        rows=[r for r in state['budget']['reservations'] if r['identity']==expected]
        require(entry['kind']=='cpu' and entry['identity']==expected and sha(expected)==key
            and entry['resources']=={'output_actual_plus_unreleased_reservations_bytes':CAP}
            and len(rows)==1 and rows[0]['resources']==entry['resources']
            and rows[0]['purpose']==entry['purpose'],'REPLAY_IDENTITY')
        # Preserve exact recovery under STOP; runner distinguishes an existing
        # submission claim from a new action and performs original cleanup.
        return key
    journal.guard();require(not stop_paths(journal),'STOP');no_model_workers(journal)
    require(state.get('current_job') is None,'PARENT_NOT_CLOSED')
    admissible(state['budget'])
    plan,spec,child=reviewed_plan(journal,parent_key,plan_pin,review_key,review_pin,store)
    experiments.validate(spec,store.root);experiments.authorized(store.replay(),spec,store.root)
    require(child not in store.replay()['jobs'],'CHILD_PREEXISTS')
    require(not any(e['identity']['role']==ROLE and e['identity']['job_id']==child
                    for e in state['events'].values()),'CHILD_ALREADY_BOUND')
    event=dict(goal_id=state['goal_id'],event_id='NEXT_JOB:'+parent_key,job_id=child,
               input_sha256=sha(request),role=ROLE)
    key,_=journal.intent(event,kind='cpu',request=request,
        resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose=plan['purpose'],
        free_disk_bytes=free_disk_bytes)
    return key


def advance(journal,parent_key,plan_pin,review_key,review_pin,store,*,free_disk_bytes):
    key=register(journal,parent_key,plan_pin,review_key,review_pin,store,free_disk_bytes=free_disk_bytes)
    state=journal.read();entry=state['events'][key]
    plan=json.loads(checked(entry['request']['plan']).read_text())
    require(set(plan)==FIELDS and plan['purpose']==entry['purpose'],'PLAN_DRIFT')
    spec=json.loads(checked(plan['spec_pin']).read_text())
    require('J-'+sha({'id':spec['id'],'spec':spec})[:16]==entry['identity']['job_id'],'CHILD_DRIFT')
    if entry.get('next_job_result'):
        result=entry['next_job_result']
        require(result.get('status')=='DECISION_PROPOSAL_RETURNED'
            and result.get('handoff_event_key')==key
            and result.get('scientific_acceptance') is False,'CACHED_RESULT')
        job=journal.canonical_job(result['child_job_event_key'])
        require(job['identity']['job_id']==entry['identity']['job_id'],'CACHED_CHILD')
        return result
    jobs=[e for e in state['events'].values() if e['kind']=='job'
          and e['identity']['job_id']==entry['identity']['job_id']]
    require(len(jobs)<=1,'CHILD_JOB_ALIAS')
    if not jobs or not jobs[0].get('submission_claim'):
        journal.guard();require(not stop_paths(journal),'STOP');no_model_workers(journal)
        # An edge registered before a crash does not preserve permission to
        # start after its review or parent evidence has changed.
        reviewed_plan(journal,parent_key,plan_pin,review_key,review_pin,store)
    outcome=run_job(journal,store=store,free_disk_bytes=free_disk_bytes,**plan)
    result=dict(status=outcome['status'],handoff_event_key=key,
                child_job_event_key=outcome.get('event_key'),scientific_acceptance=False)
    require(result['status'] in {'RESOURCE_OBSERVATION_REJECTED','DECISION_PROPOSAL_RETURNED'},'RUNNER_STATUS')
    if result['status']=='DECISION_PROPOSAL_RETURNED':
        job=journal.canonical_job(result['child_job_event_key'])
        require(job['identity']['job_id']==entry['identity']['job_id'],'RUNNER_CHILD')
        state=journal.read();state['events'][key]['next_job_result']=result;journal.save(state)
    return result


def register_review(journal,parent_key,plan_pin,store,*,executable,free_disk_bytes):
    """Register one bounded read-only review; existing model_handoff executes it.

    No Store permission is issued here. The plan and parent remain exact pins.
    Registration is durable before writing the prompt and never launches a model.
    """
    from tools.r6_decision import CLI,CLI_SHA,MODEL_CAP,ROOT
    from tools.r6_decision_packet import persist_exact
    from tools.r6_model_worker import validate_request
    journal.require_writer();journal.guard()
    require(not stop_paths(journal),'STOP');no_model_workers(journal)
    state=journal.read();require(state.get('current_job') is None,'PARENT_NOT_CLOSED')
    record_decision(journal,parent_key,store)
    state=journal.read();parent=state['events'][parent_key]['decision_record']
    plan=json.loads(checked(plan_pin).read_text())
    require(set(plan)==FIELDS and plan['purpose'] in {'development','repair','confirmation_delivery'}
        and plan['sandbox'] in {'read-only','danger-full-access'},'PLAN_FIELDS')
    spec=json.loads(checked(plan['spec_pin']).read_text())
    experiments.validate(spec,store.root)
    checked(executable)
    if state['goal_id']=='R6-CPU-TEST-FIXTURE':
        require(checked(executable).is_relative_to(journal.directory.parent.resolve()),'FIXTURE_EXECUTABLE')
    else:require(executable=={'path':str(CLI),'sha256':CLI_SHA},'PINNED_CLI')
    packet=dict(parent=parent,plan=plan_pin,spec=plan['spec_pin'])
    prompt=('Review this exact R6 next-job plan independently, read-only, then exit. '
        'Read the pinned parent decision, plan, spec and necessary original evidence. '
        'Check that the plan implements the parent decision, preserves frozen metrics, actor/runtime '
        'boundaries, required comparisons and full confirmation budget. Check exact Store authority '
        'and resource feasibility; missing evidence is MUST_FIX. Do not launch jobs, query GPUs, '
        'change files, grant authority, or wait for training. PASS requires no must_fix. '
        'Return parent_decision_sha256, plan_sha256 and both pins in evidence_pins. '
        'changed_files must be empty. Store root: '+str(store.root)+'\n'+json.dumps(packet,sort_keys=True)+'\n').encode()
    require(len(prompt)<=CAP,'REVIEW_CONTEXT_CAP')
    path=journal.directory/'next-job-reviews'/parent_key/plan_pin['sha256']/'prompt.txt'
    request=dict(prompt={'path':str(path.resolve()),'sha256':hashlib.sha256(prompt).hexdigest()},
        schema=file_pin(SCHEMA),executable=executable,cwd=str(ROOT),sandbox='read-only',
        timeout_seconds=3600,output_cap_bytes=MODEL_CAP,
        stop_paths=[str((journal.directory/'STOP').resolve()),str((journal.directory.parent/'STOP').resolve())])
    identity=dict(goal_id=state['goal_id'],event_id='NEXT_JOB_REVIEW:'+parent_key+':'+plan_pin['sha256'],
        job_id=state['events'][parent_key]['identity']['job_id'],input_sha256=sha(request),role='next-job-review')
    prior=[e for e in state['events'].values() if e['identity']['role']=='next-job-review'
        and e['identity']['event_id']==identity['event_id']]
    require(all(e['request']==request for e in prior),'REVIEW_ALREADY_BOUND')
    key,_=journal.intent(identity,kind='model',request=request,
        resources={'automatic_model_sessions':1,'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+CAP},
        purpose=plan['purpose'],free_disk_bytes=free_disk_bytes)
    persist_exact(path,prompt);validate_request(request)
    return key
