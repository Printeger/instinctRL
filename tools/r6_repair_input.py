"""Immutable bounded input for engineering repair after a validated decision.

NEEDS_REPAIR cannot grant a new profile or live attempt. This input preserves the
closed job, failed receipts and exact controller cursor for a repair worker.
"""
import json
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision import record as decision_record
from tools.r6_decision_packet import persist_exact
from tools import r6_profile_preparation as profile

ROLE='engineering-repair-input'
CAP=1024*1024


def require(ok,reason):
    if not ok:raise RecoveryError('REPAIR_INPUT_'+reason)


def prepare(journal,controller_key,store,*,free_disk_bytes):
    profile.guard(journal);state=journal.read();controller=state['events'][controller_key];cursor=controller['cursor']
    require(controller['identity']['role']=='r6-controller'
        and controller['request']['store_root']==str(store.root.resolve())
        and cursor['phase']=='REPAIR_AND_REVIEW','CONTROLLER')
    parent=cursor['parent_key'];decision=decision_record(journal,parent,store)
    require(decision['decision']['decision']=='NEEDS_REPAIR' and bool(decision['decision']['must_fix']),'DECISION')
    event=state['events'][parent];job=journal.canonical_job(cursor['job_key'],state=state)
    require(event['decision_input']['job_event_key']==cursor['job_key']
        and event['identity']['job_id']==job['identity']['job_id'],'PARENT_JOB')
    require(job.get('closure') and job.get('postrun') and job.get('analysis'),'CLOSED_JOB_EVIDENCE')
    evidence={name:job[name] for name in ('closure','inventory','postrun','analysis') if job.get(name)}
    for pin in evidence.values():checked(pin)
    receipt=event['receipt'];checked(receipt)
    related=[]
    for key,e in sorted(state['events'].items()):
        if e['identity'].get('job_id')!=job['identity']['job_id'] or e['identity'].get('role')==ROLE:continue
        row=dict(event_key=key,kind=e['kind'],identity=e['identity'],status=e.get('status','UNKNOWN'),
            receipt=e.get('receipt'),submission_claim=e.get('submission_claim'),dispatch_claim=e.get('dispatch_claim'))
        if row['receipt']:checked(row['receipt'])
        related.append(row)
    description=dict(controller_key=controller_key,cursor=cursor,job_key=cursor['job_key'],
        parent_key=parent,decision=decision,decision_receipt=receipt,job_spec=job['request']['spec'],
        closed_job_evidence=evidence,related_events=related,store_root=str(store.root.resolve()),
        store_sha256=sha(store.replay()),implementation=file_pin(__file__))
    require(len(canonical(description))<CAP//3,'INPUT_CAP')
    name='ENGINEERING_REPAIR_INPUT:'+sha(description)
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    if prior:
        key,entry=prior[0];request=entry['request'];require(request['description']==description,'DRIFT')
    else:
        request=dict(description=description,budget_before=state['budget'])
        require(len(canonical(request))<CAP//2,'BUDGET_CAP')
        identity=dict(goal_id=state['goal_id'],event_id=name,job_id=job['identity']['job_id'],input_sha256=sha(request),role=ROLE)
        key,_=journal.intent(identity,kind='cpu',request=request,
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='repair',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'repair')
    result=dict(status='ENGINEERING_REPAIR_INPUT_READY',event_key=key,description=description,
        required_disposition=['OFFLINE_EVIDENCE_REPAIR','NEW_REVIEWED_LIVE_ATTEMPT','EXTERNAL_BLOCKER_WITH_EVIDENCE'],
        constraints=['Preserve all failed receipts and counters; no implicit refund.',
            'No profile or new live attempt until exact repaired evidence and fresh review are bound.',
            'Control code changes require independently reviewed controller revision.',
            'A missing receipt or unresolved ownership is not permission to retry.'],
        scientific_acceptance=False,dispatch_authorized=False,goal_complete=False)
    data=canonical(result)+b'\n';require(len(data)+len(canonical(request))<CAP,'OUTPUT_CAP')
    path=journal.directory/'engineering-repairs'/key/'input.json'
    profile.guard(journal);persist_exact(path,data);pin=file_pin(path)
    state=journal.read();state['events'][key]['repair_input']=pin;journal.save(state)
    return dict(status=result['status'],event_key=key,repair_input=pin,dispatch_authorized=False,goal_complete=False)


def validate(journal,pin,parent_key,store):
    """Recheck immutable repair evidence after a worker has added new events."""
    value=json.loads(checked(pin).read_text());require(value['status']=='ENGINEERING_REPAIR_INPUT_READY','STATUS')
    state=journal.read();key=value['event_key'];event=state['events'][key];desc=value['description']
    profile.reservation(state,key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'repair')
    require(event['identity']['role']==ROLE and event.get('repair_input')==pin
        and event['request']['description']==desc and desc['parent_key']==parent_key
        and desc['store_root']==str(store.root.resolve()) and desc['store_sha256']==sha(store.replay()),'BINDING')
    checked(desc['implementation'])
    decision=decision_record(journal,parent_key,store)
    require(decision==desc['decision'] and decision['decision']['decision']=='NEEDS_REPAIR'
        and state['events'][parent_key]['receipt']==desc['decision_receipt'],'PARENT')
    job=journal.canonical_job(desc['job_key'],state=state)
    require(job['request']['spec']==desc['job_spec'] and state['events'][parent_key]['decision_input']['job_event_key']==desc['job_key'],'JOB')
    checked(desc['job_spec']);checked(desc['decision_receipt'])
    for name,evidence in desc['closed_job_evidence'].items():
        require(job.get(name)==evidence,'EVIDENCE_DRIFT');checked(evidence)
    for related in desc['related_events']:
        if related['receipt']:checked(related['receipt'])
    return value
