"""Consume one reviewed final batch in fixed order with no per-model decisions."""
import json
from rlflow.store import sha
from tools.r6_recovery import checked, file_pin, RecoveryError
from tools import r6_final_plans_review as review
from tools import r6_issue_final as issuer
from tools import r6_run_job as runner
from tools import r6_final_collection as collection
from tools import r6_profile_preparation as profile

ROLE='final-batch-sequence'
CAP=256*1024


def require(value,reason):
    if not value:raise RecoveryError('FINAL_SEQUENCE_'+reason)


def advance(journal,materializations_pin,review_key,review_pin,master_pin,store,*,free_disk_bytes):
    journal.require_writer()
    state=journal.read()
    request=dict(materializations=materializations_pin,review_key=review_key,review=review_pin,master=master_pin,
        store_root=str(store.root.resolve()),implementation=[file_pin(m.__file__) for m in (review,issuer,runner,collection)]+[file_pin(__file__)])
    name='FINAL_SEQUENCE:'+materializations_pin['sha256']
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-BATCH',input_sha256=sha(request),role=ROLE)
    if prior:
        key,entry=prior[0];require(entry['identity']==event and entry['request']==request,'INPUT_DRIFT')
    else:
        profile.guard(journal);review.verify(journal,review_key,review_pin,materializations_pin,store)
        key,_=journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
            purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    aggregate=json.loads(checked(materializations_pin).read_text())
    rows=aggregate['assignments']
    require(len(rows) in (3,4) and [r['assignment_index'] for r in rows]==list(range(len(rows))),'ORDER')
    completed=journal.read()['events'][key].get('completed_jobs',[])
    require(len(completed)<=len(rows) and len(set(completed))==len(completed),'COMPLETED')
    for index,job_key in enumerate(completed):
        job=journal.canonical_job(job_key)
        require(job['identity']['job_id']==rows[index]['materialization']['job_id'],'COMPLETED_JOB')
    for index in range(len(completed),len(rows)):
        row=rows[index]['materialization'];jid=row['job_id']
        jobs=[(k,e) for k,e in journal.read()['events'].items() if e['kind']=='job' and e['identity']['job_id']==jid]
        require(len(jobs)<=1,'JOB_ALIAS')
        if not jobs:
            issuer.issue(journal,materializations_pin,index,review_key,review_pin,master_pin,store,free_disk_bytes=free_disk_bytes)
        else:
            # Before a new submission, review and frozen evidence remain mandatory.
            if not jobs[0][1].get('submission_claim'):
                profile.guard(journal);review.verify(journal,review_key,review_pin,materializations_pin,store)
                issuer.plans.assignment(journal,aggregate,index,store)
        plan=json.loads(checked(row['artifacts']['plan']).read_text())
        outcome=runner.run(journal,store=store,free_disk_bytes=free_disk_bytes,**plan)
        if outcome['status']=='RESOURCE_OBSERVATION_REJECTED':
            return dict(status=outcome['status'],event_key=key,assignment_index=index,scientific_acceptance=False)
        require(outcome['status']=='FINAL_JOB_ANALYSIS_RETURNED','RUNNER_STATUS')
        job=journal.canonical_job(outcome['event_key'])
        require(job['identity']['job_id']==jid,'RUNNER_JOB')
        completed.append(outcome['event_key'])
        state=journal.read();state['events'][key]['completed_jobs']=list(completed);journal.save(state)
    profile.guard(journal)
    result=collection.collect(journal,aggregate['batch'],completed,store)
    return dict(status='FINAL_BATCH_COLLECTED_REQUIRES_INDEPENDENT_AUDIT',event_key=key,
        completed_jobs=completed,collection=result,scientific_acceptance=False,dispatch_authorized=False)
