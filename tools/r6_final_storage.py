"""Controller-owned final profile to prepaid dual-volume/materialization inputs.

No model call or profile editing, mounting, Store authority or job submission.
"""
import json
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact
from tools import r6_final_preparation as preparation
from tools import r6_profile_preparation as profile
from tools import r6_final_profile as validator
from tools import r6_final_job as final_job
from tools import r6_storage_allocation as storage

ROLE='final-storage-preparation'
CAP=128*1024


def require(value,reason):
    if not value:raise RecoveryError('FINAL_STORAGE_'+reason)


def prepared(journal,pin,index):
    result=json.loads(checked(pin).read_text());state=journal.read()
    require(result['status']=='FINAL_BATCH_CONTEXTS_READY_NOT_RESERVED'
        and result['scientific_acceptance']is False and result['dispatch_authorized']is False,'RESULT')
    key=result['event_key'];entry=state['events'][key];request=entry['request']
    require(entry['identity']['role']==preparation.ROLE and entry['result']==pin
        and entry['identity']['input_sha256']==sha(request),'REGISTERED_PREPARATION')
    profile.reservation(state,key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':preparation.CAP},'confirmation_delivery')
    for tool in request['implementation']:checked(tool)
    require(file_pin(preparation.__file__) in request['implementation'],'IMPLEMENTATION')
    q=json.loads(checked(request['inputs']).read_text())
    require(result['batch']==q['batch'] and result['snapshot']==request['snapshot']
        and result['payload_bytes']==q['payload_bytes'] and len(result['contexts'])==len(q['experiment_ids']),'INPUT_BINDING')
    require(type(index)is int and 0<=index<len(result['contexts']),'INDEX')
    reviewed=result['compatibility_review'];reviewer=preparation.review
    review_event=state['events'][reviewed['event_key']]
    require(review_event['identity']['role']==reviewer.ROLE,'REVIEW_ROLE')
    profile.reservation(state,reviewed['event_key'],'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':reviewer.MODEL_CAP+65536},'confirmation_delivery')
    original=reviewer.verified_decision(journal,reviewed['event_key'])
    require(original['decision']=='PASS_R6_FINAL_COMPATIBILITY_REVIEW' and not original['must_fix']
        and original['changed_files']==[],'REVIEW_DECISION')
    item=json.loads(checked(reviewed['reviews'][str(index)]).read_text())
    receipt=json.loads(checked(review_event['receipt']).read_text())
    require(item['parent_review']==receipt['last_message'] and item['worker_receipt']==review_event['receipt']
        and {k:v for k,v in item.items() if k not in ('parent_review','worker_receipt')} in original['reviews']
        and item['kind']==str(index) and item['decision']=='PASS_R6_FINAL_COMPATIBILITY' and not item['must_fix'],'ITEM_RECEIPT')
    payload=json.loads(checked(reviewed['payloads'][str(index)]).read_text())
    accepted=json.loads(checked(reviewed['accepted'][str(index)]).read_text())
    require(item['payload_sha256']==sha(payload) and accepted==dict(payload,independent_review=reviewed['reviews'][str(index)]),'ACCEPTED_COMPATIBILITY')
    context=json.loads(checked(result['contexts'][index]).read_text())
    require(set(context)==profile.CONTEXT and context['snapshot_receipt']==result['snapshot']
        and context['runner']['purpose']=='confirmation_delivery','CONTEXT')
    draft=json.loads(checked(context['baseline_profile']).read_text())
    require(draft['experiment_id']==q['experiment_ids'][index] and draft['wall_seconds']==q['wall_seconds']
        and draft['final_selection']==dict(schema_version='r6-final-selection/1',batch=result['batch'],assignment_index=index)
        and draft['checkpoint_compatibility']==reviewed['accepted'][str(index)],'EXACT_PROFILE')
    final_job.validate(journal,draft,'confirmation_delivery')
    validator.validate(result['snapshot'],draft,result['payload_bytes'])
    return result,context,draft


def prepare(journal,preparation_pin,index,*,minimum_free_inodes=512,free_disk_bytes):
    profile.guard(journal)
    result,context,draft=prepared(journal,preparation_pin,index)
    require(type(minimum_free_inodes)is int and 1<=minimum_free_inodes<=(result['payload_bytes']+96*1024**2)//262144,'INODES')
    state=journal.read();request=dict(preparation=preparation_pin,assignment_index=index,minimum_free_inodes=minimum_free_inodes,
        implementation=[file_pin(m.__file__) for m in (preparation,validator,final_job,storage)]+[file_pin(__file__)])
    name='FINAL_STORAGE:'+result['event_key']+':'+str(index)
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-STORAGE:'+draft['experiment_id'],input_sha256=sha(request),role=ROLE)
    if prior:
        key,entry=prior[0];require(entry['identity']==event and entry['request']==request,'INPUT_DRIFT')
    else:
        key,_=journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
            purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    folder=journal.directory/'final-storage'/key
    def save(name,value):
        profile.guard(journal);path=folder/name;data=canonical(value)+b'\n'
        size=sum(p.stat().st_size for p in folder.iterdir() if p.name!=name) if folder.exists() else 0
        require(size+len(data)<=CAP,'OUTPUT_CAP');persist_exact(path,data);return file_pin(path)
    volume_id='J-'+sha(dict(goal=state['goal_id'],preparation=preparation_pin,assignment_index=index))[:16]
    payload=result['payload_bytes'];native=storage.native_tool.plan(volume_id)
    target=storage.result_tool.ATTEMPTS/draft['experiment_id']/'results'
    result_plan=storage.result_tool.plan(volume_id,target,payload+96*1024**2,payload,minimum_free_inodes)
    np=save('native-plan.json',native);rp=save('result-plan.json',result_plan)
    allocation=storage.register(journal,draft['experiment_id'],np,rp,purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    inputs=dict(snapshot_receipt=context['snapshot_receipt'],profile_draft=context['baseline_profile'],storage_key=allocation,
        stage_id=context['stage_id'],conditions=context['conditions'],runner=context['runner'])
    output=dict(status='FINAL_PROFILE_STORAGE_READY_NOT_PROVISIONED',event_key=key,preparation=preparation_pin,
        assignment_index=index,storage_key=allocation,inputs=save('materialization-inputs.json',inputs),dispatch_authorized=False)
    pin=save('result.json',output);state=journal.read();state['events'][key]['result']=pin;journal.save(state)
    return output


def materialize_all(journal,preparation_pin,store,*,minimum_free_inodes=512,free_disk_bytes):
    """Prepare every exact plan or report all missing preapproved volumes."""
    from tools import r6_materialize_job as materializer
    profile.guard(journal);result=json.loads(checked(preparation_pin).read_text())
    rows=[];missing=[]
    for index in range(len(result['contexts'])):
        planned=prepare(journal,preparation_pin,index,minimum_free_inodes=minimum_free_inodes,free_disk_bytes=free_disk_bytes)
        materialized=materializer.materialize(journal,planned['inputs'],store,free_disk_bytes=free_disk_bytes)
        require(materialized['status'] in ('WAITING_FOR_ADMIN_PROVISION','EXACT_INPUTS_MATERIALIZED_NOT_AUTHORIZED'),
            'MATERIALIZATION_STATUS')
        require(materialized['dispatch_authorized']is False,'NO_DISPATCH')
        if materialized['status']=='WAITING_FOR_ADMIN_PROVISION':
            missing.append(dict(assignment_index=index,missing=materialized['missing']))
        rows.append(dict(assignment_index=index,storage=planned,materialization=materialized))
    return dict(status='WAITING_FOR_ADMIN_PROVISION' if missing else 'FINAL_BATCH_PLANS_READY_NOT_AUTHORIZED',
        preparation=preparation_pin,batch=result['batch'],assignments=rows,missing=missing,
        scientific_acceptance=False,dispatch_authorized=False)
