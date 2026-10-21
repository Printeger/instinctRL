"""Prepare all immutable final profiles from a released batch and one source review.

The caller supplies a charged read-only final snapshot. No model selection,
volume provisioning, job reservation or dispatch occurs in this preparation.
"""
import json
from rlflow.store import canonical,sha
from tools.r6_budget import reserve
from training_workloads.unified_h1.native_volume import MAXIMUM
from training_workloads.unified_h1.result_volume import CONTROL_ALLOWANCE
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact
from tools import r6_profile_preparation as profile
from tools import r6_final_compatibility_review as review
from tools import r6_final_profile as validator
from tools import r6_holdout_publication as publication
from tools import r6_final_job as job_guard

ROLE='final-batch-preparation'
CAP=512*1024
FIELDS={'batch','context','origin_configs','experiment_ids','wall_seconds','payload_bytes'}


def require(value,reason):
    if not value:raise RecoveryError('FINAL_PREPARATION_'+reason)


def prepare(journal,snapshot_pin,input_pin,store,*,executable,free_disk_bytes):
    profile.guard(journal);q=json.loads(checked(input_pin).read_text());require(set(q)==FIELDS,'FIELDS')
    batch=publication.validate_batch(journal,q['batch']);count=len(batch['assignments'])
    require(len(q['origin_configs'])==len(q['experiment_ids'])==count and len(set(q['experiment_ids']))==count,'COMPLETE_BATCH')
    for p in q['origin_configs']:checked(p)
    require(type(q['payload_bytes'])is int and q['payload_bytes']>0 and q['payload_bytes']%4096==0,'PAYLOAD')
    context=json.loads(checked(q['context']).read_text());require(set(context)==profile.CONTEXT,'CONTEXT')
    require(context['runner']['purpose']=='confirmation_delivery','PURPOSE')
    reference=json.loads(checked(context['baseline_profile']).read_text())
    freeze=json.loads(checked(batch['freeze']).read_text())
    require(reference['baseline_bindings']==freeze['evidence']['baseline_bindings']
        and reference['r6']['protocol']==batch['protocol'],'BASELINE_PROTOCOL')
    snapshot=json.loads(checked(snapshot_pin).read_text())
    require(snapshot.get('status')=='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY','SNAPSHOT')
    config=json.loads(checked(snapshot['config']).read_text())
    require(config.get('schema_version')=='unified-h1-r6-final-config.v1','FINAL_CONFIG')
    validator.verify_frozen(snapshot,config)
    require(type(q['wall_seconds'])is int and 1<=q['wall_seconds']<=43200,'WALL')
    protocol=json.loads(checked(batch['protocol']).read_text())
    transitions=32960+32*sum(b['steps'] for b in protocol['blocks'])
    resources=dict(live_starts=count,simulator_initializations=count,nontraining_transitions=count*transitions,
        live_wall_seconds=count*(q['wall_seconds']+190),allocated_gpu_seconds=count*(q['wall_seconds']+190),
        output_actual_plus_unreleased_reservations_bytes=count*(q['payload_bytes']+96*1024**2+MAXIMUM+CONTROL_ALLOWANCE))
    state=journal.read();request=dict(inputs=input_pin,snapshot=snapshot_pin,
        implementation=[file_pin(m.__file__) for m in (profile,review,validator,publication,job_guard)]+[file_pin(__file__)])
    name='FINAL_PREPARATION:'+q['batch']['sha256']
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-BATCH',input_sha256=sha(request),role=ROLE)
    if prior:
        key,entry=prior[0];require(entry['identity']==event and entry['request']==request,'INPUT_DRIFT')
    else:
        key,_=journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
            purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    folder=journal.directory/'final-preparation'/key
    def save(name,value):
        profile.guard(journal);path=folder/name;data=canonical(value)+b'\n'
        size=sum(p.stat().st_size for p in folder.iterdir() if p.name!=name) if folder.exists() else 0
        require(size+len(data)<=CAP,'OUTPUT_CAP');persist_exact(path,data);return file_pin(path)
    reserve(journal.read()['budget'],event_key=dict(event,role='budget-check-only'),resources=resources,
        purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    review_input=save('compatibility-input.json',dict(batch=q['batch'],origin_configs=q['origin_configs']))
    reviewed=review.prepare(journal,snapshot_pin,review_input,store,executable=executable,free_disk_bytes=free_disk_bytes)
    require(reviewed['status']=='EXACT_FINAL_COMPATIBILITY_REVIEWS_RECORDED_NOT_AUTHORIZED'
        and reviewed['dispatch_authorized']is False and set(reviewed['accepted'])=={str(i) for i in range(count)},'REVIEW')
    contexts=[]
    for i,assignment in enumerate(batch['assignments']):
        selection=dict(schema_version='r6-final-selection/1',batch=q['batch'],assignment_index=i)
        seed=20260908 if assignment['model_role']=='baseline' else freeze['evidence']['streams'][int(assignment['model_role'][-1])-1]['training_seed']
        draft={k:reference[k] for k in ('master_authority','baseline_bindings','extension_inputs','group','evaluation_seed','protocol_review')}
        draft.update(experiment_id=q['experiment_ids'][i],wall_seconds=q['wall_seconds'],training_seed=seed,
            final_selection=selection,checkpoint_compatibility=reviewed['accepted'][str(i)],
            r6=dict(reference['r6'],run_id=q['experiment_ids'][i],seed=assignment['evaluation_seed'],split='final'))
        job_guard.validate(journal,draft,'confirmation_delivery')
        validator.validate(snapshot_pin,draft,q['payload_bytes'])
        old=reference.get('checkpoint_compatibility',{})
        new_context=dict(context,snapshot_receipt=snapshot_pin,baseline_profile=save(str(i)+'-profile.json',draft),
            conditions=[c for c in context['conditions'] if c['sha256']!=old.get('sha256')])
        contexts.append(save(str(i)+'-context.json',new_context))
    # Check the entire remaining final job batch without reserving or refunding it.
    reserve(journal.read()['budget'],event_key=dict(event,role='budget-check-only'),resources=resources,
        purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    result=dict(status='FINAL_BATCH_CONTEXTS_READY_NOT_RESERVED',event_key=key,batch=q['batch'],snapshot=snapshot_pin,
        contexts=contexts,payload_bytes=q['payload_bytes'],compatibility_review=reviewed,
        planned_job_resources=resources,resource_scope='FINAL_JOBS_ONLY_NOT_REVIEW_DELIVERY_OR_REPAIR',
        scientific_acceptance=False,dispatch_authorized=False)
    pin=save('result.json',result);state=journal.read();state['events'][key]['result']=pin;journal.save(state)
    return result
