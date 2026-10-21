"""Persist reviewed confirmation plans and contexts without changing frozen source.

The caller supplies the already frozen training snapshot. This preparation
does not reserve training jobs, provision volumes, or grant dispatch authority.
"""
import json
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact
from tools import r6_profile_preparation as profile
from tools import r6_training_review as review
from tools import r6_training_profile as validator
from tools import r6_confirmation_plans as plans

ROLE='confirmation-preparation'
CAP=512*1024
FIELDS={'snapshot','context','recipe','recipe_review','selection_evidence','origin_config',
        'experiment_ids','wall_seconds','storage'}


def require(value,reason):
    if not value:raise RecoveryError('CONFIRMATION_PREPARATION_'+reason)


def prepare(journal,parent_key,input_pin,store,*,executable,free_disk_bytes):
    journal.require_writer()
    q=json.loads(checked(input_pin).read_text());require(set(q)==FIELDS,'FIELDS')
    for field in ('snapshot','context','recipe','recipe_review','selection_evidence','origin_config'):checked(q[field])
    parent=profile.parent_decision(journal,parent_key,store)
    require(parent['decision']['decision']=='PROPOSE_CONFIRMATION','PARENT_DECISION')
    context=json.loads(checked(q['context']).read_text());require(set(context)==profile.CONTEXT,'CONTEXT')
    require(context['runner']['purpose']=='confirmation_delivery','PURPOSE')
    reference=json.loads(checked(context['baseline_profile']).read_text())
    snapshot=json.loads(checked(q['snapshot']).read_text())
    recipe=json.loads(checked(q['recipe']).read_text())
    require(snapshot.get('status')=='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY'
        and snapshot['source_identity']==recipe['source_identity'],'FROZEN_RECIPE_SOURCE')
    state=journal.read()
    request=dict(parent=state['events'][parent_key]['decision_record'],inputs=input_pin,
        implementation=[file_pin(m.__file__) for m in (profile,review,validator,plans)]+[file_pin(__file__)])
    name='CONFIRMATION_PREPARATION:'+parent_key
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    event=dict(goal_id=state['goal_id'],event_id=name,job_id=state['events'][parent_key]['identity']['job_id'],
        input_sha256=sha(request),role=ROLE)
    if prior:
        key,entry=prior[0];require(entry['identity']==event and entry['request']==request,'INPUT_DRIFT')
    else:
        profile.guard(journal)
        key,_=journal.intent(event,kind='cpu',request=request,
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
            purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    folder=journal.directory/'confirmation-preparation'/key

    def save(name,value):
        path=folder/name;data=canonical(value)+b'\n'
        size=sum(p.stat().st_size for p in folder.iterdir() if p.name!=name) if folder.exists() else 0
        require(size+len(data)<=CAP,'OUTPUT_CAP')
        if not path.exists():profile.guard(journal)
        persist_exact(path,data);return file_pin(path)

    def reviewed(name,request):
        pin=save(name+'-input.json',request)
        cached=journal.read()['events'][key].get(name)
        if cached:
            result=json.loads(checked(cached).read_text())
            review.verified_decision(journal,result['event_key'])
            for group in ('payloads','reviews','accepted'):
                for value in result[group].values():checked(value)
        else:
            result=review.prepare(journal,parent_key,q['snapshot'],pin,store,executable=executable,
                free_disk_bytes=free_disk_bytes,purpose='confirmation_delivery')
            require(result['status']=='EXACT_TRAINING_REVIEWS_RECORDED_NOT_AUTHORIZED'
                and result['dispatch_authorized']is False,'REVIEW_STATUS')
            result_pin=save(name+'-result.json',result)
            state=journal.read();state['events'][key][name]=result_pin;journal.save(state)
        return result

    freeze=reviewed('freeze',dict(kind='freeze',recipe=q['recipe'],recipe_review=q['recipe_review'],
        selection_evidence=q['selection_evidence']))
    request=dict(freeze_pin=freeze['accepted']['freeze'],recipe_review=q['recipe_review'],
        baseline_bindings=reference['baseline_bindings'],experiment_ids=q['experiment_ids'],
        wall_seconds=q['wall_seconds'],storage=q['storage'])
    built=plans.prepare(q['snapshot'],request)
    require(built['status']=='THREE_CONFIRMATION_PLANS_PREPARED_NOT_RESERVED'
        and built['dispatch_authorized']is False,'PLAN_STATUS')
    bundle=save('plans.json',built)
    contexts=[]
    for index,plan in enumerate(built['bundle']['plans']):
        plan_pin=save(f'plan-{index}.json',plan)
        stage=reviewed(f'stage-{index}',dict(kind='stage',plan=plan_pin,origin_config=q['origin_config']))
        draft={k:reference[k] for k in ('master_authority','baseline_bindings','extension_inputs','group','evaluation_seed')}
        draft.update(experiment_id=plan['experiment_id'],training_seed=plan['training_seed'],r6_training_plan=plan,
            r6_expected_mass_kg=1.181517923,capacity_review=stage['accepted']['capacity'],
            checkpoint_compatibility=stage['accepted']['compatibility'])
        validator.validate(q['snapshot'],draft)
        old=reference.get('checkpoint_compatibility',{})
        new=dict(context,snapshot_receipt=q['snapshot'],baseline_profile=save(f'profile-{index}.json',draft),
            conditions=[c for c in context['conditions'] if c['sha256']!=old.get('sha256')])
        contexts.append(save(f'context-{index}.json',new))
    profile.guard(journal)
    result=dict(status='THREE_CONFIRMATION_CONTEXTS_READY_NOT_RESERVED',event_key=key,
        snapshot=q['snapshot'],freeze=freeze['accepted']['freeze'],plans=bundle,contexts=contexts,
        training_jobs_reserved=False,dispatch_authorized=False,scientific_acceptance=False)
    save('result.json',result)
    return result
