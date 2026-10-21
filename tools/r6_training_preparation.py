"""Bind committed training design to frozen source, reviews and a profile context."""
import json
from pathlib import Path
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact
from tools import r6_profile_preparation as profile
from tools import r6_training_review as review
from tools import r6_training_profile as validator
from tools import r6_source_preparation as source

ROLE='training-preparation'
CAP=256*1024


def require(value,reason):
    if not value:raise RecoveryError('TRAINING_PREPARATION_'+reason)


def prepare(journal,parent_key,snapshot_pin,context_pin,source_result,store,*,executable,free_disk_bytes):
    journal.require_writer();state=journal.read()
    sources=[(k,e) for k,e in state['events'].items() if e['identity'].get('role')==source.ROLE
        and e.get('source_input',{}).get('parent_key')==parent_key and e.get('source_result')]
    require(len(sources)==1,'SOURCE_EVENT')
    source_key,entry=sources[0]
    require(json.loads(checked(entry['source_result']).read_text())==source_result
        and source_result['worker_receipt']==entry['receipt'],'SOURCE_RESULT')
    profile.reservation(state,source_key,'model',entry['resources'],entry['purpose'])
    snapshot=json.loads(checked(snapshot_pin).read_text())
    require(snapshot.get('status')=='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY'
        and snapshot['source_identity']['workload_commit']==source_result['source_commit'],'SNAPSHOT')
    context=json.loads(checked(context_pin).read_text());require(set(context)==profile.CONTEXT,'CONTEXT')
    reference=json.loads(checked(context['baseline_profile']).read_text())
    design=json.loads(checked(source_result['training_design']).read_text())
    require(set(design)=={'schema_version','recipe_template','plan_template','origin_config'}
        and design['schema_version']=='r6-training-design/1','DESIGN')
    purpose=context['runner']['purpose']
    require(design['plan_template']['phase']!='confirmation' or purpose=='confirmation_delivery','CONFIRMATION_RESERVE')
    request=dict(parent_key=parent_key,snapshot=snapshot_pin,context=context_pin,source=entry['source_result'],
        implementation=[file_pin(m.__file__) for m in (profile,review,validator)]+[file_pin(__file__)])
    event=dict(goal_id=state['goal_id'],event_id='TRAINING_PREPARATION:'+parent_key+':'+sha(request),
        job_id=entry['identity']['job_id'],input_sha256=sha(request),role=ROLE)
    key=sha(event)
    if key not in state['events']:
        profile.guard(journal)
        journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
            purpose=purpose,free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},purpose)
    folder=journal.directory/'training-preparation'/key
    def save(name,value):
        data=canonical(value)+b'\n';path=folder/name
        size=sum(p.stat().st_size for p in folder.iterdir() if p.name!=name) if folder.exists() else 0
        require(size+len(data)<=CAP,'OUTPUT_CAP')
        if not path.exists():profile.guard(journal)
        persist_exact(path,data);return file_pin(path)
    require('source_identity' not in design['recipe_template'] and not
        {'recipe','recipe_review'}.intersection(design['plan_template']),'CONTROLLER_BINDINGS')
    recipe=save('recipe.json',dict(design['recipe_template'],source_identity=snapshot['source_identity']))
    first_input=save('recipe-review-input.json',dict(kind='recipe',recipe=recipe))
    cached=journal.read()['events'][key].get('recipe_review_result')
    if cached:
        first=json.loads(checked(cached).read_text())
        review.verified_decision(journal,first['event_key'])
        checked(first['reviews']['recipe'])
    else:
        first=review.prepare(journal,parent_key,snapshot_pin,first_input,store,
            executable=executable,free_disk_bytes=free_disk_bytes,purpose=purpose)
        require(first['status']=='EXACT_TRAINING_REVIEWS_RECORDED_NOT_AUTHORIZED'
            and not first['dispatch_authorized'],'RECIPE_REVIEW')
        result_pin=save('recipe-review-result.json',first)
        state=journal.read();state['events'][key]['recipe_review_result']=result_pin;journal.save(state)
    require(first['payloads']['recipe']==recipe,'RECIPE_BINDING')
    plan=dict(design['plan_template'],recipe=recipe,recipe_review=first['reviews']['recipe'])
    plan_pin=save('plan.json',plan)
    second_input=save('stage-review-input.json',dict(kind='stage',plan=plan_pin,origin_config=design['origin_config']))
    second=review.prepare(journal,parent_key,snapshot_pin,second_input,store,
        executable=executable,free_disk_bytes=free_disk_bytes,purpose=purpose)
    require(second['status']=='EXACT_TRAINING_REVIEWS_RECORDED_NOT_AUTHORIZED' and not second['dispatch_authorized'],'STAGE_REVIEW')
    draft={k:reference[k] for k in ('master_authority','baseline_bindings','extension_inputs','group','evaluation_seed')}
    draft.update(experiment_id=plan['experiment_id'],training_seed=plan['training_seed'],r6_training_plan=plan,
        r6_expected_mass_kg=1.181517923,capacity_review=second['accepted']['capacity'],
        checkpoint_compatibility=second['accepted']['compatibility'])
    validator.validate(snapshot_pin,draft);profile.guard(journal)
    old=reference.get('checkpoint_compatibility',{})
    new_context=dict(context,snapshot_receipt=snapshot_pin,baseline_profile=save('profile-reference.json',draft),
        conditions=[c for c in context['conditions'] if c['sha256']!=old.get('sha256')])
    return dict(status='TRAINING_REVIEWS_ACCEPTED_CONTEXT_READY',context=save('context.json',new_context),
        recipe_review=first,stage_review=second,dispatch_authorized=False)
