"""Bind committed candidate design to frozen source, reviews and a profile context."""
import json
from pathlib import Path
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact
from tools import r6_profile_preparation as profile
from tools import r6_candidate_review as review
from tools import r6_candidate_profile as validator
from tools import r6_source_preparation as source

ROLE='candidate-preparation'
CAP=256*1024


def require(value,reason):
    if not value:raise RecoveryError('CANDIDATE_PREPARATION_'+reason)


def validate_design(design):
    from training_workloads.unified_h1.robustness_evaluation_selection import FIELDS
    require(isinstance(design,dict) and set(design)=={'schema_version','selection_template','evaluation_template','origin_config','result_payload_bytes'}
        and design['schema_version']=='r6-candidate-design/1','DESIGN')
    require(isinstance(design['selection_template'],dict) and set(design['selection_template'])==FIELDS-{'independent_review'},'SELECTION_TEMPLATE')
    require(isinstance(design['evaluation_template'],dict) and set(design['evaluation_template'])=={'experiment_id','wall_seconds','protocol_review','r6'},'EVALUATION_TEMPLATE')
    require(type(design['result_payload_bytes'])is int and design['result_payload_bytes']>0 and design['result_payload_bytes']%4096==0,'PAYLOAD')
    checked(design['origin_config'])


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
    design=json.loads(checked(source_result['candidate_design']).read_text())
    validate_design(design)
    purpose=context['runner']['purpose']
    request=dict(parent_key=parent_key,snapshot=snapshot_pin,context=context_pin,source=entry['source_result'],
        implementation=[file_pin(m.__file__) for m in (profile,review,validator)]+[file_pin(__file__)])
    event=dict(goal_id=state['goal_id'],event_id='CANDIDATE_PREPARATION:'+parent_key+':'+sha(request),
        job_id=entry['identity']['job_id'],input_sha256=sha(request),role=ROLE)
    key=sha(event)
    if key not in state['events']:
        profile.guard(journal)
        journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
            purpose=purpose,free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},purpose)
    folder=journal.directory/'candidate-preparation'/key
    def save(name,value):
        data=canonical(value)+b'\n';path=folder/name
        size=sum(p.stat().st_size for p in folder.iterdir() if p.name!=name) if folder.exists() else 0
        require(size+len(data)<=CAP,'OUTPUT_CAP')
        if not path.exists():profile.guard(journal)
        persist_exact(path,data);return file_pin(path)
    selection_pin=save('selection-draft.json',design['selection_template'])
    first_input=save('selection-review-input.json',dict(kind='selection',selection=selection_pin))
    cached=journal.read()['events'][key].get('selection_review_result')
    if cached:
        first=json.loads(checked(cached).read_text())
        review.verified_decision(journal,first['event_key'])
        checked(first['reviews']['selection']);checked(first['accepted']['selection'])
    else:
        first=review.prepare(journal,parent_key,snapshot_pin,first_input,store,
            executable=executable,free_disk_bytes=free_disk_bytes,purpose=purpose)
        require(first['status']=='EXACT_CANDIDATE_REVIEWS_RECORDED_NOT_AUTHORIZED'
            and not first['dispatch_authorized'],'SELECTION_REVIEW')
        result_pin=save('selection-review-result.json',first)
        state=journal.read();state['events'][key]['selection_review_result']=result_pin;journal.save(state)
    selected=json.loads(checked(first['accepted']['selection']).read_text())
    require(selected==dict(design['selection_template'],independent_review=first['reviews']['selection']),'SELECTION_BINDING')
    second_input=save('compatibility-review-input.json',dict(kind='compatibility',selection=first['accepted']['selection'],origin_config=design['origin_config']))
    second=review.prepare(journal,parent_key,snapshot_pin,second_input,store,
        executable=executable,free_disk_bytes=free_disk_bytes,purpose=purpose)
    require(second['status']=='EXACT_CANDIDATE_REVIEWS_RECORDED_NOT_AUTHORIZED' and not second['dispatch_authorized'],'COMPATIBILITY_REVIEW')
    draft={k:reference[k] for k in ('master_authority','baseline_bindings','extension_inputs','group','evaluation_seed')}
    seed=20260908 if selected['role']=='baseline_control' else selected['training_plan']['training_seed']
    draft.update(design['evaluation_template'],evaluation_selection=selected,training_seed=seed,
        checkpoint_compatibility=second['accepted']['compatibility'])
    validator.validate(snapshot_pin,draft,design['result_payload_bytes']);profile.guard(journal)
    old=reference.get('checkpoint_compatibility',{})
    new_context=dict(context,snapshot_receipt=snapshot_pin,baseline_profile=save('profile-reference.json',draft),
        conditions=[c for c in context['conditions'] if c['sha256']!=old.get('sha256')])
    return dict(status='CANDIDATE_REVIEWS_ACCEPTED_CONTEXT_READY',context=save('context.json',new_context),
        selection_review=first,compatibility_review=second,dispatch_authorized=False)
