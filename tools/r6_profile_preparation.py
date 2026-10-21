"""One read-only profile decision, followed by controller-owned storage planning.

The model returns JSON data, not executable commands or authority. Existing
storage and materialization tools retain all accounting and validation gates.
"""
import json
import re
from pathlib import Path

from rlflow.store import sha,canonical
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision import record as parent_decision,CLI,CLI_SHA,MODEL_CAP,model_stop_paths
from tools.r6_decision_packet import persist_exact
from tools.r6_batch import model_handoff,verified_decision
from tools.r6_budget import totals
from tools.r6_job_handoff import no_model_workers
from tools.r6_postrun import stop_paths
from tools import r6_storage_allocation as storage
from tools import r6_materialize_job as materializer
from tools import r6_training_profile as training_profile
from tools import r6_candidate_profile as candidate_profile
from tools.r6_prepare_authorization import verify_frozen

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'docs/experiments/r6-robustness-auto/prompts/profile-preparation.schema.json'
ROLE='profile-preparation'
CAP=256*1024
CONTEXT={'snapshot_receipt','baseline_profile','stage_id','conditions','runner'}


def require(value,reason):
    if not value:raise RecoveryError('PROFILE_PREPARATION_'+reason)


def guard(journal):
    journal.require_writer();journal.guard();require(not stop_paths(journal),'STOP')
    no_model_workers(journal);require(journal.read().get('current_job') is None,'ACTIVE_JOB')


def reservation(state,key,kind,resources,purpose):
    entry=state['events'][key];identity=entry['identity']
    require(sha(identity)==key and identity['goal_id']==state['goal_id']
        and identity['input_sha256']==sha(entry['request']) and entry['kind']==kind
        and entry['resources']==resources and entry['purpose']==purpose,'EVENT_BINDING')
    rows=[r for r in state['budget']['reservations'] if r['identity']==identity]
    require(len(rows)==1 and rows[0]['resources']==resources and rows[0]['purpose']==purpose
        and rows[0]['event_key']==json.dumps(tuple(identity[k] for k in sorted(identity)),separators=(',',':')),
        'RESERVATION')


def packet(journal,parent_key,context_pin,store,*,free_disk_bytes):
    guard(journal);parent=parent_decision(journal,parent_key,store)
    require(parent['decision']['decision']!='REQUEST_FINAL_REVIEW','FINAL_ROUTE')
    context=json.loads(checked(context_pin).read_text());require(set(context)==CONTEXT,'CONTEXT')
    runner=context['runner']
    require(isinstance(runner,dict) and set(runner)=={'query_executable','decision_executable',
        'queue_review','sampler_review','purpose','sandbox'}
        and runner['purpose'] in {'development','repair','confirmation_delivery'}
        and runner['sandbox'] in {'read-only','danger-full-access'},'RUNNER')
    for name in ('query_executable','decision_executable','queue_review','sampler_review'):checked(runner[name])
    receipt=json.loads(checked(context['snapshot_receipt']).read_text())
    require(receipt.get('status')=='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY','SNAPSHOT')
    verify_frozen(receipt,json.loads(checked(receipt['config']).read_text()))
    checked(context['baseline_profile'])
    state=journal.read();parent_pin=state['events'][parent_key]['decision_record']
    name='PROFILE_INPUT:'+parent_key+':'+context_pin['sha256']
    previous=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(previous)<=1,'PACKET_ALIAS')
    if previous:
        key,entry=previous[0];value=entry['request']['packet']
        require(value['context']==context_pin and value['parent']==parent_pin,'PACKET_DRIFT')
    else:
        value=dict(parent=parent_pin,parent_key=parent_key,context=context_pin,
            budget=dict(caps=state['budget']['caps'],charged=totals(state['budget']),
                protected=state['budget']['protected_confirmation_delivery_repair']),
            store_sha256=sha(store.replay()),implementation=file_pin(__file__),
            training_validator=file_pin(training_profile.__file__),candidate_validator=file_pin(candidate_profile.__file__))
        request=dict(packet=value)
        event=dict(goal_id=state['goal_id'],event_id=name,
            job_id=state['events'][parent_key]['identity']['job_id'],input_sha256=sha(request),role='profile-input')
        key,_=journal.intent(event,kind='cpu',request=request,
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
            purpose=context['runner']['purpose'],free_disk_bytes=free_disk_bytes)
    checked(value['implementation']);checked(value['training_validator']);checked(value['candidate_validator'])
    reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},runner['purpose'])
    folder=journal.directory/'profile-preparation'/key
    text=('Prepare one exact R6 evaluation or training profile from the pinned scientific decision and source '
        'context, READ ONLY, then exit. Do not edit files, run GPUs/simulation, create volumes or '
        'snapshots, mutate Journal/Store, issue approval, spawn agents, or submit jobs. Read the '
        'parent original evidence, frozen source contract and baseline_profile. Preserve actor '
        'information boundaries, frozen metrics/splits and budgets. Return PROFILE_READY only '
        'when this frozen source supports the intended next step; otherwise return NEEDS_SOURCE_CHANGE '
        'with concrete must_fix. Never substitute baseline evaluation for required training/final '
        'confirmation. profile_json must encode a full profile draft except experiment_card, '
        'native_volume, result_volume and output_cap_bytes, which controller alone supplies. Use '
        'a new experiment ID. For training, include the exact r6_training_plan, separately reviewed '
        'recipe, capacity_review and training checkpoint_compatibility. Preserve the exact reviewed plan '
        'experiment_id; never rename a training plan to fit a different profile. The frozen training draft '
        'validator must accept before volume reservation. Never reuse baseline compatibility for '
        'training. For candidate evaluation include the independently reviewed evaluation_selection and '
        'selection-specific checkpoint_compatibility, retain the frozen protocol_review and all 16 '
        'blocks, and use the selected checkpoint training seed. Candidate draft validation in the '
        'frozen dedicated candidate config must pass before storage reservation. Do not combine '
        'evaluation_selection and r6_training_plan. Preserve exact immutable master authority/baseline/extension pins. '
        'Specify aligned result_payload_bytes and minimum_free_inodes from full raw/checkpoint '
        'requirements. No profile is permission. Include exact packet/context/parent pins in '
        'evidence_pins, changed_files=[], and remaining automatic sessions = packet budget cap '
        'minus charged minus one. For NEEDS_SOURCE_CHANGE use empty profile_json and zero storage '
        'dimensions. The controller checks all output, reserves storage, and later uses the frozen '
        'validator and independent exact prelive review. Packet follows:\n'
        +json.dumps(dict(value,packet_path=str((folder/'packet.json').resolve())),sort_keys=True)+'\n')
    data=canonical(value)+b'\n';require(len(text.encode())+len(data)<=65536,'CONTEXT_CAP')
    persist_exact(folder/'packet.json',data);persist_exact(folder/'prompt.txt',text.encode())
    return key,file_pin(folder/'packet.json'),file_pin(folder/'prompt.txt')


def register(journal,parent_key,context_pin,store,*,executable,free_disk_bytes):
    guard(journal);exe=checked(executable)
    if journal.read()['goal_id']=='R6-CPU-TEST-FIXTURE':
        require(exe.is_relative_to(journal.directory.parent.resolve()),'FIXTURE_EXECUTABLE')
    else:require(executable==dict(path=str(CLI),sha256=CLI_SHA),'CLI')
    packet_key,packet_pin,prompt_pin=packet(journal,parent_key,context_pin,store,free_disk_bytes=free_disk_bytes)
    state=journal.read();context=json.loads(checked(context_pin).read_text())
    request=dict(prompt=prompt_pin,schema=file_pin(SCHEMA),executable=executable,cwd=str(ROOT),
        sandbox='read-only',timeout_seconds=3600,output_cap_bytes=MODEL_CAP,stop_paths=model_stop_paths(journal))
    event=dict(goal_id=state['goal_id'],event_id='PROFILE_MODEL:'+packet_key,
        job_id=state['events'][parent_key]['identity']['job_id'],input_sha256=sha(request),role=ROLE)
    # A context cannot silently select a different executable/schema and spend twice.
    aliases=[e for e in state['events'].values() if e['identity'].get('event_id')==event['event_id']]
    require(all(e['identity']==event for e in aliases),'MODEL_ALIAS')
    key,_=journal.intent(event,kind='model',request=request,
        resources={'automatic_model_sessions':1,'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},
        purpose=context['runner']['purpose'],free_disk_bytes=free_disk_bytes)
    return key,packet_key,packet_pin


def record(journal,key,packet_key,packet_pin,store,*,free_disk_bytes):
    guard(journal);state=journal.read();entry=state['events'][key]
    value=json.loads(checked(packet_pin).read_text());context=json.loads(checked(value['context']).read_text())
    checked(value['implementation']);checked(value['training_validator']);checked(value['candidate_validator'])
    snapshot=json.loads(checked(context['snapshot_receipt']).read_text())
    verify_frozen(snapshot,json.loads(checked(snapshot['config']).read_text()))
    require(entry['identity']['event_id']=='PROFILE_MODEL:'+packet_key
        and entry['identity']['role']==ROLE and sha(entry['identity'])==key
        and entry['identity']['input_sha256']==sha(entry['request'])
        and entry['request']['schema']==file_pin(SCHEMA),'MODEL_BINDING')
    require(state['events'][packet_key]['request']=={'packet':value},'PACKET_BINDING')
    purpose=context['runner']['purpose']
    reservation(state,packet_key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},purpose)
    reservation(state,key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},purpose)
    parent_decision(journal,value['parent_key'],store)
    require(state['events'][value['parent_key']]['decision_record']==value['parent'],'PARENT_DRIFT')
    require(sha(store.replay())==value['store_sha256'],'STORE_MUTATED')
    result=verified_decision(journal,key)
    require(result['changed_files']==[] and result['packet_sha256']==packet_pin['sha256'],'RESULT_BINDING')
    require(all(p in result['evidence_pins'] for p in (packet_pin,value['parent'],value['context'])),'EVIDENCE')
    for pin in result['evidence_pins']:checked(pin)
    require(result['remaining_budget']['automatic_sessions_remaining']==
        value['budget']['caps']['automatic_model_sessions']-value['budget']['charged']['automatic_model_sessions']-1,'BUDGET')
    if result['decision']=='NEEDS_SOURCE_CHANGE':
        require(bool(result['must_fix']) and result['profile_json']==''
            and result['result_payload_bytes']==result['minimum_free_inodes']==0,'SOURCE_CHANGE_RESULT')
        return dict(status='SOURCE_PREPARATION_REQUIRED',model_event_key=key,findings=result['must_fix'],dispatch_authorized=False)
    require(result['decision']=='PROFILE_READY' and not result['must_fix'],'NOT_READY')
    # A scientific candidate must first go through the source preparation route,
    # which owns its one candidate charge and committed design card.
    parent=parent_decision(journal,value['parent_key'],store)
    if parent['decision']['decision']=='PROPOSE_CANDIDATE':
        from tools import r6_source_preparation as source
        matches=[e for e in state['events'].values() if e['identity'].get('role')==source.ROLE
            and e.get('source_input',{}).get('parent_key')==value['parent_key']]
        require(len(matches)==1 and 'source_result' in matches[0]
            and matches[0]['resources'].get('major_scientific_candidates')==1,'CANDIDATE_SOURCE_REQUIRED')
        prepared=json.loads(checked(matches[0]['source_result']).read_text())
        require(prepared['config']==snapshot['config'],'CANDIDATE_CONFIG')
    profile=json.loads(result['profile_json'])
    require(isinstance(profile,dict) and not materializer.MANAGED.intersection(profile),'MANAGED_FIELDS')
    reference=json.loads(checked(context['baseline_profile']).read_text())
    training='r6_training_plan' in profile
    candidate='evaluation_selection' in profile
    require(not (training and candidate),'AMBIGUOUS_STAGE')
    immutable=('master_authority','baseline_bindings','extension_inputs','group','evaluation_seed')
    if 'r6_training_plan' in reference:
        require(training and not candidate,'REGISTERED_TRAINING_STAGE')
        immutable+=('r6_training_plan','training_seed','experiment_id','capacity_review','checkpoint_compatibility')
    if candidate:
        from training_workloads.unified_h1.robustness_contract import FIELDS
        expected=(FIELDS-{'experiment_card','output_cap_bytes'})|{'extension_inputs','evaluation_selection'}
        require(set(profile)==expected,'PROFILE_FIELDS')
        immutable+=('protocol_review',)
        if 'evaluation_selection' in reference:
            immutable+=('evaluation_selection','checkpoint_compatibility')
    elif not training:
        require(set(profile)==set(reference)-materializer.MANAGED,'PROFILE_FIELDS')
        immutable+=('checkpoint_compatibility','protocol_review','training_seed')
    for name in immutable:
        require(profile.get(name)==reference.get(name),'IMMUTABLE_PROFILE_BINDING')
    if not training:
        require(isinstance(profile.get('r6'),dict) and isinstance(reference.get('r6'),dict)
            and profile['r6'].get('protocol')==reference['r6'].get('protocol'),'PROTOCOL_BINDING')
    experiment=profile.get('experiment_id');require(isinstance(experiment,str)
        and re.fullmatch('R6-[A-Z0-9-]{1,90}',experiment),'EXPERIMENT_ID')
    require(not any(j.get('spec',{}).get('id')==experiment for j in store.replay()['jobs'].values()),'USED_EXPERIMENT')
    payload=result['result_payload_bytes'];inodes=result['minimum_free_inodes']
    require(type(payload)is int and payload>0 and payload%4096==0 and type(inodes)is int and inodes>0,'STORAGE_DIMENSIONS')
    if candidate:
        required=candidate_profile.validate(context['snapshot_receipt'],profile,payload)
        require(payload==required,'CANDIDATE_PAYLOAD')
    elif training:
        required=training_profile.validate(context['snapshot_receipt'],profile)
        require(payload==required,'TRAINING_PAYLOAD')
    else:
        quotas=profile['r6'].get('output_quotas',{})
        require(isinstance(quotas,dict) and quotas and set(quotas)==set(profile['r6'].get('block_ids',[])), 'OUTPUT_QUOTAS')
        required=64*1024**2
        for quota in quotas.values():
            require(isinstance(quota,dict) and set(quota)=={'range_bytes','audit_bytes'}
                and all(type(v)is int and 0<v<=2*1024**3 for v in quota.values()),'OUTPUT_QUOTAS')
            required+=sum(quota.values())+16*1024**2
        require(payload>=required,'PAYLOAD_UNDERSIZED')
    guard(journal)
    volume_id='J-'+sha(dict(goal=state['goal_id'],parent=value['parent'],profile=profile,payload=payload,inodes=inodes))[:16]
    native=storage.native_tool.plan(volume_id)
    target=storage.result_tool.ATTEMPTS/experiment/'results'
    results=storage.result_tool.plan(volume_id,target,payload+96*1024**2,payload,inodes)
    folder=journal.directory/'profile-preparation'/packet_key
    outputs={'profile-draft.json':profile,'native-plan.json':native,'result-plan.json':results}
    require(sum(len(canonical(v)) for v in outputs.values())+sum(p.stat().st_size for p in folder.iterdir() if p.name not in outputs)<CAP-65536,'OUTPUT_CAP')
    for name,obj in outputs.items():persist_exact(folder/name,canonical(obj)+b'\n')
    allocation=storage.register(journal,experiment,file_pin(folder/'native-plan.json'),
        file_pin(folder/'result-plan.json'),purpose=context['runner']['purpose'],free_disk_bytes=free_disk_bytes)
    inputs=dict(snapshot_receipt=context['snapshot_receipt'],profile_draft=file_pin(folder/'profile-draft.json'),
        storage_key=allocation,stage_id=context['stage_id'],conditions=context['conditions'],runner=context['runner'])
    persist_exact(folder/'materialization-inputs.json',canonical(inputs)+b'\n')
    return dict(status='PROFILE_AND_STORAGE_PLANS_READY_NOT_PROVISIONED',model_event_key=key,
        storage_key=allocation,inputs=file_pin(folder/'materialization-inputs.json'),dispatch_authorized=False)


def prepare_profile(journal,parent_key,context_pin,store,*,executable,free_disk_bytes):
    journal.require_writer();state=journal.read()
    packets=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==
             'PROFILE_INPUT:'+parent_key+':'+context_pin['sha256']]
    require(len(packets)<=1,'PACKET_ALIAS')
    models=[]
    if packets:models=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')=='PROFILE_MODEL:'+packets[0][0]]
    require(len(models)<=1,'MODEL_ALIAS')
    if models and (models[0][1].get('dispatch_claim') or models[0][1].get('receipt')):
        key,entry=models[0];packet_key=packets[0][0]
        require(entry['request']['executable']==executable,'EXECUTABLE_DRIFT')
        packet_pin=file_pin(journal.directory/'profile-preparation'/packet_key/'packet.json')
    else:key,packet_key,packet_pin=register(journal,parent_key,context_pin,store,executable=executable,free_disk_bytes=free_disk_bytes)
    entry=journal.read()['events'][key]
    if not entry.get('dispatch_claim') and not entry.get('receipt'):
        value=json.loads(checked(packet_pin).read_text());expected=dict(value['budget']['charged'])
        expected['automatic_model_sessions']+=1
        expected['output_actual_plus_unreleased_reservations_bytes']+=CAP+MODEL_CAP+65536
        require(totals(journal.read()['budget'])==expected,'STALE_BUDGET')
    model_handoff(journal,key)
    return record(journal,key,packet_key,packet_pin,store,free_disk_bytes=free_disk_bytes)


def after_job(journal,job_key,context_pin,store,*,executable,free_disk_bytes):
    journal.require_writer();state=journal.read();job=journal.canonical_job(job_key,state=state)
    from tools.r6_decision_repair import active
    parent=active(journal,job_key)
    require(parent is not None,'EXACT_JOB_DECISION')
    require(state['events'][parent].get('decision_input',{}).get('job_event_key')==job_key,'PARENT_JOB')
    return prepare_profile(journal,parent,context_pin,store,executable=executable,
                           free_disk_bytes=free_disk_bytes)


def main():
    import argparse
    import shutil
    from rlflow.store import Store
    from tools.r6_recovery import Journal
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('job_event_key')
    parser.add_argument('context',type=Path);parser.add_argument('--context-sha256',required=True)
    parser.add_argument('--store',required=True,type=Path)
    parser.add_argument('--executable',required=True,type=Path)
    args=parser.parse_args();root=args.store.resolve(strict=True)
    require((root/'runtime/research.db').is_file(),'STORE_REQUIRED')
    journal=Journal(args.directory.resolve())
    with journal.writer():
        result=after_job(journal,args.job_event_key,
            dict(path=str(args.context.resolve()),sha256=args.context_sha256),Store(root),
            executable=file_pin(args.executable),free_disk_bytes=shutil.disk_usage(journal.directory).free)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
