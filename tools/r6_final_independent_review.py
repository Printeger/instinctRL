"""One independent final science, requirements and visual audit; no publication."""
import json
from pathlib import Path
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision import CLI,CLI_SHA,MODEL_CAP,model_stop_paths
from tools.r6_decision_packet import persist_exact,pin_bytes
from tools.r6_batch import model_handoff,verified_decision
from tools.r6_budget import totals
from tools import r6_profile_preparation as profile
from tools import r6_final_audit_packet as plans

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'docs/experiments/r6-robustness-auto/prompts/final-independent.schema.json'
CAP=256*1024


def require(value,reason):
    if not value:raise RecoveryError('FINAL_AUDIT_REVIEW_'+reason)


def prepare(journal,preparation_pin,store,*,executable,free_disk_bytes):
    journal.require_writer();state=journal.read();name='FINAL_AUDIT_INPUT:'+preparation_pin['sha256']
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    if prior:
        key,entry=prior[0];packet=entry['request']['packet']
        require(packet['preparation']==preparation_pin and packet['executable']==executable,'INPUT_DRIFT')
    else:
        profile.guard(journal);plans.verify(journal,preparation_pin,store)
        exe=checked(executable)
        if state['goal_id']=='R6-CPU-TEST-FIXTURE':require(exe.is_relative_to(journal.directory.parent.resolve()),'FIXTURE_EXE')
        else:require(executable==dict(path=str(CLI),sha256=CLI_SHA),'CLI')
        packet=dict(preparation=preparation_pin,executable=executable,implementation=file_pin(__file__),validator=file_pin(plans.__file__),
            store_root=str(store.root.resolve()),store_sha256=sha(store.replay()),budget=totals(state['budget']),caps=state['budget']['caps'],
            protected=state['budget']['protected_confirmation_delivery_repair'])
        event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-AUDIT',input_sha256=sha(dict(packet=packet)),role='final-audit-input')
        key,_=journal.intent(event,kind='cpu',request=dict(packet=packet),
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='final_review',free_disk_bytes=free_disk_bytes)
    for field in ('preparation','executable','implementation','validator'):checked(packet[field])
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'final_review')
    folder=journal.directory/'final-audit-review'/key
    prompt=('Independently audit the complete R6 goal evidence, read-only, then exit. Read packet.preparation, '
        'its description.requirements original contract and every checklist criterion, all artifacts and necessary original '
        'raw/code/receipt evidence. Do not accept summaries or green tests as proof of live scientific results. '
        'Inspect EVERY visual_samples.image using actual image viewing tools, correlate labeled frame/time/scene/pose '
        'with original video/telemetry; include exact image pins and frame indices in visual_audit. If image access is '
        'unavailable or insufficient, return MUST_FIX, never claim visual inspection from filenames/hashes. '
        'Audit all models/lanes/conditions including stress and native FAILED, all clocks/RNG/first-rollout gates, '
        'holdout/attribution, actual model-free wait evidence and resource/model ledgers. Preserve all failures. '
        'For each native failed job, supply disposition/reason/evidence; required scientific data from failed jobs '
        'needs explicit independently justified scientific_evidence_accepted. Every checklist key is mandatory; '
        'NOT_APPLICABLE only for training-specific items on BASELINE_SUFFICIENT, or baseline_sufficient on trained branch. '
        'R6_TARGET_DOMAIN_PASS and BASELINE_SUFFICIENT require all corresponding frozen gates; '
        'SCIENTIFIC_NEGATIVE_WITH_COMPLETE_EVIDENCE requires necessary registered experiments completed, not budget exhaustion. '
        'MUST_FIX must name missing evidence/engineering defects and leave scientific_status INCOMPLETE. '
        'Any uncertain or missing requirement is MUST_FIX. Scope is pre-publication audit; final accounting must '
        'include this review receipt and deterministic publication before completion. Read-only: do not edit files, '
        'mutate Journal/Store, query GPU, train, simulate, issue authority or spawn agents. '
        'reviewer=r6-independent-final-audit; changed_files=[]; preparation_sha256 matches packet.preparation. '
        'Evidence pins include packet.preparation and exact originals used. Remaining sessions=caps minus budget minus one.\n'
        +json.dumps(packet,sort_keys=True)+'\n').encode()
    require(len(prompt)<=65536,'PROMPT_CAP')
    request=dict(prompt=pin_bytes(folder/'prompt.txt',prompt),schema=file_pin(SCHEMA),executable=executable,
        cwd=str(ROOT),sandbox='read-only',timeout_seconds=3600,output_cap_bytes=MODEL_CAP,stop_paths=model_stop_paths(journal))
    event=dict(goal_id=state['goal_id'],event_id='FINAL_AUDIT_REVIEW:'+key,job_id='FINAL-AUDIT',input_sha256=sha(request),role='final-audit-review')
    prior=[(k,e) for k,e in journal.read()['events'].items() if e['identity'].get('event_id')==event['event_id']]
    require(len(prior)<=1,'MODEL_ALIAS')
    if prior:
        model_key,entry=prior[0];require(entry['identity']==event and entry['request']==request,'MODEL_DRIFT')
    else:
        profile.guard(journal)
        model_key,_=journal.intent(event,kind='model',request=request,
            resources={'automatic_model_sessions':1,'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},
            purpose='final_review',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),model_key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},'final_review')
    entry=journal.read()['events'][model_key]
    if not entry.get('dispatch_claim') and not entry.get('receipt'):
        expected=dict(packet['budget']);expected['automatic_model_sessions']+=1
        expected['output_actual_plus_unreleased_reservations_bytes']+=CAP+MODEL_CAP+65536
        require(totals(journal.read()['budget'])==expected,'STALE_BUDGET')
        profile.guard(journal);plans.verify(journal,preparation_pin,store)
        require(sha(store.replay())==packet['store_sha256'],'STORE_DRIFT')
        persist_exact(folder/'prompt.txt',prompt)
    model_handoff(journal,model_key);profile.guard(journal)
    plans.verify(journal,preparation_pin,store)
    require(sha(store.replay())==packet['store_sha256'],'STORE_DRIFT')
    result=verified_decision(journal,model_key)
    assess(result,preparation_pin)
    require(result['remaining_budget']['automatic_sessions_remaining']==
        packet['caps']['automatic_model_sessions']-packet['budget']['automatic_model_sessions']-1,'BUDGET')
    receipt=json.loads(checked(journal.read()['events'][model_key]['receipt']).read_text())
    review=receipt['last_message']
    require(json.loads(checked(review).read_text())==result,'RECEIPT_OUTPUT')
    output=dict(status='FINAL_AUDIT_MUST_FIX_REQUIRED' if result['decision']=='MUST_FIX' else 'FINAL_AUDIT_ACCEPTED_PENDING_PUBLICATION',preparation=preparation_pin,review=review,
        event_key=model_key,scientific_acceptance=False,dispatch_authorized=False)
    persist_exact(folder/'result.json',canonical(output)+b'\n')
    return output


def verify(journal,model_key,review_pin,preparation_pin,store):
    """Verify the original independent receipt without another model call."""
    plans.verify(journal,preparation_pin,store)
    state=journal.read();entry=state['events'][model_key];identity=entry['identity']
    require(entry['kind']=='model' and identity['role']=='final-audit-review'
        and identity['goal_id']==state['goal_id'] and identity['job_id']=='FINAL-AUDIT'
        and sha(identity)==model_key and identity['input_sha256']==sha(entry['request']),'MODEL_IDENTITY')
    prefix='FINAL_AUDIT_REVIEW:'
    require(identity['event_id'].startswith(prefix),'MODEL_EVENT')
    cpu_key=identity['event_id'][len(prefix):];cpu=state['events'][cpu_key];packet=cpu['request']['packet']
    require(cpu['kind']=='cpu' and cpu['identity']['role']=='final-audit-input'
        and cpu['identity']['goal_id']==state['goal_id'] and sha(cpu['identity'])==cpu_key
        and cpu['identity']['input_sha256']==sha(cpu['request'])
        and packet['preparation']==preparation_pin and packet['store_root']==str(store.root.resolve()),'PACKET')
    require(packet['implementation']==file_pin(__file__) and packet['validator']==file_pin(plans.__file__),'IMPLEMENTATION')
    for field in ('preparation','executable','implementation','validator'):checked(packet[field])
    profile.reservation(state,cpu_key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'final_review')
    profile.reservation(state,model_key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},'final_review')
    require(entry['request']['schema']==file_pin(SCHEMA) and entry['request']['sandbox']=='read-only'
        and entry['request']['executable']==packet['executable'],'MODEL_CONTRACT')
    receipt=json.loads(checked(entry['receipt']).read_text())
    require(receipt['last_message']==review_pin,'REVIEW_PIN')
    result=verified_decision(journal,model_key)
    require(result==json.loads(checked(review_pin).read_text()),'RECEIPT_OUTPUT')
    assess(result,preparation_pin)
    require(result['remaining_budget']['automatic_sessions_remaining']==
        packet['caps']['automatic_model_sessions']-packet['budget']['automatic_model_sessions']-1,'BUDGET')
    return result


def assess(result,preparation_pin):
    data=plans.read(preparation_pin);description=data['description'];final=description['final'];branch=description['branch']
    require(result['preparation_sha256']==preparation_pin['sha256'] and result['reviewer']=='r6-independent-final-audit'
        and result['changed_files']==[] and preparation_pin in result['evidence_pins'],'IDENTITY')
    for pin in result['evidence_pins']:checked(pin)
    audit=result['requirements_audit'];require(set(audit)==set(plans.CHECKS),'CHECKLIST')
    permitted=plans.BASELINE_NA if branch=='BASELINE_SUFFICIENT' else {'baseline_sufficient'}
    for name,item in audit.items():
        require(item['status'] in ('PASS','NOT_APPLICABLE','MUST_FIX') and bool(item['reason'].strip()),'CHECK_STATUS')
        require(item['status']!='NOT_APPLICABLE' or name in permitted,'INVALID_NOT_APPLICABLE')
        require(item['status']=='MUST_FIX' or bool(item['evidence_pins']),'CHECK_EVIDENCE')
        for pin in item['evidence_pins']:checked(pin)
    models=plans.read(data['artifacts']['MODEL_USAGE_AUDIT_PRE_REVIEW.json'])
    if not models['all_registered_local_waits_verified'] or not models['platform_inactivity_proven']:
        require(audit['zero_model_wait']['status']=='MUST_FIX','UNPROVEN_ZERO_MODEL_WAIT')
    manifest=plans.read(data['artifacts']['TRAINING_AND_EVALUATION_MANIFEST.json'])
    failed={j['event_key']:j for j in manifest['jobs'] if j['native_status']=='FAILED'}
    assessments=result['native_failure_assessments'];require(len(assessments)==len(failed) and {a['event_key'] for a in assessments}==set(failed),'FAILURE_COVERAGE')
    for a in assessments:
        require(a['disposition'] in ('preserved_unusable','scientific_evidence_accepted') and bool(a['reason'].strip()) and bool(a['evidence_pins']),'FAILURE_REASON')
        for pin in a['evidence_pins']:checked(pin)
    media=plans.read(description['video_manifest']);samples=media['visual_samples'];visual=result['visual_audit']
    seen=set()
    for row in visual:
        index=row['index'];require(type(index)is int and 0<=index<len(samples) and index not in seen,'VISUAL_INDEX');seen.add(index)
        expected=samples[index]
        require(row['image']==expected['image'] and row['frames']==expected['frames'] and bool(row['findings'].strip()),'VISUAL_BINDING');checked(row['image'])
    if result['decision']=='MUST_FIX':
        require(bool(result['must_fix']) and result['scientific_status']=='INCOMPLETE','MUST_FIX_STATUS');return result
    require(result['decision']=='PASS_R6_FINAL_AUDIT' and result['must_fix']==[]
        and all(v['status']!='MUST_FIX' for v in audit.values()) and len(visual)==len(samples),'INCOMPLETE_ACCEPTANCE')
    required={r['job_event_key'] for r in final['results']}
    require(all(a['disposition']=='scientific_evidence_accepted' for a in assessments if a['event_key'] in required),'FAILED_SCIENCE_NOT_ACCEPTED')
    status=result['scientific_status']
    if final['required_models_pass']:
        require(status==('BASELINE_SUFFICIENT' if branch=='BASELINE_SUFFICIENT' else 'R6_TARGET_DOMAIN_PASS'),'POSITIVE_BRANCH')
    else:require(status=='SCIENTIFIC_NEGATIVE_WITH_COMPLETE_EVIDENCE','NEGATIVE_NOT_COMPLETE')
    if branch=='BASELINE_SUFFICIENT':
        require(len(final['results'])==3 and len({r['assignment']['evaluation_seed'] for r in final['results']})==3,'BASELINE_STREAMS')
        resources=plans.read(data['artifacts']['RESOURCE_AUDIT_PRE_REVIEW.json'])
        require(resources['charged']['training_transitions']==0,'BASELINE_TRAINING')
    else:require(branch=='THREE_CONTINUATIONS' and len(final['results'])==4,'THREE_CONFIRMATIONS')
    return result
