"""One read-only review of the exact frozen replay parameters and capacity."""
import json
from pathlib import Path
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision import CLI,CLI_SHA,MODEL_CAP,model_stop_paths
from tools.r6_decision_packet import persist_exact,pin_bytes
from tools.r6_batch import model_handoff,verified_decision
from tools.r6_budget import totals
from tools import r6_profile_preparation as profile
from tools import r6_render_preparation as plans

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'docs/experiments/r6-robustness-auto/prompts/render-plan-review.schema.json'
CAP=256*1024


def require(value,reason):
    if not value:raise RecoveryError('RENDER_PLAN_REVIEW_'+reason)


def prepare(journal,preparation_pin,store,*,executable,free_disk_bytes):
    journal.require_writer();state=journal.read();name='RENDER_PLAN_INPUT:'+preparation_pin['sha256']
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    if prior:
        key,entry=prior[0];packet=entry['request']['packet']
        require(packet['preparation']==preparation_pin and packet['executable']==executable,'INPUT_DRIFT')
    else:
        profile.guard(journal);plans.describe_prepared(journal,preparation_pin,store,free_disk_bytes=free_disk_bytes)
        exe=checked(executable)
        if state['goal_id']=='R6-CPU-TEST-FIXTURE':require(exe.is_relative_to(journal.directory.parent.resolve()),'FIXTURE_EXE')
        else:require(executable==dict(path=str(CLI),sha256=CLI_SHA),'CLI')
        packet=dict(preparation=preparation_pin,executable=executable,implementation=file_pin(__file__),validator=file_pin(plans.__file__),
            store_root=str(store.root.resolve()),store_sha256=sha(store.replay()),budget=totals(state['budget']),caps=state['budget']['caps'],
            protected=state['budget']['protected_confirmation_delivery_repair'])
        event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-RENDER',input_sha256=sha(dict(packet=packet)),role='render-plan-input')
        key,_=journal.intent(event,kind='cpu',request=dict(packet=packet),
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    for field in ('preparation','executable','implementation','validator'):checked(packet[field])
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    folder=journal.directory/'render-plan-review'/key
    prompt=('Independently review this exact R6 recorded-trajectory replay plan, read-only, then exit. '
        'Read preparation, parameters, frozen source inventory, extraction receipt/raw provenance, volume plans/manifests '
        'and resource ledger. Verify new replay identity, all full selected trajectories, preserved failures/resets/clocks, '
        'no interpolation or new physics/policy inference, exact scene/assets, immutable source dependencies and runtime. '
        'Check native/result hard quotas, fresh admission, zero transitions with nonzero GPU/wall/live/init charges, '
        'bounded owner cleanup, no model waiting and sufficient final delivery/repair reserves. '
        'Reuse of the existing final snapshot grants no inherited render approval. Store approval comes later. '
        'Inspect the deterministic spec/entry builders; do not edit files, mutate Journal/Store, query GPUs, launch '
        'simulation, spawn agents or issue permission. PASS_R6_RENDER_PLAN is only prospective plan acceptance; '
        'otherwise return MUST_FIX. reviewer=r6-independent-render-plan; changed_files=[]; evidence_pins must include '
        'packet.preparation and original evidence used. preparation_sha256 must match that pin; payload_sha256 '
        'must hash the exact parameters object excluding independent_review, canonical JSON. '
        'Remaining sessions=caps minus budget minus one.\n'+json.dumps(packet,sort_keys=True)+'\n').encode()
    require(len(prompt)<=65536,'PROMPT_CAP')
    request=dict(prompt=pin_bytes(folder/'prompt.txt',prompt),schema=file_pin(SCHEMA),executable=executable,
        cwd=str(ROOT),sandbox='read-only',timeout_seconds=3600,output_cap_bytes=MODEL_CAP,stop_paths=model_stop_paths(journal))
    event=dict(goal_id=state['goal_id'],event_id='RENDER_PLAN_REVIEW:'+key,job_id='FINAL-RENDER',input_sha256=sha(request),role='render-plan-review')
    prior=[(k,e) for k,e in journal.read()['events'].items() if e['identity'].get('event_id')==event['event_id']]
    require(len(prior)<=1,'MODEL_ALIAS')
    if prior:
        model_key,entry=prior[0];require(entry['identity']==event and entry['request']==request,'MODEL_DRIFT')
    else:
        profile.guard(journal)
        model_key,_=journal.intent(event,kind='model',request=request,
            resources={'automatic_model_sessions':1,'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},
            purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),model_key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},'confirmation_delivery')
    entry=journal.read()['events'][model_key]
    if not entry.get('dispatch_claim') and not entry.get('receipt'):
        expected=dict(packet['budget']);expected['automatic_model_sessions']+=1
        expected['output_actual_plus_unreleased_reservations_bytes']+=CAP+MODEL_CAP+65536
        require(totals(journal.read()['budget'])==expected,'STALE_BUDGET')
        profile.guard(journal);plans.describe_prepared(journal,preparation_pin,store,free_disk_bytes=free_disk_bytes)
        require(sha(store.replay())==packet['store_sha256'],'STORE_DRIFT')
        persist_exact(folder/'prompt.txt',prompt)
    model_handoff(journal,model_key);profile.guard(journal)
    plans.describe_prepared(journal,preparation_pin,store,free_disk_bytes=free_disk_bytes)
    require(sha(store.replay())==packet['store_sha256'],'STORE_DRIFT')
    result=verified_decision(journal,model_key)
    require(result['decision']=='PASS_R6_RENDER_PLAN' and result['must_fix']==[]
        and result['changed_files']==[] and result['reviewer']=='r6-independent-render-plan'
        and result['payload_sha256']==plans.payload_sha(preparation_pin)
        and result['preparation_sha256']==preparation_pin['sha256'],'REJECTED')
    require(preparation_pin in result['evidence_pins'],'EVIDENCE')
    for pin in result['evidence_pins']:checked(pin)
    require(result['remaining_budget']['automatic_sessions_remaining']==
        packet['caps']['automatic_model_sessions']-packet['budget']['automatic_model_sessions']-1,'BUDGET')
    receipt=json.loads(checked(journal.read()['events'][model_key]['receipt']).read_text())
    review=receipt['last_message']
    require(json.loads(checked(review).read_text())==result,'RECEIPT_OUTPUT')
    output=dict(status='RENDER_PLAN_REVIEWED_NOT_AUTHORIZED',preparation=preparation_pin,review=review,
        event_key=model_key,scientific_acceptance=False,dispatch_authorized=False)
    persist_exact(folder/'result.json',canonical(output)+b'\n')
    return output


def verify(journal,model_key,review_pin,preparation_pin,store):
    """Verify the original batch receipt after earlier assignments changed Store."""
    state=journal.read();entry=state['events'][model_key];identity=entry['identity']
    require(entry['kind']=='model' and identity['role']=='render-plan-review'
        and identity['goal_id']==state['goal_id'] and identity['job_id']=='FINAL-RENDER'
        and sha(identity)==model_key and identity['input_sha256']==sha(entry['request']),'MODEL_IDENTITY')
    prefix='RENDER_PLAN_REVIEW:'
    require(identity['event_id'].startswith(prefix),'MODEL_EVENT')
    cpu_key=identity['event_id'][len(prefix):];cpu=state['events'][cpu_key];packet=cpu['request']['packet']
    require(cpu['kind']=='cpu' and cpu['identity']['role']=='render-plan-input'
        and cpu['identity']['goal_id']==state['goal_id'] and sha(cpu['identity'])==cpu_key
        and cpu['identity']['input_sha256']==sha(cpu['request'])
        and packet['preparation']==preparation_pin and packet['store_root']==str(store.root.resolve()),'PACKET')
    require(packet['implementation']==file_pin(__file__) and packet['validator']==file_pin(plans.__file__),'IMPLEMENTATION')
    for field in ('preparation','executable','implementation','validator'):checked(packet[field])
    profile.reservation(state,cpu_key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    profile.reservation(state,model_key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},'confirmation_delivery')
    require(entry['request']['schema']==file_pin(SCHEMA) and entry['request']['sandbox']=='read-only'
        and entry['request']['executable']==packet['executable'],'MODEL_CONTRACT')
    receipt=json.loads(checked(entry['receipt']).read_text())
    require(receipt['last_message']==review_pin,'REVIEW_PIN')
    result=verified_decision(journal,model_key)
    require(result==json.loads(checked(review_pin).read_text()) and result['decision']=='PASS_R6_RENDER_PLAN'
        and result['must_fix']==[] and result['changed_files']==[]
        and result['reviewer']=='r6-independent-render-plan'
        and result['payload_sha256']==plans.payload_sha(preparation_pin)
        and result['preparation_sha256']==preparation_pin['sha256']
        and preparation_pin in result['evidence_pins'],'REJECTED')
    for pin in result['evidence_pins']:checked(pin)
    require(result['remaining_budget']['automatic_sessions_remaining']==
        packet['caps']['automatic_model_sessions']-packet['budget']['automatic_model_sessions']-1,'BUDGET')
    return result
