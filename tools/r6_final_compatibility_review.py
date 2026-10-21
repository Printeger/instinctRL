"""One budgeted independent compatibility review for every frozen final assignment."""
import json
from pathlib import Path
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_budget import totals
from tools.r6_batch import model_handoff,verified_decision
from tools.r6_decision import CLI,CLI_SHA,MODEL_CAP,model_stop_paths
from tools.r6_decision_packet import persist_exact,pin_bytes
from tools import r6_profile_preparation as profile
from tools.r6_holdout_publication import validate_batch
from tools import r6_final_review_inputs as inputs

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'docs/experiments/r6-robustness-auto/prompts/final-compatibility-review.schema.json'
ROLE='final-compatibility-independent-review'
CAP=1024*1024
DECISIONS={str(i):'PASS_R6_FINAL_COMPATIBILITY' for i in range(4)}


def require(value,reason):
    if not value:raise RecoveryError('FINAL_COMPATIBILITY_REVIEW_'+reason)


def prepare(journal,snapshot_pin,input_pin,store,*,executable,free_disk_bytes):
    purpose='confirmation_delivery'
    journal.require_writer();state=journal.read()
    batch_pin=json.loads(checked(input_pin).read_text())['batch']
    name='FINAL_COMPATIBILITY_REVIEW_INPUT:'+batch_pin['sha256']+':'+snapshot_pin['sha256']+':'+input_pin['sha256']
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    if not prior:
        profile.guard(journal);validate_batch(journal,batch_pin)
        request=json.loads(checked(input_pin).read_text())
        prepared=inputs.prepare(snapshot_pin,request)
        require(prepared['status']=='REVIEW_PAYLOAD_PREPARED_NOT_ACCEPTED' and not prepared['dispatch_authorized'],'PAYLOAD_STATUS')
        profile.guard(journal);state=journal.read()
        packet=dict(batch=batch_pin,snapshot=snapshot_pin,
            inputs=input_pin,prepared=prepared,implementation=file_pin(__file__),validator=file_pin(inputs.__file__),
            store_sha256=sha(store.replay()),budget=totals(state['budget']),
            caps=state['budget']['caps'],protected=state['budget']['protected_confirmation_delivery_repair'])
        event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-BATCH',
            input_sha256=sha(dict(packet=packet)),role='final-compatibility-review-input')
        key,_=journal.intent(event,kind='cpu',request=dict(packet=packet),
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose=purpose,free_disk_bytes=free_disk_bytes)
    else:
        key,entry=prior[0];packet=entry['request']['packet']
    for field in ('batch','snapshot','inputs','implementation','validator'):checked(packet[field])
    require(packet['snapshot']==snapshot_pin and packet['inputs']==input_pin,'PACKET_BINDING')
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},purpose)
    folder=journal.directory/'final-compatibility-reviews'/key
    def save(name,value):
        data=canonical(value)+b'\n'
        size=sum(p.stat().st_size for p in folder.iterdir() if p.name!=name) if folder.exists() else 0
        require(size+len(data)<=CAP,'OUTPUT_CAP')
        profile.guard(journal)
        persist_exact(folder/name,data);return file_pin(folder/name)
    payloads=packet['prepared']['payloads']
    expected={str(i) for i in range(len(json.loads(checked(batch_pin).read_text())['assignments']))}
    require(set(payloads)==expected and len(payloads) in (3,4),'PAYLOAD_KINDS')
    pins={k:pin_bytes(folder/(k+'-payload.json'),canonical(v)+b'\n') for k,v in payloads.items()}
    prompt_packet={k:v for k,v in packet.items() if k!='prepared'}
    prompt_packet['prepared']={k:packet['prepared'][k] for k in ('status','request','source_identity')}
    prompt=('Independently review the complete R6 final evaluation source compatibility batch, read only, then exit. '
        'Do not modify files or Journal/Store, query GPU, simulate, provision volumes, submit jobs or spawn agents. '
        'Verify the released batch and exact fixed assignments, original-u640 baseline versus frozen confirmed models, '
        'checkpoint provenance and original source versus frozen final execution source. No model/method/seed '
        'selection or tuning using holdout is permitted. Audit exact source deltas, strict serializer/Adam/RNG '
        'semantics, actor information boundaries, unchanged runtime and original scoring, full final block '
        'denominators and baseline ICS .15. Confirmed method bindings must match the already frozen recipe. '
        'Audit the whole batch in one response; prospective compatibility grants no scientific acceptance or '
        'dispatch authority. Return one reviews item per numeric assignment key, matching payload_sha256, '
        'decision PASS_R6_FINAL_COMPATIBILITY or MUST_FIX, reviewer=r6-independent-final-compatibility-review. '
        'Top decision PASS_R6_FINAL_COMPATIBILITY_REVIEW requires every item pass, no must_fix and changed_files=[]; '
        'otherwise MUST_FIX. Include all payload pins, batch, snapshot and inputs in evidence_pins. '
        'Remaining sessions=caps.automatic_model_sessions-budget.automatic_model_sessions-1.\n'
        +json.dumps(dict(prompt_packet,payload_pins=pins,decisions=DECISIONS,
            payload_hashes={k:sha(v) for k,v in payloads.items()}),sort_keys=True)+'\n').encode()
    require(len(prompt)<=65536,'PROMPT_CAP')
    request=dict(prompt=pin_bytes(folder/'prompt.txt',prompt),schema=file_pin(SCHEMA),executable=executable,
        cwd=str(ROOT),sandbox='read-only',timeout_seconds=3600,output_cap_bytes=MODEL_CAP,stop_paths=model_stop_paths(journal))
    model_event='FINAL_COMPATIBILITY_REVIEW_MODEL:'+key
    event=dict(goal_id=state['goal_id'],event_id=model_event,job_id='FINAL-BATCH',
        input_sha256=sha(request),role=ROLE)
    models=[(k,e) for k,e in journal.read()['events'].items() if e['identity'].get('event_id')==model_event]
    require(len(models)<=1,'MODEL_ALIAS')
    if models:
        model_key,entry=models[0];require(entry['identity']==event and entry['request']==request,'MODEL_DRIFT')
    else:
        profile.guard(journal);exe=checked(executable)
        if state['goal_id']=='R6-CPU-TEST-FIXTURE':require(exe.is_relative_to(journal.directory.parent.resolve()),'FIXTURE_EXE')
        else:require(executable==dict(path=str(CLI),sha256=CLI_SHA),'CLI')
        model_key,_=journal.intent(event,kind='model',request=request,
            resources={'automatic_model_sessions':1,'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},
            purpose=purpose,free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),model_key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},purpose)
    entry=journal.read()['events'][model_key]
    if not entry.get('dispatch_claim') and not entry.get('receipt'):
        expected_budget=dict(packet['budget']);expected_budget['automatic_model_sessions']+=1
        expected_budget['output_actual_plus_unreleased_reservations_bytes']+=CAP+MODEL_CAP+65536
        require(totals(journal.read()['budget'])==expected_budget,'STALE_BUDGET')
        profile.guard(journal);validate_batch(journal,batch_pin)
        require(packet['batch']==batch_pin and sha(store.replay())==packet['store_sha256'],'BATCH_OR_STORE_DRIFT')
        for kind,value in payloads.items():save(kind+'-payload.json',value)
        persist_exact(folder/'prompt.txt',prompt)
    model_handoff(journal,model_key)
    profile.guard(journal);validate_batch(journal,batch_pin)
    require(packet['batch']==batch_pin and sha(store.replay())==packet['store_sha256'],'BATCH_OR_STORE_DRIFT')
    result=verified_decision(journal,model_key)
    require(result['decision']=='PASS_R6_FINAL_COMPATIBILITY_REVIEW' and not result['must_fix']
        and result['changed_files']==[],'REVIEW_REJECTED')
    require(result['remaining_budget']['automatic_sessions_remaining']==
        packet['caps']['automatic_model_sessions']-packet['budget']['automatic_model_sessions']-1,'BUDGET')
    require(all(pin in result['evidence_pins'] for pin in [*pins.values(),packet['batch'],snapshot_pin,input_pin]),'EVIDENCE')
    for pin in result['evidence_pins']:checked(pin)
    reviews=result['reviews']
    require(len(reviews)==len(payloads) and {r['kind'] for r in reviews}==set(payloads),'REVIEW_KINDS')
    for r in reviews:
        require(r['decision']==DECISIONS[r['kind']] and not r['must_fix']
            and r['reviewer']=='r6-independent-final-compatibility-review'
            and r['payload_sha256']==sha(payloads[r['kind']]),'ITEM_REJECTED')
    require(inputs.prepare(snapshot_pin,json.loads(checked(input_pin).read_text()))==packet['prepared'],'PAYLOAD_DRIFT')
    profile.guard(journal)
    receipt_pin=journal.read()['events'][model_key]['receipt'];receipt=json.loads(checked(receipt_pin).read_text())
    require(receipt['event_key']==model_key and receipt['request_sha256']==sha(request),'RECEIPT_BINDING')
    accepted={};review_pins={}
    for r in reviews:
        kind=r['kind'];review_pins[kind]=save(kind+'-review.json',dict(r,
            parent_review=receipt['last_message'],worker_receipt=receipt_pin))
        accepted[kind]=save(kind+'-accepted.json',dict(payloads[kind],independent_review=review_pins[kind]))
    return dict(status='EXACT_FINAL_COMPATIBILITY_REVIEWS_RECORDED_NOT_AUTHORIZED',event_key=model_key,
        payloads=pins,reviews=review_pins,accepted=accepted,dispatch_authorized=False)
