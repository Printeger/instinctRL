"""Budgeted independent model selection or selected-source compatibility review; no live authority."""
import json
from pathlib import Path
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_budget import totals
from tools.r6_batch import model_handoff,verified_decision
from tools.r6_decision import CLI,CLI_SHA,MODEL_CAP,model_stop_paths,record as parent_decision
from tools.r6_decision_packet import persist_exact,pin_bytes
from tools import r6_profile_preparation as profile
from tools import r6_candidate_review_inputs as inputs

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'docs/experiments/r6-robustness-auto/prompts/candidate-review.schema.json'
ROLE='candidate-independent-review'
CAP=256*1024
DECISIONS={'selection':'PASS_R6_EVALUATION_SELECTION','compatibility':'PASS_R6_CANDIDATE_COMPATIBILITY'}


def require(value,reason):
    if not value:raise RecoveryError('CANDIDATE_REVIEW_'+reason)


def prepare(journal,parent_key,snapshot_pin,input_pin,store,*,executable,free_disk_bytes,purpose='development'):
    journal.require_writer();state=journal.read()
    require(purpose in ('development','repair','confirmation_delivery'),'PURPOSE')
    name='CANDIDATE_REVIEW_INPUT:'+parent_key+':'+snapshot_pin['sha256']+':'+input_pin['sha256']
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    if not prior:
        profile.guard(journal);parent=parent_decision(journal,parent_key,store)
        require(parent['decision']['decision']!='REQUEST_FINAL_REVIEW','FINAL_ROUTE')
        request=json.loads(checked(input_pin).read_text())
        prepared=inputs.prepare(snapshot_pin,request)
        require(prepared['status']=='REVIEW_PAYLOAD_PREPARED_NOT_ACCEPTED' and not prepared['dispatch_authorized'],'PAYLOAD_STATUS')
        profile.guard(journal);state=journal.read()
        packet=dict(parent=state['events'][parent_key]['decision_record'],snapshot=snapshot_pin,
            inputs=input_pin,prepared=prepared,implementation=file_pin(__file__),validator=file_pin(inputs.__file__),
            store_sha256=sha(store.replay()),budget=totals(state['budget']),
            caps=state['budget']['caps'],protected=state['budget']['protected_confirmation_delivery_repair'])
        event=dict(goal_id=state['goal_id'],event_id=name,job_id=state['events'][parent_key]['identity']['job_id'],
            input_sha256=sha(dict(packet=packet)),role='candidate-review-input')
        key,_=journal.intent(event,kind='cpu',request=dict(packet=packet),
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose=purpose,free_disk_bytes=free_disk_bytes)
    else:
        key,entry=prior[0];packet=entry['request']['packet']
    for field in ('parent','snapshot','inputs','implementation','validator'):checked(packet[field])
    require(packet['snapshot']==snapshot_pin and packet['inputs']==input_pin,'PACKET_BINDING')
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},purpose)
    folder=journal.directory/'candidate-reviews'/key
    def save(name,value):
        data=canonical(value)+b'\n'
        size=sum(p.stat().st_size for p in folder.iterdir() if p.name!=name) if folder.exists() else 0
        require(size+len(data)<=CAP,'OUTPUT_CAP')
        profile.guard(journal)
        persist_exact(folder/name,data);return file_pin(folder/name)
    kind=packet['prepared']['request'].get('kind','compatibility')
    payloads={kind:packet['prepared']['payload']}
    require(kind in DECISIONS,'KIND')
    expected={kind}
    require(set(payloads)==expected,'PAYLOAD_KINDS')
    pins={k:pin_bytes(folder/(k+'-payload.json'),canonical(v)+b'\n') for k,v in payloads.items()}
    prompt=('Independently review the exact R6 candidate evaluation input, read only, then exit. '
        'Do not modify files or Journal/Store, query GPU, simulate, provision volumes, submit jobs, or spawn agents. '
        'For selection: inspect the original diagnosis, comparison design and selected checkpoint provenance. '
        'For a trained candidate, audit exact closed training job, inventory/postrun, native failure, complete '
        'checkpoint/Adam/RNG evidence and first-rollout probability gate. Adjudicate evidence limits explicitly; '
        'do not treat CPU structural checks as actual training success. For fixed baseline control require exact '
        'u640. Both roles must share protocol/realization/method; do not attribute ICS effects to learning. '
        'Reject final holdout selection, hidden tuning and unreviewed method/actor/metric changes. '
        'For compatibility: inspect exact selected model origin versus frozen execution source bytes and changes; '
        'verify actor, runtime and original scoring invariants and strict restore semantics. '
        'These prospective reviews confer no scientific acceptance or live authority. '
        'Return exactly one reviews item for the payload key with matching payload_sha256, decision from '
        'decisions or MUST_FIX, reviewer=r6-independent-candidate-review, summary and must_fix. '
        'Top decision PASS_R6_CANDIDATE_REVIEW requires the item pass, no must_fix and changed_files=[]; '
        'otherwise MUST_FIX. Include payload, parent, snapshot and inputs pins in evidence_pins. '
        'Remaining sessions=caps.automatic_model_sessions-budget.automatic_model_sessions-1.\n'
        +json.dumps(dict(packet,payload_pins=pins,decisions=DECISIONS,
            payload_hashes={k:sha(v) for k,v in payloads.items()}),sort_keys=True)+'\n').encode()
    require(len(prompt)<=65536,'PROMPT_CAP')
    request=dict(prompt=pin_bytes(folder/'prompt.txt',prompt),schema=file_pin(SCHEMA),executable=executable,
        cwd=str(ROOT),sandbox='read-only',timeout_seconds=3600,output_cap_bytes=MODEL_CAP,stop_paths=model_stop_paths(journal))
    model_event='CANDIDATE_REVIEW_MODEL:'+key
    event=dict(goal_id=state['goal_id'],event_id=model_event,job_id=state['events'][parent_key]['identity']['job_id'],
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
        profile.guard(journal);parent_decision(journal,parent_key,store)
        require(journal.read()['events'][parent_key]['decision_record']==packet['parent']
            and sha(store.replay())==packet['store_sha256'],'PARENT_OR_STORE_DRIFT')
        for kind,value in payloads.items():save(kind+'-payload.json',value)
        persist_exact(folder/'prompt.txt',prompt)
    model_handoff(journal,model_key)
    profile.guard(journal);parent_decision(journal,parent_key,store)
    require(journal.read()['events'][parent_key]['decision_record']==packet['parent']
        and sha(store.replay())==packet['store_sha256'],'PARENT_OR_STORE_DRIFT')
    result=verified_decision(journal,model_key)
    require(result['decision']=='PASS_R6_CANDIDATE_REVIEW' and not result['must_fix']
        and result['changed_files']==[],'REVIEW_REJECTED')
    require(result['remaining_budget']['automatic_sessions_remaining']==
        packet['caps']['automatic_model_sessions']-packet['budget']['automatic_model_sessions']-1,'BUDGET')
    require(all(pin in result['evidence_pins'] for pin in [*pins.values(),packet['parent'],snapshot_pin,input_pin]),'EVIDENCE')
    for pin in result['evidence_pins']:checked(pin)
    reviews=result['reviews']
    require(len(reviews)==len(payloads) and {r['kind'] for r in reviews}==set(payloads),'REVIEW_KINDS')
    for r in reviews:
        require(r['decision']==DECISIONS[r['kind']] and not r['must_fix']
            and r['reviewer']=='r6-independent-candidate-review'
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
    return dict(status='EXACT_CANDIDATE_REVIEWS_RECORDED_NOT_AUTHORIZED',event_key=model_key,
        payloads=pins,reviews=review_pins,accepted=accepted,dispatch_authorized=False)
