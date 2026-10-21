"""One read-only review of every exact final plan before batch issuance."""
import json
from pathlib import Path
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision import CLI,CLI_SHA,MODEL_CAP,model_stop_paths
from tools.r6_decision_packet import persist_exact,pin_bytes
from tools.r6_batch import model_handoff,verified_decision
from tools.r6_budget import totals
from tools import r6_profile_preparation as profile
from tools import r6_final_plans as plans

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'docs/experiments/r6-robustness-auto/prompts/final-plans-review.schema.json'
CAP=256*1024


def require(value,reason):
    if not value:raise RecoveryError('FINAL_PLANS_REVIEW_'+reason)


def prepare(journal,materializations_pin,store,*,executable,free_disk_bytes):
    journal.require_writer();state=journal.read();name='FINAL_PLANS_INPUT:'+materializations_pin['sha256']
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    if prior:
        key,entry=prior[0];packet=entry['request']['packet']
        require(packet['materializations']==materializations_pin and packet['executable']==executable,'INPUT_DRIFT')
    else:
        profile.guard(journal);plans.describe(journal,materializations_pin,store,free_disk_bytes=free_disk_bytes)
        exe=checked(executable)
        if state['goal_id']=='R6-CPU-TEST-FIXTURE':require(exe.is_relative_to(journal.directory.parent.resolve()),'FIXTURE_EXE')
        else:require(executable==dict(path=str(CLI),sha256=CLI_SHA),'CLI')
        packet=dict(materializations=materializations_pin,executable=executable,implementation=file_pin(__file__),validator=file_pin(plans.__file__),
            store_root=str(store.root.resolve()),store_sha256=sha(store.replay()),budget=totals(state['budget']),caps=state['budget']['caps'],
            protected=state['budget']['protected_confirmation_delivery_repair'])
        event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-BATCH',input_sha256=sha(dict(packet=packet)),role='final-plans-input')
        key,_=journal.intent(event,kind='cpu',request=dict(packet=packet),
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    for field in ('materializations','executable','implementation','validator'):checked(packet[field])
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    folder=journal.directory/'final-plans-review'/key
    prompt=('Independently review this complete exact final R6 job batch, read-only, then exit. '
        'Read the pinned materializations and original preparation, storage, source compatibility, '
        'model freeze, holdout publication, authorization, spec, runner and budget evidence. '
        'Verify all three distinct-stream baseline validations or baseline plus all three confirmed '
        'models, exact assignment order, full original and frozen target/stress suites, model/seed/source '
        'identity, unchanged actor/runtime/metrics, zero training and optimizer/checkpoint writes, '
        'distinct attempts and prepaid dual volumes. Check aggregate remaining resource capacity '
        'including issuance, closing, final audit, videos, delivery and repair; the job-only check '
        'is not a delivery capacity guarantee. No per-model scientific choice or tuning is permitted. '
        'Store approvals are issued after this review; their absence is expected. Do not edit files, '
        'Journal or Store, query GPUs, run simulation/training, spawn agents or submit jobs. '
        'PASS_R6_FINAL_PLANS approves only these prospective exact plans, not scientific success '
        'or dispatch. Missing evidence requires MUST_FIX. changed_files=[]; reviewer=r6-independent-final-plans. '
        'materializations_sha256 must match packet.materializations.sha256 and evidence_pins must include '
        'that pin and originals relied on. Remaining sessions=caps minus budget minus one.\n'
        +json.dumps(packet,sort_keys=True)+'\n').encode()
    require(len(prompt)<=65536,'PROMPT_CAP')
    request=dict(prompt=pin_bytes(folder/'prompt.txt',prompt),schema=file_pin(SCHEMA),executable=executable,
        cwd=str(ROOT),sandbox='read-only',timeout_seconds=3600,output_cap_bytes=MODEL_CAP,stop_paths=model_stop_paths(journal))
    event=dict(goal_id=state['goal_id'],event_id='FINAL_PLANS_REVIEW:'+key,job_id='FINAL-BATCH',input_sha256=sha(request),role='final-plans-review')
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
        profile.guard(journal);plans.describe(journal,materializations_pin,store,free_disk_bytes=free_disk_bytes)
        require(sha(store.replay())==packet['store_sha256'],'STORE_DRIFT')
        persist_exact(folder/'prompt.txt',prompt)
    model_handoff(journal,model_key);profile.guard(journal)
    plans.describe(journal,materializations_pin,store,free_disk_bytes=free_disk_bytes)
    require(sha(store.replay())==packet['store_sha256'],'STORE_DRIFT')
    result=verified_decision(journal,model_key)
    require(result['decision']=='PASS_R6_FINAL_PLANS' and result['must_fix']==[]
        and result['changed_files']==[] and result['reviewer']=='r6-independent-final-plans'
        and result['materializations_sha256']==materializations_pin['sha256'],'REJECTED')
    require(materializations_pin in result['evidence_pins'],'EVIDENCE')
    for pin in result['evidence_pins']:checked(pin)
    require(result['remaining_budget']['automatic_sessions_remaining']==
        packet['caps']['automatic_model_sessions']-packet['budget']['automatic_model_sessions']-1,'BUDGET')
    receipt=json.loads(checked(journal.read()['events'][model_key]['receipt']).read_text())
    review=receipt['last_message']
    require(json.loads(checked(review).read_text())==result,'RECEIPT_OUTPUT')
    output=dict(status='FINAL_BATCH_PLANS_REVIEWED_NOT_AUTHORIZED',materializations=materializations_pin,review=review,
        event_key=model_key,scientific_acceptance=False,dispatch_authorized=False)
    persist_exact(folder/'result.json',canonical(output)+b'\n')
    return output


def verify(journal,model_key,review_pin,materializations_pin,store):
    """Verify the original batch receipt after earlier assignments changed Store."""
    state=journal.read();entry=state['events'][model_key];identity=entry['identity']
    require(entry['kind']=='model' and identity['role']=='final-plans-review'
        and identity['goal_id']==state['goal_id'] and identity['job_id']=='FINAL-BATCH'
        and sha(identity)==model_key and identity['input_sha256']==sha(entry['request']),'MODEL_IDENTITY')
    prefix='FINAL_PLANS_REVIEW:'
    require(identity['event_id'].startswith(prefix),'MODEL_EVENT')
    cpu_key=identity['event_id'][len(prefix):];cpu=state['events'][cpu_key];packet=cpu['request']['packet']
    require(cpu['kind']=='cpu' and cpu['identity']['role']=='final-plans-input'
        and cpu['identity']['goal_id']==state['goal_id'] and sha(cpu['identity'])==cpu_key
        and cpu['identity']['input_sha256']==sha(cpu['request'])
        and packet['materializations']==materializations_pin and packet['store_root']==str(store.root.resolve()),'PACKET')
    require(packet['implementation']==file_pin(__file__) and packet['validator']==file_pin(plans.__file__),'IMPLEMENTATION')
    for field in ('materializations','executable','implementation','validator'):checked(packet[field])
    profile.reservation(state,cpu_key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    profile.reservation(state,model_key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},'confirmation_delivery')
    require(entry['request']['schema']==file_pin(SCHEMA) and entry['request']['sandbox']=='read-only'
        and entry['request']['executable']==packet['executable'],'MODEL_CONTRACT')
    receipt=json.loads(checked(entry['receipt']).read_text())
    require(receipt['last_message']==review_pin,'REVIEW_PIN')
    result=verified_decision(journal,model_key)
    require(result==json.loads(checked(review_pin).read_text()) and result['decision']=='PASS_R6_FINAL_PLANS'
        and result['must_fix']==[] and result['changed_files']==[]
        and result['reviewer']=='r6-independent-final-plans'
        and result['materializations_sha256']==materializations_pin['sha256']
        and materializations_pin in result['evidence_pins'],'REJECTED')
    for pin in result['evidence_pins']:checked(pin)
    require(result['remaining_budget']['automatic_sessions_remaining']==
        packet['caps']['automatic_model_sessions']-packet['budget']['automatic_model_sessions']-1,'BUDGET')
    return result
