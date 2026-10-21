"""Independent final model freeze review with original receipt provenance."""
import json
from pathlib import Path
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision import CLI,CLI_SHA,MODEL_CAP,model_stop_paths
from tools.r6_decision_packet import persist_exact,pin_bytes
from tools.r6_batch import model_handoff,verified_decision
from tools.r6_budget import totals
from tools import r6_profile_preparation as profile
from tools import r6_final_models as models

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'docs/experiments/r6-robustness-auto/prompts/final-freeze-review.schema.json'
CAP=256*1024


def require(value,reason):
    if not value:raise RecoveryError('FINAL_FREEZE_REVIEW_'+reason)


def prepare(journal,freeze_pin,store,*,executable,free_disk_bytes):
    journal.require_writer();state=journal.read();name='FINAL_FREEZE_INPUT:'+freeze_pin['sha256']
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    if prior:
        key,entry=prior[0];packet=entry['request']['packet']
        require(packet['freeze']==freeze_pin and packet['executable']==executable,'INPUT_DRIFT')
    else:
        profile.guard(journal);models.validate(journal,json.loads(checked(freeze_pin).read_text()),store)
        exe=checked(executable)
        if state['goal_id']=='R6-CPU-TEST-FIXTURE':require(exe.is_relative_to(journal.directory.parent.resolve()),'FIXTURE_EXE')
        else:require(executable==dict(path=str(CLI),sha256=CLI_SHA),'CLI')
        packet=dict(freeze=freeze_pin,executable=executable,implementation=file_pin(__file__),validator=file_pin(models.__file__),
            baseline_validator=file_pin(models.baseline_validation.__file__),
            store_sha256=sha(store.replay()),budget=totals(state['budget']),caps=state['budget']['caps'],
            protected=state['budget']['protected_confirmation_delivery_repair'])
        event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-BATCH',input_sha256=sha(dict(packet=packet)),role='final-freeze-input')
        key,_=journal.intent(event,kind='cpu',request=dict(packet=packet),
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    for field in ('freeze','executable','implementation','validator','baseline_validator'):checked(packet[field])
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'confirmation_delivery')
    folder=journal.directory/'final-freeze-review'/key
    prompt=('Independently review the final R6 model freeze, read-only, then exit. Do not change files, '
        'Journal/Store, run GPU/simulation/training, generate or reveal entropy, submit jobs or spawn agents. '
        'Read the exact freeze and original evidence. For THREE_CONTINUATIONS verify all three actual common-u640 confirmation '
        'trajectories, preregistered distinct RNG streams and their real execution, full frozen recipe and '
        'terminal checkpoint rule, loaded actor/critic/Adam and first-rollout probability gates, source/method '
        'identity and complete raw inventories/cleanup. Adjudicate native failures explicitly; structural '
        'checks alone are insufficient. Verify the selected baseline identity and that no final samples '
        'were used to choose models. Audit physics and measurement/time provenance, original actor boundary, '
        'control-versus-learning attribution and all pending semantic audit requirements. Require full '
        'remaining final paired evaluation/delivery/repair capacity under every ledger cap, counting failed '
        'and unreleased reservations. For BASELINE_SUFFICIENT instead verify the unchanged u640/method/ICS .15, '
        'zero training, complete original and all frozen target development gates, preserved stress failures, '
        'and a fixed plan to use all three distinct final environment realizations with full suites each. '
        'Do not require training in that branch or claim its three final validations have already happened. '
        'PASS_R6_FINAL_FREEZE approves model identities for prospective final '
        'evaluation only, never target-domain success or live dispatch. Return MUST_FIX with concrete '
        'findings if any evidence or capacity is incomplete. changed_files=[]; evidence_pins must include '
        'the freeze and all originals you relied on. reviewer=r6-independent-final-freeze. '
        'freeze_sha256 must match packet.freeze.sha256. Remaining sessions=caps minus budget minus one.\n'
        +json.dumps(packet,sort_keys=True)+'\n').encode()
    require(len(prompt)<=65536,'PROMPT_CAP')
    request=dict(prompt=pin_bytes(folder/'prompt.txt',prompt),schema=file_pin(SCHEMA),executable=executable,
        cwd=str(ROOT),sandbox='read-only',timeout_seconds=3600,output_cap_bytes=MODEL_CAP,stop_paths=model_stop_paths(journal))
    event=dict(goal_id=state['goal_id'],event_id='FINAL_FREEZE_REVIEW:'+key,job_id='FINAL-BATCH',input_sha256=sha(request),role='final-freeze-review')
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
        profile.guard(journal);models.validate(journal,json.loads(checked(freeze_pin).read_text()),store)
        require(sha(store.replay())==packet['store_sha256'],'STORE_DRIFT')
        persist_exact(folder/'prompt.txt',prompt)
    model_handoff(journal,model_key);profile.guard(journal)
    models.validate(journal,json.loads(checked(freeze_pin).read_text()),store)
    require(sha(store.replay())==packet['store_sha256'],'STORE_DRIFT')
    result=verified_decision(journal,model_key)
    require(result['decision']=='PASS_R6_FINAL_FREEZE' and result['must_fix']==[]
        and result['changed_files']==[] and result['reviewer']=='r6-independent-final-freeze'
        and result['freeze_sha256']==freeze_pin['sha256'],'REJECTED')
    require(freeze_pin in result['evidence_pins'],'EVIDENCE')
    for pin in result['evidence_pins']:checked(pin)
    require(result['remaining_budget']['automatic_sessions_remaining']==
        packet['caps']['automatic_model_sessions']-packet['budget']['automatic_model_sessions']-1,'BUDGET')
    receipt=json.loads(checked(journal.read()['events'][model_key]['receipt']).read_text())
    review=receipt['last_message']
    require(json.loads(checked(review).read_text())==result,'RECEIPT_OUTPUT')
    output=dict(status='FINAL_MODEL_FREEZE_REVIEWED_NOT_RELEASED',freeze=freeze_pin,review=review,
        event_key=model_key,holdout_released=False,scientific_acceptance=False,dispatch_authorized=False)
    persist_exact(folder/'result.json',canonical(output)+b'\n')
    return output
