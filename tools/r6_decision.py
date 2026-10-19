"""One post-analysis decision per job, with budgeted handoff and proposal-only output."""
import json
from pathlib import Path

from rlflow.store import canonical,sha
from tools.r6_recovery import RecoveryError,checked,file_pin
from tools.r6_budget import totals
from tools.r6_batch import model_handoff,verified_decision
from tools.r6_decision_packet import prepare,persist_exact,RESERVATION as PACKET_BYTES
from tools.r6_model_worker import validate_request

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'docs/experiments/r6-robustness-auto/prompts/post-analysis-decision.schema.json'
MODEL_CAP=64*1024**2
CLI=Path('/usr/local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex')
CLI_SHA='9a820c17865fa825d04db416818679a9d63bd72e50835c396f496e5684626c9c'


def check_budget_snapshot(state,packet_input,*,model_reserved):
    packet=json.loads(checked(packet_input['packet']).read_text())
    snapshot=packet['budget_before_packet_reservation'];expected=dict(snapshot['charged'])
    expected['output_actual_plus_unreleased_reservations_bytes']+=PACKET_BYTES
    if model_reserved:
        expected['output_actual_plus_unreleased_reservations_bytes']+=MODEL_CAP+65536
        expected['automatic_model_sessions']+=1
    budget=state['budget']
    if (totals(budget,enforce_caps=False)!=expected or snapshot['caps']!=budget['caps']
            or snapshot['protected']!=budget['protected_confirmation_delivery_repair']
            or snapshot['protected_model_sessions']!=budget['protected_model_sessions']):
        raise RecoveryError('DECISION_BUDGET_SNAPSHOT_STALE')


def register(journal,job_key,store,*,executable,free_disk_bytes,sandbox='read-only'):
    journal.guard()
    exe=checked(executable);state=journal.read()
    if state['goal_id']=='R6-CPU-TEST-FIXTURE':
        if not exe.is_relative_to(journal.directory.parent.resolve()):raise RecoveryError('DECISION_FIXTURE_EXECUTABLE')
    elif executable!={'path':str(CLI),'sha256':CLI_SHA}:raise RecoveryError('DECISION_PINNED_CLI_REQUIRED')
    packet=prepare(journal,job_key,store,free_disk_bytes=free_disk_bytes)
    state=journal.read();job=journal.canonical_job(job_key,state=state)
    request=dict(prompt=packet['prompt'],schema=file_pin(SCHEMA),executable=executable,cwd=str(ROOT),
        sandbox=sandbox,timeout_seconds=3600,output_cap_bytes=MODEL_CAP,
        stop_paths=[str((journal.directory.parent/'STOP').resolve()),str((journal.directory/'STOP').resolve())])
    validate_request(request)
    for prior in state['events'].values():
        if (prior['kind']=='model' and prior['identity']['job_id']==job['identity']['job_id']
                and prior['identity']['role']=='post-analysis-decision' and prior['request']!=request):
            raise RecoveryError('DECISION_JOB_ALREADY_BOUND')
    if not any(e['kind']=='model' and e['identity']['job_id']==job['identity']['job_id']
            and e['identity']['role']=='post-analysis-decision' for e in state['events'].values()):
        check_budget_snapshot(state,packet,model_reserved=False)
    event=dict(goal_id=state['goal_id'],event_id='DECISION:'+job_key,job_id=job['identity']['job_id'],
        input_sha256=sha(request),role='post-analysis-decision')
    key,_=journal.intent(event,kind='model',request=request,
        resources={'automatic_model_sessions':1,'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},
        purpose=job['purpose'],free_disk_bytes=free_disk_bytes)
    state=journal.read();entry=state['events'][key]
    if entry.get('decision_input') is not None and entry['decision_input']!=packet:
        raise RecoveryError('DECISION_INPUT_DRIFT')
    # A crash before this assignment is recoverable: the immutable prompt pin
    # was already part of the reserved model request and cannot change.
    entry['decision_input']=packet;journal.save(state)
    return key


def record(journal,key,store):
    journal.guard();state=journal.read();entry=state['events'][key]
    if (entry['kind']!='model' or entry['identity']['role']!='post-analysis-decision'
            or sha(entry['identity'])!=key or entry['identity']['input_sha256']!=sha(entry['request'])):
        raise RecoveryError('DECISION_EVENT_IDENTITY')
    packet_input=entry['decision_input'];packet=json.loads(checked(packet_input['packet']).read_text())
    job_key=packet_input['job_event_key'];job=journal.canonical_job(job_key,state=state)
    # Revalidate the original analysis, not only the model's evidence citations.
    from tools.r6_analysis_binding import bind_analysis
    binding=json.loads(checked(job['analysis']).read_text())
    if (job['analysis']!=packet['analysis'] or binding!=bind_analysis(job,job_key,binding['report'],store)
            or entry['identity']['job_id']!=job['identity']['job_id'] or entry['identity']['event_id']!='DECISION:'+job_key
            or entry['request']['prompt']!=packet_input['prompt'] or entry['request']['schema']!=file_pin(SCHEMA)):
        raise RecoveryError('DECISION_ANALYSIS_DRIFT')
    result=verified_decision(journal,key)
    if result['packet_sha256']!=packet_input['packet']['sha256']:raise RecoveryError('DECISION_PACKET_IDENTITY')
    if result['changed_files']:raise RecoveryError('DECISION_READ_ONLY_REQUIRED')
    pins=result['evidence_pins']
    if packet_input['packet'] not in pins or packet['report'] not in pins:raise RecoveryError('DECISION_EVIDENCE_REQUIRED')
    if len({p['path'] for p in pins})!=len(pins):raise RecoveryError('DECISION_DUPLICATE_EVIDENCE')
    for pin in pins:checked(pin)
    if packet['issues'] and result['decision']!='NEEDS_REPAIR':raise RecoveryError('DECISION_UNRESOLVED_EVIDENCE')
    if (result['decision']=='NEEDS_REPAIR') != bool(result['must_fix']):raise RecoveryError('DECISION_REPAIR_FINDINGS')
    expected=packet['budget_before_packet_reservation']['remaining']['automatic_model_sessions']-1
    if result['remaining_budget']['automatic_sessions_remaining']!=expected:raise RecoveryError('DECISION_BUDGET_MISMATCH')
    route={'NEEDS_REPAIR':'REPAIR_AND_REVIEW','PROPOSE_CANDIDATE':'REGISTER_CANDIDATE_AND_REVIEW',
        'PROPOSE_CONFIRMATION':'FREEZE_RECIPE_AND_REVIEW','PROPOSE_BASELINE_VALIDATION':'REGISTER_DISTINCT_STREAM_VALIDATION',
        'REQUEST_FINAL_REVIEW':'FINAL_REQUIREMENTS_AUDIT'}[result['decision']]
    output=dict(schema_version='r6-post-analysis-proposal/1',model_event_key=key,job_event_key=job_key,
        packet=packet_input['packet'],analysis=job['analysis'],receipt=entry['receipt'],decision=result,
        next_step=route,scientific_acceptance=False,dispatch_authorized=False)
    path=journal.directory/'decisions'/packet_input['packet_event_key']/'decision.json'
    data=canonical(output)+b'\n'
    if len(data)>65536:raise RecoveryError('DECISION_OUTPUT_BOUND')
    if entry.get('decision_record') and checked(entry['decision_record'])!=path.resolve():raise RecoveryError('DECISION_RECORD_PATH')
    persist_exact(path,data);entry['decision_record']=file_pin(path)
    # The proposal is deliberately separate from native job/control status.
    journal.save(state)
    return output


def handoff(journal,key,store,*,poll_seconds=15):
    journal.require_writer()
    entry=journal.read()['events'][key]
    if not entry.get('decision_input'):raise RecoveryError('DECISION_INPUT_NOT_BOUND')
    if not entry.get('dispatch_claim') and not entry.get('receipt'):
        check_budget_snapshot(journal.read(),entry['decision_input'],model_reserved=True)
    model_handoff(journal,key,poll_seconds=poll_seconds)
    return record(journal,key,store)
