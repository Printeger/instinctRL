"""Bounded post-analysis model input. Immutable packets are not live permissions."""
import hashlib
import json
from pathlib import Path

from rlflow.store import atomic,canonical,sha
from tools.r6_recovery import RecoveryError,checked,file_pin
from tools.r6_budget import totals
from tools.r6_analysis_binding import bind_analysis

ROOT=Path(__file__).resolve().parents[1]
CAP=65536
RESERVATION=256*1024
INSTRUCTIONS='''You are the R6 post-analysis decision worker. Complete one read-only decision and exit.
Do not launch jobs, query GPUs, train, simulate, change files or wait for training.
Use the pinned report and original evidence as needed; the summary is not independent acceptance.
Preserve all failed/missing lanes, native failures, actor boundary, frozen metrics/splits and budgets.
Prioritize implementation/evidence repair when the packet lists issues. Otherwise propose the next
scientific step with an explicit hypothesis, comparison and required review. No PASS authorizes dispatch.
Do not claim learning gain from controller changes, loss reduction or q-positive coverage alone.
Return the requested structured decision with this packet's identity and exact evidence pins.
The controller alone owns budget, dispatch and terminal state. Never reinterpret unknown usage as zero.
Packet JSON follows:\n'''


def pin_bytes(path,data):
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(data).hexdigest())


def persist_exact(path,data):
    if path.is_symlink():raise RecoveryError('DECISION_PACKET_SYMLINK')
    if path.exists():
        if path.read_bytes()!=data:raise RecoveryError('DECISION_PACKET_DRIFT')
    else:atomic(path,data)


def prepare(journal,job_key,store,*,free_disk_bytes):
    journal.guard();state=journal.read();job=journal.canonical_job(job_key,state=state)
    if state.get('current_job') is not None or state.get('current_model_event') is not None:
        raise RecoveryError('DECISION_PACKET_NOT_IDLE')
    if any(e['kind']=='cpu' and e.get('dispatch_claim') and not e.get('receipt') for e in state['events'].values()):
        raise RecoveryError('DECISION_PACKET_CPU_UNRESOLVED')
    if not job.get('analysis'):raise RecoveryError('DECISION_PACKET_ANALYSIS_REQUIRED')
    binding=json.loads(checked(job['analysis']).read_text())
    if binding!=bind_analysis(job,job_key,binding['report'],store):raise RecoveryError('DECISION_PACKET_BINDING_DRIFT')
    old=[(k,e) for k,e in state['events'].items() if e['identity']['job_id']==job['identity']['job_id']
        and e['identity']['role']=='decision-packet']
    if len(old)>1:raise RecoveryError('DECISION_PACKET_ALIAS')
    if old:
        key,entry=old[0];request=entry['request'];packet=request['packet']
        if (entry['kind']!='cpu' or request['analysis']!=job['analysis'] or packet['report']!=binding['report']
                or sha(entry['identity'])!=key or entry['identity']['input_sha256']!=sha(request)):
            raise RecoveryError('DECISION_PACKET_REPLAY_DRIFT')
    else:
        report=json.loads(checked(binding['report']).read_text());summary=report['summary']
        if len(canonical(summary))>48*1024:raise RecoveryError('DECISION_PACKET_SUMMARY_BOUND')
        used=totals(state['budget'],enforce_caps=False)
        packet=dict(schema_version='r6-decision-packet/1',goal_id=state['goal_id'],job_event_key=job_key,
            job_id=job['identity']['job_id'],analysis=job['analysis'],report=binding['report'],
            original_evidence_inventory=job['inventory'],protocol=binding['protocol'],
            native_status=binding['native_status'],native_exit_code=binding['native_exit_code'],
            issues=binding['issues'],summary=summary,
            budget_before_packet_reservation=dict(caps=state['budget']['caps'],charged=used,
                remaining={k:v-used[k] for k,v in state['budget']['caps'].items()},
                protected=state['budget']['protected_confirmation_delivery_repair'],
                protected_model_sessions=state['budget']['protected_model_sessions']),
            capacity_pass=state['budget']['capacity_pass'],scientific_acceptance=False,
            source=file_pin(Path(__file__)))
        request=dict(analysis=job['analysis'],packet=packet)
        # Both duplicate stored packet bytes and the model prompt fit within
        # the reserved256KiB. Crash recovery uses this committed request.
        if len(canonical(request))+len(INSTRUCTIONS.encode())>CAP:raise RecoveryError('DECISION_PACKET_CONTEXT_CAP')
        event=dict(goal_id=state['goal_id'],event_id='DECISION_PACKET:'+job_key,job_id=job['identity']['job_id'],
            input_sha256=sha(request),role='decision-packet')
        key,_=journal.intent(event,kind='cpu',request=request,
            resources={'output_actual_plus_unreleased_reservations_bytes':RESERVATION},
            purpose=job['purpose'],free_disk_bytes=free_disk_bytes)
    checked(packet['source'])
    directory=journal.directory/'decisions'/key
    packet_data=canonical(packet)+b'\n'
    prompt=('Input packet path: '+str((directory/'packet.json').resolve())+'\nInput packet SHA256: '+hashlib.sha256(packet_data).hexdigest()+'\n'+INSTRUCTIONS+json.dumps(packet,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
    if len(prompt)>CAP:raise RecoveryError('DECISION_PACKET_CONTEXT_CAP')
    directory=journal.directory/'decisions'/key
    persist_exact(directory/'packet.json',packet_data);persist_exact(directory/'prompt.txt',prompt)
    result=dict(schema_version='r6-decision-input/1',packet_event_key=key,job_event_key=job_key,
        packet=file_pin(directory/'packet.json'),prompt=file_pin(directory/'prompt.txt'),
        scientific_acceptance=False,dispatch_authorized=False)
    data=canonical(result)+b'\n';persist_exact(directory/'input.json',data)
    state=journal.read();entry=state['events'][key]
    entry['packet_artifact']=file_pin(directory/'input.json');entry['status']='CPU_PACKET_RECORDED'
    # Do not overwrite overrun, STOP or native failure statuses.
    journal.save(state)
    return result
