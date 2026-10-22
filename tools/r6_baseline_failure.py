"""One bounded repair disposition for an exact failed first-baseline boundary.

No job retry, code writer, authority issuance or inferred receipt recovery. The
existing replacement registrars revalidate cleanup, inputs and reservations.
"""
import copy
import hashlib
import json
from pathlib import Path
from rlflow.jobs import owns
from rlflow.store import sha,canonical
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_worker_processes import cleanup_verified
from tools.r6_decision_packet import persist_exact
from tools.r6_batch import model_handoff,verified_decision
from tools.r6_budget import admissible
from tools import r6_profile_preparation as profile,r6_initial_baseline as initial
from tools import r6_analysis_repair as analysis,r6_decision_repair as decision_repair
from tools.r6_decision import CLI,CLI_SHA,MODEL_CAP,model_stop_paths

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'docs/experiments/r6-robustness-auto/prompts/baseline-failure.schema.json'
ROLE='baseline-failure-disposition'
CAP=256*1024


def require(ok,reason):
    if not ok:raise RecoveryError('BASELINE_FAILURE_'+reason)


def failed(journal,key):
    e=journal.read()['events'][key]
    if e.get('status')!='RECEIPT_RECORDED' or not e.get('receipt'):return False
    r=json.loads(checked(e['receipt']).read_text())
    require(r.get('event_key')==key and r.get('request_sha256')==sha(e['request'])
        and r.get('process')==e.get('process') and type(r.get('exit_code')) is int
        and 'error' in r and cleanup_verified(r.get('cleanup')),'RECEIPT_OR_CLEANUP')
    for ident in (r.get('process'),r.get('child_identity')):
        require(ident is None or (isinstance(ident,dict) and set(ident)=={'pid','pgid','boot_id','starttime'}
            and type(ident['pid']) is int and ident['pid']>0 and type(ident['pgid']) is int
            and ident['pgid']>0 and isinstance(ident['boot_id'],str) and bool(ident['boot_id'])
            and isinstance(ident['starttime'],str) and ident['starttime'].isdigit()),'PROCESS_IDENTITY')
        require(ident is None or not owns(ident),'PROCESS_ALIVE')
    for name in ('events','stderr','stdout','last_message'):
        if r.get(name):checked(r[name])
    if r['exit_code']!=0 or r['error'] is not None or r.get('schema_valid') is False:return True
    if e['identity']['role']==initial.ROLE:
        out=verified_decision(journal,key)
        return out['decision']=='MUST_FIX' and bool(out['must_fix'])
    return False


def select(journal,initial_args):
    """Only an active exact baseline attempt, never an unrelated old failure."""
    state=journal.read();plan_pin=initial_args['plan_pin'];plan=json.loads(checked(plan_pin).read_text())
    if set(plan)!=initial.FIELDS:return None
    spec=json.loads(checked(plan['spec_pin']).read_text())
    if spec.get('id')!=initial.EXPERIMENT:return None
    jid='J-'+sha(dict(id=spec['id'],spec=spec))[:16]
    jobs=[k for k,e in state['events'].items() if e['kind']=='job' and e['identity']['job_id']==jid]
    require(len(jobs)<=1,'JOB_ALIAS')
    if not jobs:
        if any(e['identity'].get('role')=='initial-baseline-authorization' for e in state['events'].values()):return None
        key=initial.active_review(journal,plan_pin)
        return dict(boundary='initial_review',failed_key=key,job_key=None) if key and failed(journal,key) else None
    job_key=jobs[0];job=journal.canonical_job(job_key)
    if not job.get('closure'):return None
    checked(job['closure'])
    for boundary,key in [('analysis',analysis.active(journal,job_key)),('decision',decision_repair.active(journal,job_key))]:
        if key and failed(journal,key):return dict(boundary=boundary,failed_key=key,job_key=job_key)
    return None


def capture(journal,controller_key,initial_args,store,*,free_disk_bytes):
    profile.guard(journal);state=journal.read();owner=state['events'][controller_key]
    require(owner['identity']['role']=='r6-controller' and owner['cursor']['phase']=='FIRST_JOB'
        and not owner['cursor']['completed_jobs'] and owner['request']['store_root']==str(store.root.resolve()),'CONTROLLER')
    target=select(journal,initial_args)
    if target is None:return None
    old=state['events'][target['failed_key']];name='BASELINE_FAILURE:'+target['failed_key']+':'+old['receipt']['sha256']
    previous=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(previous)<=1,'INPUT_ALIAS')
    if previous:
        key,entry=previous[0];packet=entry['request']
        require(packet['cursor']==owner['cursor'] and packet['initial']==initial_args
            and packet['store_sha256']==sha(store.replay()),'INPUT_DRIFT')
    else:
        packet=dict(controller_key=controller_key,cursor=copy.deepcopy(owner['cursor']),initial=initial_args,
            target=target,failed_receipt=old['receipt'],failed_request=old['request'],
            store_sha256=sha(store.replay()),implementation=file_pin(__file__),
            model_remaining=state['budget']['caps']['automatic_model_sessions']-admissible(state['budget'])['automatic_model_sessions'])
        event=dict(goal_id=state['goal_id'],event_id=name,job_id=old['identity']['job_id'],input_sha256=sha(packet),role='baseline-failure-input')
        key,_=journal.intent(event,kind='cpu',request=packet,
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='repair',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'repair')
    folder=journal.directory/'baseline-failure'/key
    persist_exact(folder/'packet.json',canonical(packet)+b'\n')
    return key


def register(journal,input_key,*,free_disk_bytes):
    profile.guard(journal);state=journal.read();packet=state['events'][input_key]['request'];checked(packet['implementation'])
    profile.reservation(state,input_key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'repair')
    exe=packet['initial']['review_executable'];path=checked(exe)
    if state['goal_id']=='R6-CPU-TEST-FIXTURE':require(path.is_relative_to(journal.directory.parent.resolve()),'FIXTURE_CLI')
    else:require(exe==dict(path=str(CLI),sha256=CLI_SHA),'CLI')
    folder=journal.directory/'baseline-failure'/input_key;packet_pin=file_pin(folder/'packet.json')
    require(json.loads(checked(packet_pin).read_text())==packet,'PACKET')
    text=('Read-only diagnosis of ONE failed first-baseline boundary. Exit with a bounded repair disposition. '
        'Do not edit files, spawn agents, query GPUs, simulate, submit jobs, issue permissions or change runtime. '
        'Read the exact failed request/receipt and original error evidence. Preserve all prior attempts and charges. '
        'Use REPLACE_FAILED_ATTEMPT only with concrete evidence explaining why corrected input resolves the failure. '
        'Initial-review replacement is an independent fresh review, never approval; decision replacement receives '
        'your repair reasoning. For analysis only a changed timeout between 1 and 3600 is available here: require '
        'evidence that the old bound caused the failure; this cannot repair code. Otherwise return REPAIR_CODE_REQUIRED '
        'or EXTERNAL_BLOCKER with exact must_fix, without pretending the failure is resolved. No authentication '
        'fallback or blind transient retry. timeout_seconds=0 for non-analysis or non-replacement. '
        'Include packet and failed_receipt in evidence_pins, changed_files=[], next_plan, and remaining_budget '
        'automatic_sessions_remaining = model_remaining minus one. Packet follows:\n'
        +json.dumps(dict(packet,packet_pin=packet_pin),sort_keys=True)+'\n').encode()
    require(len(text)<=65536,'CONTEXT_CAP')
    prompt=folder/'prompt.txt'
    request=dict(prompt=dict(path=str(prompt.resolve()),sha256=hashlib.sha256(text).hexdigest()),
        schema=file_pin(SCHEMA),executable=exe,cwd=str(ROOT),sandbox='read-only',timeout_seconds=3600,
        output_cap_bytes=MODEL_CAP,stop_paths=model_stop_paths(journal))
    event=dict(goal_id=state['goal_id'],event_id='BASELINE_FAILURE_MODEL:'+input_key,
        job_id=state['events'][input_key]['identity']['job_id'],input_sha256=sha(request),role=ROLE)
    previous=[e for e in state['events'].values() if e['identity'].get('event_id')==event['event_id']]
    require(len(previous)<=1 and all(e['identity']==event for e in previous),'MODEL_ALIAS')
    if not previous:
        require(packet['model_remaining']==state['budget']['caps']['automatic_model_sessions']-admissible(state['budget'])['automatic_model_sessions'],'BUDGET_STALE')
        if state['goal_id']!='R6-CPU-TEST-FIXTURE':
            from tools.r6_review_access import check
            check(exe,packet_pin)
    key,_=journal.intent(event,kind='model',request=request,resources={'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},purpose='repair',free_disk_bytes=free_disk_bytes)
    persist_exact(prompt,text)
    return key


def validated(journal,key):
    state=journal.read();e=state['events'][key];name=e['identity']['event_id']
    require(e['identity']['role']==ROLE and name.startswith('BASELINE_FAILURE_MODEL:'),'MODEL_IDENTITY')
    input_key=name.split(':',1)[1];packet=state['events'][input_key]['request']
    profile.reservation(state,input_key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'repair')
    profile.reservation(state,key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},'repair')
    require(e['request']['schema']==file_pin(SCHEMA) and e['request']['sandbox']=='read-only'
        and e['request']['executable']==packet['initial']['review_executable'],'MODEL_CONTRACT')
    receipt=json.loads(checked(e['receipt']).read_text())
    require(receipt['event_key']==key and receipt['request_sha256']==sha(e['request'])
        and receipt.get('process')==e.get('process'),'MODEL_RECEIPT')
    out=verified_decision(journal,key);packet_pin=file_pin(journal.directory/'baseline-failure'/input_key/'packet.json')
    require(json.loads(checked(packet_pin).read_text())==packet and out['packet_sha256']==packet_pin['sha256']
        and out['changed_files']==[] and out['remaining_budget']['automatic_sessions_remaining']==packet['model_remaining']-1,'OUTPUT_BINDING')
    for pin in out['evidence_pins']:checked(pin)
    require(packet_pin in out['evidence_pins'] and packet['failed_receipt'] in out['evidence_pins'],'OUTPUT_EVIDENCE')
    return packet,out


def prepare(journal,controller_key,store,*,free_disk_bytes):
    profile.guard(journal);state=journal.read();owner=state['events'][controller_key];cursor=owner['cursor']
    require(cursor['phase']=='BASELINE_FAILURE_HANDOFF','PHASE')
    input_key=cursor['baseline_failure_input'];packet=state['events'][input_key]['request']
    require(packet['controller_key']==controller_key and packet['store_sha256']==sha(store.replay())
        and dict(packet['cursor'],phase='BASELINE_FAILURE_HANDOFF',baseline_failure_input=input_key)==cursor,'CURSOR_OR_STORE')
    checked(packet['implementation'])
    model_key=register(journal,input_key,free_disk_bytes=free_disk_bytes)
    model_handoff(journal,model_key)
    packet,out=validated(journal,model_key)
    require(packet['store_sha256']==sha(store.replay()),'STORE_CHANGED')
    if out['decision']!='REPLACE_FAILED_ATTEMPT':
        require(bool(out['must_fix']) and out['timeout_seconds']==0,'UNRESOLVED_OUTPUT')
        return dict(status=out['decision'],event_key=model_key,must_fix=out['must_fix'],goal_complete=False)
    require(not out['must_fix'] and bool(out['summary'].strip()),'UNRESOLVED_REPLACEMENT')
    target=packet['target'];old=journal.read()['events'][target['failed_key']]
    require(old.get('receipt')==packet['failed_receipt'] and failed(journal,target['failed_key']),'PREDECESSOR_DRIFT')
    document=dict(predecessor=target['failed_key'],predecessor_receipt=packet['failed_receipt'],
        reason=out['summary'],evidence_pins=out['evidence_pins']+[json.loads(checked(journal.read()['events'][model_key]['receipt']).read_text())['last_message']])
    document.update({'plan':packet['initial']['plan_pin']} if target['boundary']=='initial_review' else {'job_event_key':target['job_key']})
    path=journal.directory/'baseline-failure'/input_key/'repair.json';persist_exact(path,canonical(document)+b'\n');pin=file_pin(path)
    if target['boundary']=='analysis':
        receipt=json.loads(checked(packet['failed_receipt']).read_text())
        require(type(out['timeout_seconds']) is int and 1<=out['timeout_seconds']<=3600
            and out['timeout_seconds']>old['request']['timeout_seconds']
            and 'CPU_ANALYSIS_TIMEOUT' in str(receipt.get('error'))
            and receipt.get('elapsed_seconds',0)>=old['request']['timeout_seconds'],'ANALYSIS_SETTINGS')
        replacement=analysis.register(journal,target['job_key'],store,pin,free_disk_bytes=free_disk_bytes,
            python=old['request']['python'],program=old['request']['program'],timeout_seconds=out['timeout_seconds'])
    else:
        require(out['timeout_seconds']==0,'NON_ANALYSIS_SETTINGS')
        if target['boundary']=='initial_review':
            replacement=initial.register_review(journal,packet['initial']['plan_pin'],store,
                executable=packet['initial']['review_executable'],free_disk_bytes=free_disk_bytes,repair_pin=pin)
        else:
            from tools.r6_decision import register as register_decision
            replacement=register_decision(journal,target['job_key'],store,executable=packet['initial']['review_executable'],
                free_disk_bytes=free_disk_bytes,sandbox='read-only',repair_pin=pin)
    state=journal.read();owner=state['events'][controller_key];require(owner['cursor']==cursor,'CURSOR_DRIFT')
    transition=dict(input_key=input_key,disposition_key=model_key,predecessor=target['failed_key'],replacement=replacement)
    owner.setdefault('baseline_failure_history',[]).append(transition)
    owner.update(cursor=copy.deepcopy(packet['cursor']),status='BASELINE_REPLACEMENT_REGISTERED')
    journal.save(state)
    return dict(status='BASELINE_REPLACEMENT_REGISTERED',**transition,goal_complete=False)


def pre_analysis_model(journal,key,job_key):
    """Admit only verified initial reviews/repair judgments, never decisions."""
    state=journal.read();entry=state['events'][key];job=journal.canonical_job(job_key)
    role=entry['identity'].get('role')
    if role==ROLE:
        packet,_=validated(journal,key)
        if packet['target']['boundary']=='analysis':return packet['target']['job_key']==job_key
        if packet['target']['boundary']!='initial_review':return False
        plan_pin=packet['initial']['plan_pin']
    elif role==initial.ROLE:
        edges=[e for e in state['events'].values() if e['identity'].get('role')=='initial-baseline-authorization'
            and e['identity']['job_id']==job['identity']['job_id']]
        if len(edges)!=1 or not edges[0].get('authorization_result'):return False
        plan_pin=edges[0]['request']['plan']
    else:return False
    plan=json.loads(checked(plan_pin).read_text())
    if plan['spec_pin']!=job['request']['spec']:return False
    tip=initial.active_review(journal,plan_pin)
    if tip is None:return False
    latest=state['events'][tip];receipt=json.loads(checked(latest['receipt']).read_text())
    initial.verify_review(journal,plan_pin,tip,receipt['last_message'],plan,job['identity']['job_id'])
    prefix='INITIAL_BASELINE_REVIEW:'+plan_pin['sha256']
    prior=state['events'][packet['target']['failed_key']] if role==ROLE else entry
    return prior['identity']['event_id']==prefix or prior['identity']['event_id'].startswith(prefix+':REPAIR:')
