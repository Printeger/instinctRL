"""Explicit, newly charged CPU analysis replacement chains; no live job retry."""
import json
from rlflow.jobs import owns
from rlflow.store import sha
from tools.r6_recovery import checked, RecoveryError
from tools.r6_worker_processes import cleanup_verified


def require(ok, reason):
    if not ok: raise RecoveryError('CPU_ANALYSIS_REPAIR_'+reason)


def predecessor(journal, pin, job_key):
    from tools.r6_cpu_analysis import RESERVATION
    from tools.r6_profile_preparation import reservation
    state=journal.read(); path=checked(pin)
    require(path.stat().st_size<=65536,'INPUT_CAP')
    data=json.loads(path.read_text())
    require(set(data)=={'job_event_key','predecessor','predecessor_receipt','reason','evidence_pins'}
        and data['job_event_key']==job_key and isinstance(data['reason'],str) and data['reason'].strip()
        and isinstance(data['evidence_pins'],list) and data['evidence_pins'],'FIELDS')
    for evidence in data['evidence_pins']: checked(evidence)
    key=data['predecessor']; old=state['events'][key]
    require(old['identity']['role']=='cpu-analysis' and old['request']['job_event_key']==job_key
        and old['identity']['job_id']==journal.canonical_job(job_key,state=state)['identity']['job_id']
        and isinstance(old.get('receipt'),dict) and old['receipt']==data['predecessor_receipt']
        and old['status']=='RECEIPT_RECORDED','PREDECESSOR')
    reservation(state,key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':RESERVATION},old['purpose'])
    receipt=json.loads(checked(old['receipt']).read_text())
    require(receipt['event_key']==key and receipt['request_sha256']==sha(old['request'])
        and receipt['job_id']==old['identity']['job_id']
        and receipt.get('process')==old.get('process')
        and type(receipt.get('exit_code')) is int and 'error' in receipt
        and (receipt['exit_code']!=0 or receipt['error'] is not None),'FAILED_RECEIPT')
    require(cleanup_verified(receipt.get('cleanup')),'CLEANUP')
    for ident in (receipt.get('process'),receipt.get('child_identity')):
        require(isinstance(ident,dict) and set(ident)=={'pid','pgid','boot_id','starttime'}
            and type(ident['pid']) is int and ident['pid']>0
            and type(ident['pgid']) is int and ident['pgid']>0
            and isinstance(ident['boot_id'],str) and bool(ident['boot_id'])
            and isinstance(ident['starttime'],str) and ident['starttime'].isdigit(),'PROCESS_IDENTITY')
        require(not owns(ident),'PROCESS_ALIVE')
    for field in ('stdout','stderr'):
        if receipt.get(field):checked(receipt[field])
    for output in receipt.get('outputs',{}).values():checked(output)
    return key,old


def active(journal, job_key):
    state=journal.read();job=journal.canonical_job(job_key,state=state)
    rows={k:e for k,e in state['events'].items() if e['kind']=='cpu'
        and e['identity'].get('role')=='cpu-analysis' and e['identity']['job_id']==job['identity']['job_id']}
    if not rows:return None
    roots=[k for k,e in rows.items() if 'repair' not in e['request']]
    require(len(roots)==1,'ROOT'); children={}
    for key,event in rows.items():
        require(sha(event['identity'])==key and event['identity']['input_sha256']==sha(event['request'])
            and event['request']['job_event_key']==job_key,'IDENTITY')
        name='CPU_ANALYSIS:'+job_key
        pin=event['request'].get('repair')
        if pin:
            parent,old=predecessor(journal,pin,job_key)
            require(parent in rows and parent not in children,'FORK')
            require(any(event['request'][n]!=old['request'][n]
                for n in ('python','program','sources','timeout_seconds')),'UNCHANGED_INPUT')
            children[parent]=key; name+=':REPAIR:'+pin['sha256']
        require(event['identity']['event_id']==name,'EVENT')
    key=roots[0];seen={key}
    while key in children:
        key=children[key];require(key not in seen,'CYCLE');seen.add(key)
    require(seen==set(rows),'ORPHAN')
    return key


def register(journal, job_key, store, repair_pin, *, free_disk_bytes,
             python=None, program=None, timeout_seconds=3600):
    from tools import r6_cpu_analysis as cpu
    from tools.r6_profile_preparation import guard
    from tools.r6_analysis_binding import analysis_context
    from tools.r6_recovery import file_pin
    guard(journal);state=journal.read();job=journal.canonical_job(job_key,state=state)
    context=analysis_context(job,job_key,store)
    from tools.r6_baseline_failure import pre_analysis_model
    require(not job.get('analysis') and not any(e['kind']=='model' and
        e['identity'].get('job_id')==job['identity']['job_id']
        and not pre_analysis_model(journal,k,job_key) for k,e in state['events'].items()),'ALREADY_USED')
    parent,old=predecessor(journal,repair_pin,job_key)
    request=dict(job_event_key=job_key,spec=job['request']['spec'],closure=job['closure'],inventory=job['inventory'],
        postrun=job['postrun'],python=file_pin(cpu.PYTHON) if python is None else python,
        program=file_pin(cpu.program_for(context['spec']['h1']['mode'])) if program is None else program,
        sources=[file_pin(cpu.ROOT/p) for p in cpu.SOURCES],timeout_seconds=timeout_seconds,repair=repair_pin)
    require(all(request[n]==old['request'][n] for n in ('spec','closure','inventory','postrun')),'JOB_DRIFT')
    require(any(request[n]!=old['request'][n] for n in ('python','program','sources','timeout_seconds')),'UNCHANGED_INPUT')
    name='CPU_ANALYSIS:'+job_key+':REPAIR:'+repair_pin['sha256']
    event=dict(goal_id=state['goal_id'],event_id=name,job_id=job['identity']['job_id'],input_sha256=sha(request),role='cpu-analysis')
    tip=active(journal,job_key)
    require(tip==parent or (tip==sha(event) and state['events'][tip]['request']==request),'NOT_ACTIVE_TIP')
    require(type(timeout_seconds) is int and 1<=timeout_seconds<=3600,'TIMEOUT')
    for n in ('python','program'):checked(request[n])
    spec=context['spec']
    if spec.get('backend')=='real':
        require(request['python']==file_pin(cpu.PYTHON)
            and request['program']==file_pin(cpu.program_for(spec['h1']['mode'])),'EXECUTABLE')
    else:
        require(spec.get('backend')=='runner_cpu' and state['goal_id']=='R6-CPU-TEST-FIXTURE'
            and checked(request['program']).is_relative_to(journal.directory.parent.resolve()),'FIXTURE')
    key,_=journal.intent(event,kind='cpu',request=request,
        resources={'output_actual_plus_unreleased_reservations_bytes':cpu.RESERVATION},
        purpose='repair',free_disk_bytes=free_disk_bytes)
    cpu.validate(journal,key)
    return key
