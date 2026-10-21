"""Independent review of a pinned workload repair; acceptance is not live permission."""
import json
from pathlib import Path
from rlflow.store import sha,canonical
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact,pin_bytes
from tools.r6_decision import CLI,CLI_SHA,MODEL_CAP,model_stop_paths
from tools.r6_batch import model_handoff,verified_decision
from tools.r6_budget import totals
from tools import r6_profile_preparation as profile,r6_source_preparation as source,r6_repair_input as inputs
from tools.r6_prepare_authorization import verify_frozen

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'docs/experiments/r6-robustness-auto/prompts/engineering-repair-review.schema.json'
CAP=256*1024
ROLE='engineering-repair-review'


def require(ok,reason):
    if not ok:raise RecoveryError('ENGINEERING_REVIEW_'+reason)


def read(pin):return json.loads(checked(pin).read_text())


def snapshot_pin(journal,outcome):
    snap=outcome['snapshot'];entry=journal.read()['events'][snap['event_key']]
    require(entry['identity']['role']=='source-snapshot' and read(entry['receipt'])==snap
        and snap['status']=='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY','SNAPSHOT_RECEIPT')
    profile.reservation(journal.read(),snap['event_key'],'cpu',
        {'output_actual_plus_unreleased_reservations_bytes':source.snapshot.RESERVATION},entry['purpose'])
    require(snap['input_sha256']==sha(entry['request']),'SNAPSHOT_REQUEST')
    config=read(snap['config']);verify_frozen(snap,config)
    artifact=read(outcome['source_result'])
    require(snap['source_commit']==artifact['source_commit'] and snap['config']==artifact['config'],'SNAPSHOT_SOURCE')
    return entry['receipt']


def evidence(journal,controller_key,bound_pin,store):
    state=journal.read();controller=state['events'][controller_key];outcome=read(bound_pin)
    require(controller['identity']['role']=='r6-controller'
        and controller['request']['store_root']==str(store.root.resolve())
        and controller.get('engineering_source_result')==bound_pin
        and outcome['controller_key']==controller_key
        and outcome['status']=='ENGINEERING_SOURCE_READY_REQUIRES_INDEPENDENT_REVIEW','SOURCE_BINDING')
    input=inputs.validate(journal,outcome['repair_input'],controller['cursor']['parent_key'],store)
    desc=input['description'];cursor=controller['cursor']
    require(desc['controller_key']==controller_key and desc['parent_key']==cursor['parent_key']
        and desc['job_key']==cursor['job_key'],'PARENT_JOB')
    require(not any(k in cursor for k in ('confirmation','final_batch','final_models','final_collection')),'FROZEN_PHASE')
    artifact=source.record(journal,outcome['source_event_key'],store)
    require(read(outcome['source_result'])==artifact and artifact['engineering_repair']==outcome['repair_input'],'SOURCE_RESULT')
    return dict(source=bound_pin,repair_input=outcome['repair_input'],source_result=outcome['source_result'],
        snapshot=snapshot_pin(journal,outcome),card=artifact['card'],config=artifact['config'],
        parent=artifact['parent'],source_commit=artifact['source_commit'])


def prepare(journal,controller_key,store,*,executable,free_disk_bytes):
    journal.require_writer();state=journal.read();controller=state['events'][controller_key]
    bound=controller['engineering_source_result'];checked(bound)
    name='ENGINEERING_REVIEW_INPUT:'+controller_key+':'+bound['sha256']
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'INPUT_ALIAS')
    if prior:
        cpu_key,cpu=prior[0];packet=cpu['request']['packet']
        require(packet['source']==bound and packet['executable']==executable,'INPUT_DRIFT')
    else:
        profile.guard(journal);exe=checked(executable)
        if state['goal_id']=='R6-CPU-TEST-FIXTURE':require(exe.is_relative_to(journal.directory.parent.resolve()),'FIXTURE_EXE')
        else:
            require(executable==dict(path=str(CLI),sha256=CLI_SHA),'CLI')
            from tools.r6_review_access import check
            check(executable,bound)
        packet=dict(evidence(journal,controller_key,bound,store),controller_key=controller_key,
            store_sha256=sha(store.replay()),budget=totals(journal.read()['budget']),
            model_cap=state['budget']['caps']['automatic_model_sessions'],executable=executable,
            implementation=file_pin(__file__))
        request=dict(packet=packet)
        event=dict(goal_id=state['goal_id'],event_id=name,job_id=state['events'][controller['cursor']['parent_key']]['identity']['job_id'],input_sha256=sha(request),role='engineering-review-input')
        cpu_key,_=journal.intent(event,kind='cpu',request=request,
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='repair',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),cpu_key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'repair')
    require(journal.read()['events'][cpu_key]['identity']['role']=='engineering-review-input'
        and packet['controller_key']==controller_key,'CPU_IDENTITY')
    checked(packet['implementation'])
    prompt=('Independently review this R6 workload engineering repair, read only, then exit. '
        'Read original failure/receipts, every finding, committed diff, design card, tests and exact frozen snapshot. '
        'Verify that all findings are actually addressed, not merely copied into the card. Reject scientific '
        'method changes disguised as repair, actor privilege, metric/split relaxation, holdout tuning or lost '
        'failure evidence. Check original u640 and fresh attempt identity. A PASS only allows the next '
        'compatibility/training/candidate and exact prelive review stages; it is not a live permission. '
        'Do not edit, query GPU, simulate, train, spawn agents or mutate Journal/Store. '
        'Return PASS_R6_ENGINEERING_REPAIR only with all checks true, must_fix=[], changed_files=[], '
        'reviewer=r6-independent-engineering-repair; otherwise MUST_FIX with concrete findings. '
        'packet_sha256 is the canonical JSON hash. Cite source,repair_input,source_result,snapshot,card,config,parent '
        'pins in evidence_pins. automatic_sessions_remaining=model_cap-budget.automatic_model_sessions-1.\n'
        +json.dumps(packet,sort_keys=True)+'\n').encode()
    require(len(prompt)<=65536,'PROMPT_CAP')
    folder=journal.directory/'engineering-reviews'/cpu_key
    request=dict(prompt=pin_bytes(folder/'prompt.txt',prompt),schema=file_pin(SCHEMA),executable=executable,
        cwd=str(ROOT),sandbox='read-only',timeout_seconds=3600,output_cap_bytes=MODEL_CAP,stop_paths=model_stop_paths(journal))
    event=dict(goal_id=state['goal_id'],event_id='ENGINEERING_REVIEW:'+cpu_key,
        job_id=journal.read()['events'][cpu_key]['identity']['job_id'],input_sha256=sha(request),role=ROLE)
    key=sha(event)
    models=[(k,e) for k,e in journal.read()['events'].items() if e['identity'].get('event_id')==event['event_id']]
    require(len(models)<=1 and all(k==key for k,e in models),'MODEL_ALIAS')
    old=None if not models else models[0][1]
    if old is None:
        profile.guard(journal)
        key,_=journal.intent(event,kind='model',request=request,resources={'automatic_model_sessions':1,
            'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},purpose='repair',free_disk_bytes=free_disk_bytes)
    else:require(old['request']==request and old['identity']==event,'MODEL_DRIFT')
    profile.reservation(journal.read(),key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},'repair')
    entry=journal.read()['events'][key]
    if not entry.get('dispatch_claim') and not entry.get('receipt'):
        profile.guard(journal);current=evidence(journal,controller_key,bound,store)
        require(all(packet[k]==v for k,v in current.items()),'EVIDENCE_DRIFT')
        expected=dict(packet['budget']);expected['automatic_model_sessions']+=1
        expected['output_actual_plus_unreleased_reservations_bytes']+=CAP+MODEL_CAP+65536
        require(totals(journal.read()['budget'])==expected,'STALE_BUDGET')
        persist_exact(folder/'prompt.txt',prompt)
    model_handoff(journal,key)
    profile.guard(journal);current=evidence(journal,controller_key,bound,store)
    require(all(packet[k]==v for k,v in current.items()) and sha(store.replay())==packet['store_sha256'],'EVIDENCE_DRIFT')
    result=verified_decision(journal,key)
    require(result['packet_sha256']==sha(packet) and result['changed_files']==[]
        and result['reviewer']=='r6-independent-engineering-repair','REVIEW_BINDING')
    require(result['automatic_sessions_remaining']==packet['model_cap']-packet['budget']['automatic_model_sessions']-1,'BUDGET')
    require(all(packet[n] in result['evidence_pins'] for n in ('source','repair_input','source_result','snapshot','card','config','parent')),'REVIEW_EVIDENCE')
    for pin in result['evidence_pins']:checked(pin)
    accepted=result['decision']=='PASS_R6_ENGINEERING_REPAIR'
    require((accepted and not result['must_fix'] and all(result['checks'].values()))
        or (not accepted and result['decision']=='MUST_FIX' and bool(result['must_fix'])),'REVIEW_CHECKS')
    entry=journal.read()['events'][key];receipt=read(entry['receipt'])
    require(receipt['event_key']==key and receipt['request_sha256']==sha(request)
        and receipt.get('process')==entry.get('process'),'RECEIPT_BINDING')
    output=dict(status='ENGINEERING_REPAIR_REVIEW_ACCEPTED' if accepted else 'ENGINEERING_REPAIR_REVIEW_REJECTED',
        event_key=key,source=bound,source_result=packet['source_result'],snapshot=packet['snapshot'],
        review=receipt['last_message'],must_fix=result['must_fix'],dispatch_authorized=False,goal_complete=False)
    persist_exact(folder/'result.json',canonical(output)+b'\n')
    state=journal.read();state['events'][key]['review_result']=file_pin(folder/'result.json');journal.save(state)
    return output


def accepted(journal,controller_key,outcome):
    """Revalidate the original accepted review on a later controller wake."""
    state=journal.read();key=outcome['event_key'];entry=state['events'][key]
    require(entry['identity']['role']==ROLE and sha(entry['identity'])==key
        and entry['identity']['input_sha256']==sha(entry['request'])
        and read(entry['review_result'])==outcome,'ACCEPTANCE_RECORD')
    profile.reservation(state,key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},'repair')
    name=entry['identity']['event_id'];require(name.startswith('ENGINEERING_REVIEW:'),'ACCEPTANCE_EVENT')
    cpu_key=name.removeprefix('ENGINEERING_REVIEW:');cpu=state['events'][cpu_key];packet=cpu['request']['packet']
    profile.reservation(state,cpu_key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'repair')
    require(packet['controller_key']==controller_key and entry['request']['schema']==file_pin(SCHEMA)
        and entry['request']['sandbox']=='read-only'
        and json.loads(checked(entry['request']['prompt']).read_text().splitlines()[-1])==packet,'ACCEPTANCE_INPUT')
    result=verified_decision(journal,key);receipt=read(entry['receipt'])
    require(outcome['status']=='ENGINEERING_REPAIR_REVIEW_ACCEPTED' and result['decision']=='PASS_R6_ENGINEERING_REPAIR'
        and result['packet_sha256']==sha(packet) and not result['must_fix'] and not result['changed_files']
        and all(result['checks'].values()) and result['reviewer']=='r6-independent-engineering-repair'
        and receipt['event_key']==key and receipt['request_sha256']==sha(entry['request'])
        and receipt['last_message']==outcome['review'],'ACCEPTANCE_REVIEW')
    for field in ('source','source_result','snapshot'):
        require(outcome[field]==packet[field],'ACCEPTANCE_PIN');checked(outcome[field])
    for pin in result['evidence_pins']:checked(pin)
    require(all(packet[n] in result['evidence_pins'] for n in ('source','repair_input','source_result','snapshot','card','config','parent')),'ACCEPTANCE_EVIDENCE')
    return packet
