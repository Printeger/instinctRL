"""One bounded workload-source repair via the existing writer and snapshot path.

Controller implementation changes and offline-only repair are not dispatched
through this writer. A fresh independent repair review is still required.
"""
import json
from rlflow.store import sha,canonical
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools import r6_repair_input as inputs,r6_source_preparation as source
from tools.r6_source_repair import active
from tools.r6_decision_packet import persist_exact


def require(ok,reason):
    if not ok:raise RecoveryError('ENGINEERING_SOURCE_'+reason)


def prepare(journal,controller_key,store,*,executable,free_disk_bytes):
    journal.require_writer();state=journal.read();controller=state['events'][controller_key];cursor=controller['cursor']
    require(controller['identity']['role']=='r6-controller'
        and controller['request']['store_root']==str(store.root.resolve())
        and cursor['phase']=='REPAIR_AND_REVIEW','CONTROLLER')
    require(not any(name in cursor for name in ('confirmation','final_batch','final_models','final_collection')),'FROZEN_PHASE_REPAIR_REQUIRES_SEPARATE_REVIEW')
    cursor_hash=sha(cursor)
    attempts=controller.get('engineering_source_attempts',{})
    binding=attempts.get(cursor_hash,{}).get('input')
    previous=controller.get('engineering_source_input')
    if binding is None and previous is not None:
        if previous['parent_key']==cursor['parent_key'] or previous['job_key']==cursor['job_key']:
            require(previous['cursor_sha256']==cursor_hash,'CURSOR_DRIFT');binding=previous
    if binding is None:
        result=inputs.prepare(journal,controller_key,store,free_disk_bytes=free_disk_bytes)
        binding=dict(cursor_sha256=sha(cursor),repair_input=result['repair_input'],
            parent_key=cursor['parent_key'],job_key=cursor['job_key'])
        state=journal.read();owner=state['events'][controller_key];owner['engineering_source_input']=binding
        owner.setdefault('engineering_source_attempts',{})[cursor_hash]={'input':binding};journal.save(state)
    require(binding['cursor_sha256']==sha(cursor) and binding['parent_key']==cursor['parent_key']
        and binding['job_key']==cursor['job_key'],'CURSOR_DRIFT')
    value=json.loads(checked(binding['repair_input']).read_text())
    require(value['description']['controller_key']==controller_key
        and value['description']['cursor']==cursor,'INPUT_BINDING')
    # Reconnect a claimed source worker before new-action/STOP validation. The
    # existing handoff owns cancellation, cleanup, receipt and replay semantics.
    outcome=source.prepare_source(journal,cursor['parent_key'],store,
        executable=executable,free_disk_bytes=free_disk_bytes,engineering_pin=binding['repair_input'])
    require(outcome['preparation'].get('engineering_repair')==binding['repair_input'],'RESULT_BINDING')
    key=active(journal,cursor['parent_key']);entry=journal.read()['events'][key]
    require(json.loads(checked(entry['source_result']).read_text())==outcome['preparation'],'RESULT_PIN')
    result=dict(status='ENGINEERING_SOURCE_READY_REQUIRES_INDEPENDENT_REVIEW',
        controller_key=controller_key,repair_input=binding['repair_input'],source_event_key=key,
        source_result=entry['source_result'],snapshot=outcome['snapshot'],
        dispatch_authorized=False,scientific_acceptance=False,goal_complete=False)
    path=journal.directory/'engineering-repairs'/value['event_key']/'source-result.json'
    data=canonical(result)+b'\n';require(len(data)<65536,'OUTPUT_CAP');persist_exact(path,data)
    state=journal.read();owner=state['events'][controller_key];owner['engineering_source_result']=file_pin(path)
    owner.setdefault('engineering_source_attempts',{}).setdefault(cursor_hash,{'input':binding})['result']=file_pin(path)
    journal.save(state)
    return result
