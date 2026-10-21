"""Preserve a rejected repair and register a newly charged source attempt."""
import copy
import json
from rlflow.store import sha,canonical
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact
from tools import r6_profile_preparation as profile,r6_engineering_review as review
from tools import r6_source_preparation as source,r6_source_repair as repair
from tools import r6_repair_input as inputs

CAP=256*1024


def require(ok,reason):
    if not ok:raise RecoveryError('ENGINEERING_REWORK_'+reason)


def prepare(journal,controller_key,store,*,executable,free_disk_bytes):
    profile.guard(journal)
    state=journal.read();owner=state['events'][controller_key];cursor=owner['cursor']
    require(owner['identity']['role']=='r6-controller'
        and owner['request']['store_root']==str(store.root.resolve())
        and cursor['phase']=='ENGINEERING_REPAIR_REJECTED','CONTROLLER')
    require(not any(n in cursor for n in ('confirmation','final_batch','final_models','final_collection')),'FROZEN_PHASE')
    outcome=cursor['engineering_review'];packet=review.rejected(journal,controller_key,outcome)
    bound=review.read(packet['source']);old_key=bound['source_event_key'];old=state['events'][old_key]
    require(owner['engineering_source_result']==packet['source']
        and old['source_result']==packet['source_result'],'SOURCE_BINDING')
    original=inputs.validate(journal,packet['repair_input'],cursor['parent_key'],store)
    restored=original['description']['cursor']
    require(restored['phase']=='REPAIR_AND_REVIEW' and restored['parent_key']==cursor['parent_key']
        and restored['job_key']==cursor['job_key']
        and dict(restored,phase='ENGINEERING_REPAIR_REJECTED',engineering_review=outcome)==cursor,'CURSOR')
    rejection_pin=state['events'][outcome['event_key']]['review_result']
    name='ENGINEERING_REWORK_INPUT:'+outcome['event_key']
    previous=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(previous)<=1,'ALIAS')
    if previous:
        key,entry=previous[0];request=entry['request']
        require(request['controller_key']==controller_key and request['cursor']==cursor
            and request['rejection']==rejection_pin and request['store_sha256']==sha(store.replay()),'INPUT_DRIFT')
    else:
        require(repair.active(journal,cursor['parent_key'])==old_key,'OBSOLETE_REJECTION')
        document=dict(parent_key=cursor['parent_key'],predecessor=old_key,
            predecessor_receipt=old['receipt'],rejected_review=rejection_pin,
            reason='Resolve independent engineering review: '+'; '.join(outcome['must_fix']),
            evidence_pins=[packet['repair_input'],packet['source'],rejection_pin],
            repository_state=repair.reconcile(journal,old_key))
        require(len(canonical(document))<=65536,'DOCUMENT_CAP')
        request=dict(controller_key=controller_key,cursor=copy.deepcopy(cursor),rejection=rejection_pin,
            store_sha256=sha(store.replay()),document=document)
        event=dict(goal_id=state['goal_id'],event_id=name,job_id=old['identity']['job_id'],
            input_sha256=sha(request),role='engineering-rework-input')
        key,_=journal.intent(event,kind='cpu',request=request,
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},
            purpose='repair',free_disk_bytes=free_disk_bytes)
    profile.reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'repair')
    path=journal.directory/'engineering-rework'/key/'repair.json'
    persist_exact(path,canonical(request['document'])+b'\n')
    new_key=source.register(journal,cursor['parent_key'],store,executable=executable,
        free_disk_bytes=free_disk_bytes,repair_pin=file_pin(path),engineering_pin=packet['repair_input'])
    # Registration may have persisted before a crash. Replaying it recovers that
    # exact intent; only this atomic state update restores the original cursor.
    state=journal.read();owner=state['events'][controller_key]
    require(owner['cursor']==cursor,'CURSOR_DRIFT')
    transition=dict(rework_event_key=key,rejected_review=rejection_pin,predecessor=old_key,
        replacement=new_key,from_cursor=copy.deepcopy(cursor),to_cursor=copy.deepcopy(restored))
    owner.setdefault('engineering_rework_history',[]).append(transition)
    owner['cursor']=copy.deepcopy(restored);owner['status']='ENGINEERING_REWORK_REGISTERED'
    state['events'][key]['transition']=transition
    journal.save(state)
    return transition
