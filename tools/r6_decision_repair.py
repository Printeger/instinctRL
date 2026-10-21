"""Explicit replacement of failed post-analysis model calls, never scientific decisions."""
import json
from rlflow.jobs import owns
from rlflow.store import sha
from tools.r6_recovery import checked,RecoveryError
from tools.r6_worker_processes import cleanup_verified


def require(ok,reason):
    if not ok:raise RecoveryError('DECISION_REPAIR_'+reason)


def predecessor(journal,pin,job_key):
    from tools.r6_decision import MODEL_CAP
    from tools.r6_profile_preparation import reservation
    path=checked(pin);require(path.stat().st_size<=65536,'INPUT_CAP')
    data=json.loads(path.read_text());state=journal.read()
    require(set(data)=={'job_event_key','predecessor','predecessor_receipt','reason','evidence_pins'}
        and data['job_event_key']==job_key and isinstance(data['reason'],str) and data['reason'].strip()
        and isinstance(data['evidence_pins'],list) and data['evidence_pins'],'FIELDS')
    for evidence in data['evidence_pins']:checked(evidence)
    key=data['predecessor'];old=state['events'][key];job=journal.canonical_job(job_key,state=state)
    require(old['identity']['role']=='post-analysis-decision' and old['identity']['job_id']==job['identity']['job_id']
        and old.get('decision_input',{}).get('job_event_key')==job_key
        and old.get('decision_record') is None and isinstance(old.get('receipt'),dict)
        and old['receipt']==data['predecessor_receipt'] and old['status']=='RECEIPT_RECORDED','PREDECESSOR')
    reservation(state,key,'model',{'automatic_model_sessions':1,
        'output_actual_plus_unreleased_reservations_bytes':MODEL_CAP+65536},old['purpose'])
    receipt=json.loads(checked(old['receipt']).read_text())
    require(receipt['event_key']==key and receipt['request_sha256']==sha(old['request'])
        and receipt.get('process')==old.get('process')
        and type(receipt.get('exit_code')) is int and type(receipt.get('schema_valid')) is bool
        and 'error' in receipt and (receipt['exit_code']!=0 or receipt['error'] is not None
                                   or receipt['schema_valid'] is False),'FAILED_RECEIPT')
    require(cleanup_verified(receipt.get('cleanup')),'CLEANUP')
    for ident in (receipt.get('process'),receipt.get('child_identity')):
        require(ident is None or (isinstance(ident,dict) and set(ident)=={'pid','pgid','boot_id','starttime'}
            and type(ident['pid']) is int and ident['pid']>0 and type(ident['pgid']) is int and ident['pgid']>0
            and isinstance(ident['boot_id'],str) and bool(ident['boot_id'])
            and isinstance(ident['starttime'],str) and ident['starttime'].isdigit()),'PROCESS_IDENTITY')
        require(ident is None or not owns(ident),'PROCESS_ALIVE')
    for field in ('events','stderr','last_message'):
        if receipt.get(field):checked(receipt[field])
    return key


def for_entry(journal,entry,job_key):
    from tools.r6_decision_packet import RESERVATION
    from tools.r6_profile_preparation import reservation
    prefix='DECISION:'+job_key;name=entry['identity']['event_id']
    if name==prefix:return None
    require(name.startswith(prefix+':REPAIR:'),'EVENT')
    digest=name[len(prefix+':REPAIR:'):]
    packet_name='DECISION_PACKET:'+job_key+':REPAIR:'+digest
    rows=[(k,e) for k,e in journal.read()['events'].items() if e['identity'].get('event_id')==packet_name]
    require(len(rows)==1,'PACKET')
    key,packet=rows[0]
    require(packet['identity']['role']=='decision-packet','PACKET_ROLE')
    reservation(journal.read(),key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':RESERVATION},'repair')
    pin=packet['request']['packet']['repair']
    require(pin['sha256']==digest,'PIN');predecessor(journal,pin,job_key)
    return pin


def active(journal,job_key):
    state=journal.read();job=journal.canonical_job(job_key,state=state)
    rows={k:e for k,e in state['events'].items() if e['kind']=='model'
        and e['identity']['role']=='post-analysis-decision' and e['identity']['job_id']==job['identity']['job_id']}
    if not rows:return None
    roots=[k for k,e in rows.items() if e['identity']['event_id']=='DECISION:'+job_key]
    require(len(roots)==1,'ROOT');children={}
    for key,entry in rows.items():
        require(sha(entry['identity'])==key and entry['identity']['input_sha256']==sha(entry['request']),'IDENTITY')
        pin=for_entry(journal,entry,job_key)
        if pin:
            parent=predecessor(journal,pin,job_key)
            require(parent in rows and parent not in children,'FORK');children[parent]=key
    key=roots[0];seen={key}
    while key in children:
        key=children[key];require(key not in seen,'CYCLE');seen.add(key)
    require(seen==set(rows),'ORPHAN')
    return key
