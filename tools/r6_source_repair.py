"""Reconcile source-writer side effects before explicit replacement; no dispatch."""
import hashlib
import json
import re
from pathlib import Path
from rlflow.jobs import owns
from rlflow.store import sha
from tools.r6_recovery import checked,RecoveryError
from tools.r6_worker_processes import cleanup_verified


def require(ok,reason):
    if not ok:raise RecoveryError('SOURCE_REPAIR_'+reason)


def changes(repo,base,commit,output):
    from tools import r6_source_preparation as source
    git=source.snapshot.git
    git(repo,'merge-base','--is-ancestor',base,commit)
    names=git(repo,'diff','--name-only','-z',base,commit).decode().split('\0')[:-1]
    rows=[];total=0
    for name in names:
        require(name.startswith(('isaaclab-training/','training_workloads/','tests/',output+'/'))
            and name!='training_workloads/unified_h1/config.py','CHANGE_SCOPE')
        tree=git(repo,'ls-tree',commit,'--',name).decode().split()
        require(bool(tree) and tree[0] in ('100644','100755'),'FILE_MODE')
        data=git(repo,'show',commit+':'+name);total+=len(data)
        rows.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest()))
        if name.startswith(output+'/'):
            require(not git(repo,'ls-tree',base,'--',name),'OLD_ARTIFACT_CHANGED')
    require(total<=source.ARTIFACT_CAP,'ARTIFACT_CAP')
    return rows


def reconcile(journal,key):
    """Read-only candidate evidence; caller persists it in an explicit repair document."""
    from tools import r6_source_preparation as source
    entry=journal.read()['events'][key];packet=entry['source_input'];repo=source.repository(journal)
    require(str(repo)==packet['repository'],'REPOSITORY')
    commit=source.snapshot.git(repo,'rev-parse','HEAD').decode().strip()
    output=Path(packet['output_directory']).relative_to(repo).as_posix()
    return dict(commit=commit,changes=changes(repo,packet['base_commit'],commit,output))


def predecessor(journal,pin,parent_key):
    from tools import r6_source_preparation as source
    from tools.r6_profile_preparation import reservation
    from tools.r6_batch import verified_decision
    path=checked(pin);require(path.stat().st_size<=65536,'INPUT_CAP');data=json.loads(path.read_text())
    state=journal.read()
    require(set(data)=={'parent_key','predecessor','predecessor_receipt','reason','evidence_pins','repository_state'}
        and data['parent_key']==parent_key and isinstance(data['reason'],str) and data['reason'].strip()
        and isinstance(data['evidence_pins'],list) and data['evidence_pins'],'FIELDS')
    for evidence in data['evidence_pins']:checked(evidence)
    key=data['predecessor'];old=state['events'][key];packet=old['source_input']
    require(old['identity']['role']==source.ROLE and packet['parent_key']==parent_key
        and old['identity']['job_id']==state['events'][parent_key]['identity']['job_id']
        and packet['parent']==state['events'][parent_key]['decision_record']
        and old.get('source_result') is None and isinstance(old.get('receipt'),dict)
        and old['receipt']==data['predecessor_receipt'] and old['status']=='RECEIPT_RECORDED','PREDECESSOR')
    require(source.pin_bytes(Path(old['request']['prompt']['path']),packet['prompt'].encode())==old['request']['prompt'],'PROMPT_BINDING')
    resources=dict(automatic_model_sessions=1,
        output_actual_plus_unreleased_reservations_bytes=source.MODEL_CAP+source.ARTIFACT_CAP+65536)
    if 'major_scientific_candidates' in old['resources']:resources['major_scientific_candidates']=1
    require(old['resources']==resources,'RESOURCES')
    reservation(state,key,'model',old['resources'],old['purpose'])
    receipt=json.loads(checked(old['receipt']).read_text())
    require(receipt['event_key']==key and receipt['request_sha256']==sha(old['request'])
        and receipt.get('process')==old.get('process')
        and type(receipt.get('exit_code')) is int and type(receipt.get('schema_valid')) is bool
        and 'error' in receipt and cleanup_verified(receipt.get('cleanup')),'RECEIPT_CLEANUP')
    for ident in (receipt.get('process'),receipt.get('child_identity')):
        require(ident is None or (isinstance(ident,dict) and set(ident)=={'pid','pgid','boot_id','starttime'}
            and type(ident['pid']) is int and ident['pid']>0 and type(ident['pgid']) is int and ident['pgid']>0
            and isinstance(ident['boot_id'],str) and bool(ident['boot_id'])
            and isinstance(ident['starttime'],str) and ident['starttime'].isdigit()),'PROCESS_IDENTITY')
        require(ident is None or not owns(ident),'PROCESS_ALIVE')
    for field in ('events','stderr','last_message'):
        if receipt.get(field):checked(receipt[field])
    if receipt['exit_code']==0 and receipt['error'] is None and receipt['schema_valid']:
        result=verified_decision(journal,key)
        require(result['decision']=='PREPARATION_INCOMPLETE' and result['must_fix']
            and result['parent_decision_sha256']==packet['parent']['sha256'],'ACCEPTED_PREDECESSOR')
    repo=source.repository(journal);git=source.snapshot.git
    require(packet['repository']==str(repo),'REPOSITORY')
    evidence=data['repository_state'];require(set(evidence)=={'commit','changes'}
        and isinstance(evidence['commit'],str) and re.fullmatch('[0-9a-f]{40}',evidence['commit']),'REPOSITORY_STATE')
    output=Path(packet['output_directory']).relative_to(repo).as_posix()
    require(changes(repo,packet['base_commit'],evidence['commit'],output)==evidence['changes'],'CHANGES_DRIFT')
    git(repo,'merge-base','--is-ancestor',evidence['commit'],'HEAD')
    # Old attempt artifacts are immutable even when subsequent source code changes.
    for row in evidence['changes']:
        if row['path'].startswith(output+'/'):
            require(file_hash(repo/row['path'])==row['sha256'],'OLD_OUTPUT_DRIFT')
    return key,data


def file_hash(path):
    require(path.is_file() and not path.is_symlink() and path.resolve()==path,'OUTPUT_FILE')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_start(journal,pin,parent_key):
    from tools import r6_source_preparation as source
    key,data=predecessor(journal,pin,parent_key);packet=journal.read()['events'][key]['source_input']
    repo=source.repository(journal);git=source.snapshot.git
    require(source.control_identity()==packet['control_identity'],'CONTROL_DRIFT')
    require(sha(json.loads(checked(packet['parent']).read_text()))==sha(json.loads(checked(journal.read()['events'][parent_key]['decision_record']).read_text())),'PARENT_DRIFT')
    require(git(repo,'branch','--show-current').decode().strip()=='main','BRANCH')
    output=Path(packet['output_directory']).relative_to(repo).as_posix()
    require(not git(repo,'status','--porcelain=v1','-z','--untracked-files=all','--',
        'isaaclab-training/','training_workloads/','tests/',output),'UNCOMMITTED_SOURCE_EFFECTS')
    extra=git(repo,'diff','--name-only','-z',data['repository_state']['commit'],'HEAD').decode().split('\0')[:-1]
    repair_path=checked(pin)
    allowed={repair_path.relative_to(repo).as_posix()} if repair_path.is_relative_to(repo) else set()
    require(set(extra)<=allowed,'RECONCILIATION_HEAD_DRIFT')
    return key


def active(journal,parent_key):
    from tools import r6_source_preparation as source
    prefix='SOURCE_PREPARATION:'+parent_key
    rows={k:e for k,e in journal.read()['events'].items() if e['identity']['role']==source.ROLE
        and (e['identity']['event_id']==prefix or e['identity']['event_id'].startswith(prefix+':REPAIR:'))}
    if not rows:return None
    roots=[k for k,e in rows.items() if e['identity']['event_id']==prefix];require(len(roots)==1,'ROOT')
    children={}
    for key,entry in rows.items():
        require(sha(entry['identity'])==key and entry['identity']['input_sha256']==sha(entry['request']),'IDENTITY')
        if entry['identity']['event_id']==prefix:continue
        pin=entry['source_input']['repair'];parent,_=predecessor(journal,pin,parent_key)
        require(entry['identity']['event_id']==prefix+':REPAIR:'+pin['sha256'],'EVENT')
        require(parent in rows and parent not in children,'FORK');children[parent]=key
    key=roots[0];seen={key}
    while key in children:
        key=children[key];require(key not in seen,'CYCLE');seen.add(key)
    require(seen==set(rows),'ORPHAN')
    return key
