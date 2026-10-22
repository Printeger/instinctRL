"""Consume one accepted controller revision from a fresh interpreter process.

No phase reset, worker retry, new permission or model invocation occurs here.
The existing controller retains ownership of every subsequent action.
"""
import json
import os
from pathlib import Path
from rlflow.jobs import identity,owns
from rlflow.store import sha,canonical
from tools.r6_recovery import Journal,RecoveryError,file_pin,checked
from tools.r6_decision_packet import persist_exact
from tools import r6_profile_preparation as profile,r6_controller_revision as revision

CAP=256*1024


def require(ok,reason):
    if not ok:raise RecoveryError('CONTROLLER_RESUME_'+reason)


def run(journal,controller_key,revision_key,store,*,free_disk_bytes):
    from tools import r6_controller as controller
    profile.guard(journal);state=journal.read();owner=state['events'][controller_key]
    packet=revision.review_record(journal,revision_key)
    require(owner['identity']['role']==controller.ROLE and sha(owner['identity'])==controller_key
        and owner['identity']['input_sha256']==sha(owner['request'])
        and owner['request']['store_root']==str(store.root.resolve())
        and packet['controller_key']==controller_key
        and owner.get('implementation_revisions',[])[-1:]==[revision_key],'CONTROLLER_REVISION')
    require(revision.effective(journal,controller_key)==controller.implementation()==packet['after'],'IMPLEMENTATION')
    name='CONTROLLER_RESUME:'+revision_key
    prior=[(k,e) for k,e in state['events'].items() if e['identity'].get('event_id')==name]
    require(len(prior)<=1,'ALIAS')
    if prior:
        key,entry=prior[0];request=entry['request']
        require(request['controller_key']==controller_key and request['revision_key']==revision_key
            and request['inputs']==owner['request']['inputs'],'INPUT_DRIFT')
        profile.reservation(state,key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':CAP},'repair')
        if entry.get('resume_result'):
            value=json.loads(checked(entry['resume_result']).read_text())
            require(value['event_key']==key and value['request_sha256']==sha(request)
                and value['process']==entry['resume_claim']['process'],'RESULT_BINDING')
            return value
        require(not entry.get('resume_claim'),'CLAIMED_NO_BLIND_RETRY')
    require(sha(owner['cursor'])==packet['cursor_sha256'] and sha(store.replay())==packet['store_sha256'],'STATE_DRIFT')
    proposal=revision.read(packet['proposal']);target=proposal['to_commit']
    require(revision.committed_implementation(target)==packet['after']
        and revision.snapshot.git(revision.ROOT,'rev-parse','HEAD').decode().strip()==target,'COMMIT')
    old_process=owner.get('controller_process');current=identity(os.getpid())
    require(isinstance(old_process,dict) and set(old_process)=={'pid','pgid','boot_id','starttime'}
        and type(old_process['pid']) is int and old_process['pid']>0
        and type(old_process['pgid']) is int and old_process['pgid']>0
        and isinstance(old_process['boot_id'],str) and bool(old_process['boot_id'])
        and isinstance(old_process['starttime'],str) and old_process['starttime'].isdigit(),'OLD_PROCESS_IDENTITY')
    require(old_process!=current and not owns(old_process),'FRESH_PROCESS_REQUIRED')
    if not prior:
        request=dict(controller_key=controller_key,revision_key=revision_key,inputs=owner['request']['inputs'],
            cursor=owner['cursor'],store_root=str(store.root.resolve()),store_sha256=sha(store.replay()),
            old_process=old_process,implementation=file_pin(__file__),controller_implementation=packet['after'])
        event=dict(goal_id=state['goal_id'],event_id=name,job_id='R6_CONTROLLER',
            input_sha256=sha(request),role='controller-resume')
        key,_=journal.intent(event,kind='cpu',request=request,
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP},purpose='repair',free_disk_bytes=free_disk_bytes)
    require(request['old_process']==old_process and request['cursor']==owner['cursor'],'REQUEST_DRIFT')
    checked(request['implementation']);checked(request['inputs'])
    profile.guard(journal)
    state=journal.read();state['events'][key]['resume_claim']=dict(process=current)
    journal.save(state)
    try:
        result=controller.run(journal,request['inputs'],store,free_disk_bytes=free_disk_bytes)
        value=dict(status='CONTROLLER_RESUME_RETURNED',event_key=key,request_sha256=sha(request),
            process=current,controller_result=result)
        data=canonical(value)+b'\n';require(len(data)+len(canonical(request))<CAP,'RESULT_CAP')
        path=journal.directory/'controller-resumes'/key/'result.json';persist_exact(path,data)
        state=journal.read();state['events'][key]['resume_result']=file_pin(path)
        state['events'][key]['status']='CONTROLLER_RESUME_RETURNED';journal.save(state)
        return value
    except BaseException as error:
        state=journal.read();state['events'][key].update(status='CONTROLLER_RESUME_FAILED',
            resume_error=dict(type=type(error).__name__,message=str(error)[:4096]))
        journal.save(state)
        raise


def main():
    import argparse,shutil
    from rlflow.store import Store
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('controller_key');parser.add_argument('revision_key')
    parser.add_argument('--store',type=Path,required=True)
    args=parser.parse_args();root=args.store.resolve(strict=True)
    require((root/'runtime/research.db').is_file(),'STORE_REQUIRED')
    journal=Journal(args.directory.resolve())
    with journal.writer():
        result=run(journal,args.controller_key,args.revision_key,Store(root),free_disk_bytes=shutil.disk_usage(journal.directory).free)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
