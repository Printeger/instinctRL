"""Final deterministic evidence/ownership audit and exact completion publication.

No model, GPU query, cancellation, submission or budget refund. A successful
artifact push alone never grants completion; all current-state gates run again.
"""
import json
import shutil
from pathlib import Path
from rlflow.jobs import owns
from rlflow.store import canonical,sha
from tools.r6_recovery import checked,file_pin,RecoveryError
from tools.r6_decision_packet import persist_exact
from tools.r6_worker_processes import cleanup_verified
from tools.r6_job_inventory import record_inventory
from tools import r6_final_delivery as delivery

CAP=128*1024
STATUSES={'R6_TARGET_DOMAIN_PASS','BASELINE_SUFFICIENT','SCIENTIFIC_NEGATIVE_WITH_COMPLETE_EVIDENCE'}


def require(ok,reason):
    if not ok:raise RecoveryError('TERMINAL_AUDIT_'+reason)


def read(pin):
    path=checked(pin);require(path.stat().st_size<=64*1024**2,'RECEIPT_CAP')
    return json.loads(path.read_text())


def dead(record):
    if record is None:return
    require(isinstance(record,dict) and set(record)=={'pid','pgid','starttime','boot_id'}
        and type(record['pid'])is int and record['pid']>0 and type(record['pgid'])is int and record['pgid']>0
        and isinstance(record['starttime'],str) and record['starttime'].isdigit()
        and isinstance(record['boot_id'],str) and bool(record['boot_id']),'PROCESS_IDENTITY')
    require(not owns(record),'OWNED_PROCESS_ALIVE')


def workers(journal):
    """Model legacy exceptions stay exclusively in the existing reviewed guard."""
    delivery.profile.guard(journal);state=journal.read();receipts=[];dormant=[]
    for key,event in sorted(state['events'].items()):
        dead(event.get('process'))
        if event.get('dispatch_claim'):require(bool(event.get('receipt')),'UNRESOLVED_DISPATCH')
        if event['kind']=='job':continue  # Full native ACK/descendants are checked by inventory.
        if event.get('receipt'):
            receipt=read(event['receipt']);dead(receipt.get('process'));dead(receipt.get('child_identity'))
            if event['kind']!='model' and event.get('dispatch_claim'):
                require(receipt.get('event_key')==key and receipt.get('request_sha256')==sha(event['request'])
                    and cleanup_verified(receipt.get('cleanup')),'CPU_CLEANUP')
            receipts.append(dict(event_key=key,receipt=event['receipt']))
        elif event['kind']=='model':dormant.append(key)
        for ordinal,claim in sorted(event.get('sample_claims',{}).items()):
            require(bool(claim.get('receipt')),'UNRESOLVED_RESOURCE_SAMPLE')
            receipt=read(claim['receipt']);require(receipt.get('queue_event')==key and str(receipt.get('ordinal'))==ordinal
                and receipt.get('request_sha256')==claim['request_sha256'],'SAMPLE_RECEIPT')
            dead(receipt.get('process'))
            for row in receipt.get('queries',[]):
                dead(row.get('child_identity'));require(cleanup_verified(row.get('cleanup')),'SAMPLE_CLEANUP')
            receipts.append(dict(event_key=key,ordinal=ordinal,receipt=claim['receipt']))
    return dict(status='NO_KNOWN_OWNED_JOB_OR_WORKER_SURVIVORS',receipt_count=len(receipts),
        receipt_set_sha256=sha(receipts),journal_path=str(journal.path.resolve()),
        receipt_sources='events[*].receipt and events[*].sample_claims[*].receipt, sorted by event key and ordinal',
        unstarted_model_intents_retaining_reservations=dormant,
        model_legacy_reconciliation='EXISTING_REVIEWED_GUARD_ONLY')


def audit(journal,controller_key,store):
    delivery.profile.guard(journal);state=journal.read();controller=state['events'][controller_key]
    require(controller['identity']['role']=='r6-controller' and controller['request']['store_root']==str(store.root.resolve()),'CONTROLLER')
    cursor=controller['cursor'];pin=cursor['final_delivery'];out=delivery.verify(journal,pin,store)
    publication=cursor['final_publication'];entry=journal.read()['events'][out['event_key']]
    require(read(entry['publication'])==publication and publication['event_key']==out['event_key'] and publication['delivery']==pin
        and publication['status']=='FINAL_DELIVERY_PUBLISHED_PENDING_TERMINAL_AUDIT','PUBLICATION_RECEIPT')
    require(out['scientific_status']==publication['scientific_status'] and out['scientific_status'] in STATUSES,'SCIENTIFIC_STATUS')
    git=delivery.git;root=delivery.ROOT;commit=publication['commit'];tag=publication['tag']
    require(git('symbolic-ref','--short','HEAD')=='main' and git('rev-parse',tag)==commit,'PUBLISHED_TAG')
    # Every reviewed artifact must still be byte-for-byte the one in the exact commit.
    files=dict(out['files'],**{'DELIVERY_MANIFEST.json':pin})
    expected=sorted(str(Path(p['path']).relative_to(root)) for p in files.values())
    require(git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines()==expected
        and git('show','-s','--format=%P',commit)==entry['request']['base_commit'],'PUBLISHED_SCOPE')
    for p in files.values():
        require(delivery.snapshot.git(root,'show',commit+':'+str(Path(p['path']).relative_to(root)))==checked(p).read_bytes(),'PUBLISHED_BYTES')
    models=read(out['files']['MODEL_USAGE_AUDIT.json']);budget=read(out['files']['goal_budget.json']);resources=read(out['files']['RESOURCE_AUDIT.json'])
    require(budget==journal.read()['budget'],'FINAL_LEDGER')
    current=delivery.evidence.model_audit(journal.read())
    require({k:v for k,v in models.items() if k not in ('scope','final_review_receipt')}==
        {k:v for k,v in current.items() if k!='scope'},'MODEL_USAGE_DRIFT')
    require(models['all_registered_local_waits_verified']is True and models['platform_inactivity_proven']is True
        and type(models['pure_wait_model_calls'])is int and models['pure_wait_model_calls']==0,'ZERO_MODEL_WAIT_UNPROVEN')
    require(resources['charged']==delivery.evidence.resource_audit(dict(budget=budget))['charged'],'RESOURCE_ACCOUNTING')
    require(shutil.disk_usage(journal.directory).free>=budget['disk_free_floor_bytes'],'FREE_DISK_FLOOR')
    manifests=read(out['files']['TRAINING_AND_EVALUATION_MANIFEST.json']);actual_jobs={k for k,e in state['events'].items() if e['kind']=='job'}
    require(actual_jobs and {j['event_key'] for j in manifests['jobs']}==actual_jobs,'ALL_JOBS')
    inventories=[]
    for key in sorted(actual_jobs):
        job=journal.canonical_job(key);require(job.get('closure') and job.get('inventory') and job.get('postrun'),'JOB_EVIDENCE')
        inventory=record_inventory(journal,key,store)
        require(inventory['output_inventory_complete']is True and not job.get('inventory_violations'),'INCOMPLETE_INVENTORY')
        inventories.append(dict(event_key=key,inventory=job['inventory'],closure=job['closure'],postrun=job['postrun']))
    process_audit=workers(journal)
    decision=read(out['files']['reviews/final-independent.json'])
    require(decision['decision']=='PASS_R6_FINAL_AUDIT' and decision['must_fix']==[],'INDEPENDENT_REVIEW')
    return dict(status='TERMINAL_REQUIREMENTS_VERIFIED_PENDING_COMMIT_AND_CONTROLLER_RECORD',controller_key=controller_key,
        delivery=pin,publication=entry['publication'],scientific_status=out['scientific_status'],
        review=out['review'],review_receipt=out['review_receipt'],inventories=inventories,process_audit=process_audit,
        model_usage=out['files']['MODEL_USAGE_AUDIT.json'],resources=out['files']['RESOURCE_AUDIT.json'],
        requirements=out['files']['REQUIREMENTS_AUDIT.json'],free_disk_floor_bytes=budget['disk_free_floor_bytes'],
        completion_condition='Valid only after this exact audit commit/tag is published and the unique controller records completion.',
        audit_implementation=file_pin(__file__))


def finish(journal,controller_key,store):
    delivery.profile.guard(journal)
    value=audit(journal,controller_key,store);state=journal.read();controller=state['events'][controller_key]
    publication=controller['cursor']['final_publication'];key=publication['event_key'];entry=state['events'][key]
    require(file_pin(__file__) in entry['request']['implementation'],'UNREVIEWED_TERMINAL_IMPLEMENTATION')
    root=delivery.ROOT;folder=root/'docs/experiments/r6-robustness-auto/completions'/key
    path=folder/'COMPLETION_AUDIT.json';require(path.resolve()==path and not path.is_symlink(),'OUTPUT_ALIAS')
    data=canonical(value)+b'\n';require(len(data)<CAP,'OUTPUT_CAP')
    persist_exact(path,data);pin=file_pin(path);relative=str(path.relative_to(root));git=delivery.git
    commits=git('log','-2','--format=%H','HEAD','--',relative).splitlines();require(len(commits)<=1,'COMPLETION_HISTORY')
    if commits:commit=commits[0]
    else:
        require(git('rev-parse','HEAD')==publication['commit'],'COMPLETION_PARENT_HEAD')
        delivery.profile.guard(journal);git('add','-f','--',relative)
        delivery.profile.guard(journal);git('commit','--only','-m','R6 terminal audit '+key,'--',relative)
        commit=git('rev-parse','HEAD')
    require(git('show','-s','--format=%P',commit)==publication['commit']
        and git('show','-s','--format=%s',commit)=='R6 terminal audit '+key
        and git('diff-tree','--no-commit-id','--name-only','-r',commit)==relative
        and delivery.snapshot.git(root,'show',commit+':'+relative)==checked(pin).read_bytes(),'COMPLETION_COMMIT')
    tag='r6-complete-'+key[:16]
    if git('tag','--list',tag):require(git('rev-parse',tag)==commit,'COMPLETION_TAG')
    else:delivery.profile.guard(journal);git('tag',tag,commit)
    require(git('rev-parse','HEAD')==commit,'COMPLETION_HEAD')
    delivery.profile.guard(journal);git('push','origin','main','refs/tags/'+tag)
    refs={row.split()[1]:row.split()[0] for row in git('ls-remote','origin','refs/heads/main','refs/tags/'+tag,'refs/tags/'+publication['tag']).splitlines()}
    require(refs=={'refs/heads/main':commit,'refs/tags/'+tag:commit,'refs/tags/'+publication['tag']:publication['commit']},'COMPLETION_REMOTE')
    # Recheck the owned processes, originals, model calls and ledger after network publication.
    require(audit(journal,controller_key,store)==value,'POST_PUBLICATION_DRIFT')
    result=dict(status=value['scientific_status'],goal_complete=True,completion_audit=pin,completion_commit=commit,
        completion_tag=tag,delivery=value['delivery'],review=value['review'])
    state=journal.read();old=state.get('completion')
    require(old is None or old==result,'COMPLETION_ALREADY_BOUND')
    controller=state['events'][controller_key];controller['cursor'].update(phase='COMPLETE',completion=result)
    controller.update(status='COMPLETE');controller.pop('controller_error',None)
    state.pop('completion_revalidation_error',None)
    state.update(status=result['status'],goal_complete=True,completion=result);journal.save(state)
    return result
