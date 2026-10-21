"""Local review -> exact issuance -> next-job runner composition.

Consumes a concrete prepared plan. Does not invent scientific plans, create
snapshots or grant initial-baseline authority. Existing phases own recovery.
"""
import argparse
import json
from pathlib import Path
import shutil
from tools.r6_recovery import Journal,checked,RecoveryError
from tools import r6_next_job as next_job
from tools.r6_issue_next import issue
from tools.r6_batch import model_handoff


def continue_job(journal,parent_key,plan_pin,master_pin,review_executable,store,*,free_disk_bytes):
    journal.require_writer()
    state=journal.read()
    # An edge already registered by advance owns all job recovery, including
    # STOP cleanup. Do not reissue authority or call a model for that edge.
    edges=[e for e in state['events'].values() if e['identity']['role']==next_job.ROLE
        and e['request'].get('parent_key')==parent_key]
    if len(edges)>1:raise RecoveryError('CONTINUE_PARENT_ALIAS')
    if edges:
        request=edges[0]['request']
        if request['plan']!=plan_pin:raise RecoveryError('CONTINUE_PLAN_DRIFT')
        return next_job.advance(journal,parent_key,plan_pin,request['review_key'],request['review'],
            store,free_disk_bytes=free_disk_bytes)
    # A crash after registering the model but before its exit must reconnect
    # through model_handoff; register_review's idle guard would reject it.
    event_id='NEXT_JOB_REVIEW:'+parent_key+':'+plan_pin['sha256']
    reviews=[(k,e) for k,e in state['events'].items()
        if e['identity']['role']=='next-job-review' and e['identity']['event_id']==event_id]
    if len(reviews)>1:raise RecoveryError('CONTINUE_REVIEW_ALIAS')
    if reviews:
        key,review=reviews[0]
        if review['request']['executable']!=review_executable:
            raise RecoveryError('CONTINUE_REVIEW_EXECUTABLE_DRIFT')
        if not review.get('dispatch_claim') and not review.get('receipt'):
            # Registration commits intent before prompt bytes. Replay repairs
            # that pre-dispatch crash window, preserving request and charge.
            recovered=next_job.register_review(journal,parent_key,plan_pin,store,
                executable=review_executable,free_disk_bytes=free_disk_bytes)
            if recovered!=key:raise RecoveryError('CONTINUE_REVIEW_RECOVERY_DRIFT')
    else:
        key=next_job.register_review(journal,parent_key,plan_pin,store,
            executable=review_executable,free_disk_bytes=free_disk_bytes)
    model_handoff(journal,key)
    receipt=json.loads(checked(journal.read()['events'][key]['receipt']).read_text())
    review_pin=receipt['last_message']
    issue(journal,parent_key,plan_pin,key,review_pin,master_pin,store,free_disk_bytes=free_disk_bytes)
    return next_job.advance(journal,parent_key,plan_pin,key,review_pin,store,free_disk_bytes=free_disk_bytes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('request',type=Path)
    parser.add_argument('--request-sha256',required=True);parser.add_argument('--store',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(checked(dict(path=str(args.request.resolve()),sha256=args.request_sha256)).read_text())
    if set(request)!={'parent_key','plan_pin','master_pin','review_executable'}:
        raise RecoveryError('CONTINUE_REQUEST_FIELDS')
    from rlflow.store import Store
    root=args.store.resolve(strict=True)
    if not (root/'runtime/research.db').is_file():raise RecoveryError('CONTINUE_STORE_REQUIRED')
    journal=Journal(args.directory.resolve())
    with journal.writer():
        result=continue_job(journal,store=Store(root),free_disk_bytes=shutil.disk_usage(journal.directory).free,**request)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
