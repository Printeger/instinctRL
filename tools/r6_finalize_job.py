"""Inventory one already closed job and complete its two exact postrun queries.

No submission, retry policy, model decision, budget refund or science acceptance.
"""
import argparse
import json
from pathlib import Path
import shutil

from tools.r6_job_inventory import record_inventory
from tools.r6_postrun import register_postruns, complete_postruns
from tools.r6_recovery import Journal, RecoveryError, checked, file_pin


def finalize(journal, key, store, executable, *, free_disk_bytes):
    journal.require_writer()
    checked(executable)
    inventory=record_inventory(journal,key,store)
    state=journal.read();job=journal.canonical_job(key,state=state)
    existing={}
    for event_key,event in state['events'].items():
        if event['kind']=='postrun' and event['identity']['job_id']==job['identity']['job_id']:
            ordinal=event['request'].get('ordinal')
            expected=dict(job_event_key=key,closure=job['closure'],inventory=job['inventory'],
                          executable=executable,ordinal=ordinal)
            if type(ordinal) is not int or ordinal not in (0,1) or ordinal in existing or event['request']!=expected:
                raise RecoveryError('FINALIZE_POSTRUN_BINDING')
            existing[ordinal]=event_key
    # Recover existing events directly, including after STOP. Registration of a
    # missing ordinal still traverses the normal STOP and budget gates.
    keys=([existing[0],existing[1]] if len(existing)==2 else
          register_postruns(journal,key,executable,free_disk_bytes=free_disk_bytes))
    postrun=complete_postruns(journal,key,keys)
    return dict(inventory=inventory,postrun=postrun,scientific_acceptance=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('event_key')
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--executable',type=Path,required=True)
    args=parser.parse_args(argv)
    from rlflow.store import Store
    journal=Journal(args.directory.resolve())
    with journal.writer():
        root=args.store.resolve(strict=True)
        if not (root/'runtime/research.db').is_file():raise RecoveryError('FINALIZE_STORE_DATABASE_REQUIRED')
        result=finalize(journal,args.event_key,Store(root),file_pin(args.executable),
                        free_disk_bytes=shutil.disk_usage(journal.directory).free)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
