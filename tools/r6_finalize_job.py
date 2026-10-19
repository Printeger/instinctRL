"""Inventory one already closed job and complete its two exact postrun queries.

Optional CPU analysis follows; no submission, retry policy, model decision,
budget refund or science acceptance.
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


def analyze(journal, key, store, *, free_disk_bytes, python=None, program=None, timeout_seconds=3600):
    """Register once or reconnect the same bounded CPU analysis event.

Overrides exist for isolated CPU fixtures; the existing analysis validator
requires the frozen interpreter/analyzer for every real-mode job.
"""
    from tools import r6_cpu_analysis as cpu
    journal.require_writer()
    state=journal.read();job=journal.canonical_job(key,state=state)
    existing=[event_key for event_key,event in state['events'].items()
        if event['kind']=='cpu' and event['identity']['job_id']==job['identity']['job_id']
        and event['identity']['role']=='cpu-analysis']
    expected=dict(python=file_pin(cpu.PYTHON) if python is None else python,
        program=file_pin(cpu.ROOT/cpu.SOURCES[0]) if program is None else program,
        timeout_seconds=timeout_seconds)
    if existing:
        if len(existing)!=1:raise RecoveryError('FINALIZE_ANALYSIS_ALIAS')
        event_key=existing[0];entry,_=cpu.validate(journal,event_key)
        if (entry['request']['job_event_key']!=key
                or any(entry['request'][name]!=value for name,value in expected.items())):
            raise RecoveryError('FINALIZE_ANALYSIS_BINDING')
    else:
        event_key=cpu.register_analysis(journal,key,store,free_disk_bytes=free_disk_bytes,**expected)
    return cpu.handoff(journal,event_key,store)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('event_key')
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--analyze',action='store_true',help='Run/recover the bounded CPU replay after postrun')
    args=parser.parse_args(argv)
    from rlflow.store import Store
    journal=Journal(args.directory.resolve())
    with journal.writer():
        root=args.store.resolve(strict=True)
        if not (root/'runtime/research.db').is_file():raise RecoveryError('FINALIZE_STORE_DATABASE_REQUIRED')
        result=finalize(journal,args.event_key,Store(root),file_pin(args.executable),
                        free_disk_bytes=shutil.disk_usage(journal.directory).free)
        if args.analyze:
            result['analysis']=analyze(journal,args.event_key,Store(root),
                free_disk_bytes=shutil.disk_usage(journal.directory).free)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
