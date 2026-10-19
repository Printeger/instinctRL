"""R6 model handoff used by the local single-writer batch.

Consumes registered model events and closes already submitted jobs. No permission issuance, job
submission, budget reservation, automatic retry or scientific acceptance here.
Registered CPU analysis workers are supported; resource/live dispatch and next-plan phases remain incomplete.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

from rlflow.jobs import identity, owns
from tools.r6_recovery import Journal, RecoveryError, checked, file_pin
from tools.r6_model_worker import validate_request

ROOT=Path(__file__).resolve().parents[1]


def worker_environment():
    return dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
                PYTHONPATH='.:tools/rlflow',GIT_TERMINAL_PROMPT='0')


def verified_decision(journal,key):
    """Receipt recording is separate from validating the returned decision."""
    entry=journal.read()['events'][key]
    receipt=json.loads(checked(entry['receipt']).read_text())
    if receipt.get('exit_code')!=0 or receipt.get('error') is not None or receipt.get('schema_valid') is not True:
        raise RecoveryError('MODEL_FAILED_PRESERVE_RECEIPT')
    if receipt.get('cleanup',{}).get('status')!='VERIFIED_NO_OWNED_DESCENDANTS':
        raise RecoveryError('MODEL_CLEANUP_NOT_VERIFIED')
    for keyname in ('events','stderr','last_message'):
        checked(receipt[keyname])
    _,_,_,schema=validate_request(entry['request'])
    result=json.loads(checked(receipt['last_message']).read_text())
    import jsonschema
    jsonschema.validate(result,schema)
    return result


def model_handoff(journal,key,*,poll_seconds=15):
    """Caller holds the sole writer lock throughout the read-only review.

    Dispatch claim survives a crash before Popen; that ambiguous event is never
    retried. After Popen, worker.json permits exact reattachment even if the
    controller died before binding the PID into state. A receipt permits closing
    an already exited worker. STOP is handled by the worker and never removed.
    """
    journal.require_writer()
    if not 0<poll_seconds<=60:raise ValueError('LOCAL_POLL_INTERVAL')
    state=journal.read();entry=state['events'][key]
    if entry['kind']!='model':raise RecoveryError('MODEL_EVENT_REQUIRED')
    call=journal.directory/'calls'/key
    receipt_path=call/'receipt.json'
    process=None
    if entry['receipt']:
        return verified_decision(journal,key)
    if not entry.get('dispatch_claim'):
        if any(e['kind']=='cpu' and ((e.get('dispatch_claim') and not e.get('receipt'))
                or any(not c.get('receipt') for c in e.get('sample_claims',{}).values()))
                for e in state['events'].values()):
            raise RecoveryError('CPU_WORKER_HANDOFF_UNRESOLVED')
        journal.require_dispatch_budget()
        if entry['process'] is not None or entry['status']!='INTENT' or call.exists():
            raise RecoveryError('UNCLAIMED_SIDE_EFFECT_REQUIRES_RECONCILIATION')
        validate_request(entry['request'])
        entry['dispatch_claim']={'controller':identity(os.getpid()),'time':time.time(),
                                 'worker':file_pin(ROOT/'tools/r6_model_worker.py')}
        state.update(status='MODEL_REVIEW',current_model_event=key,controller_identity=identity(os.getpid()))
        journal.save(state)
        process=subprocess.Popen([sys.executable,'-B','-m','tools.r6_model_worker',
            str(journal.directory.resolve()),key],cwd=ROOT,env=worker_environment(),
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            start_new_session=True)
        live=identity(process.pid)
        if live is not None:journal.bind_process(key,live)
    while True:
        if process is not None:process.poll() # Reap our child before checking owns().
        entry=journal.read()['events'][key]
        # Worker identity is persisted before any model side effect. The event
        # key plus exact Linux start/boot identity prevents PID-only attachment.
        worker_path=call/'worker.json'
        if entry['process'] is None and worker_path.exists():
            worker=json.loads(worker_path.read_text())
            if worker['event_key']!=key:raise RecoveryError('WORKER_EVENT_MISMATCH')
            if owns(worker['identity']):journal.bind_process(key,worker['identity'])
        action=journal.observe(key,receipt_path=receipt_path)['action']
        if action=='ACCEPT_RECEIPT':
            journal.consume(key,receipt_path)
            state=journal.read();state.update(status='MODEL_RECEIPT_RECORDED',current_model_event=None)
            journal.save(state)
            return verified_decision(journal,key)
        if action=='ALREADY_CONSUMED':return verified_decision(journal,key)
        if action=='RECONCILE':
            # A live process owned by this invocation may not have created its
            # worker.json yet; its exact Popen identity is already bound above.
            raise RecoveryError('MODEL_AMBIGUOUS_NO_RETRY')
        if action not in ('WAIT_OWNED_WORKER','WAIT_OWNED_EXIT','CANCEL_OWNED'):
            raise RecoveryError('UNEXPECTED_MODEL_OBSERVATION: '+action)
        # STOP is observed by the worker's bounded local loop, including cleanup.
        # No second model is called to ask whether the first one has finished.
        time.sleep(poll_seconds)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path);ap.add_argument('event_key')
    action=ap.add_mutually_exclusive_group()
    action.add_argument('--job-store',type=Path,help='Close an existing exact job in this rlflow Store; never submit')
    action.add_argument('--inventory-store',type=Path,help='Inventory exact closed-job originals; no GPU queries')
    action.add_argument('--postrun-keys',nargs=2,metavar=('GPU_EVENT','PROCESS_EVENT'),help='Complete two registered postrun queries for event_key job')
    action.add_argument('--decision',action='store_true',help='Recover a registered post-analysis decision and validate its proposal')
    action.add_argument('--cpu-analysis',action='store_true',help='Execute/recover a registered CPU analysis event; never call a model')
    action.add_argument('--analysis-report',type=Path,help='Bind an existing CPU motion replay report; no scientific acceptance')
    ap.add_argument('--analysis-store',type=Path,help='Exact Store for --analysis-report')
    a=ap.parse_args();journal=Journal(a.directory.resolve())
    if bool(a.analysis_report is not None or a.cpu_analysis or a.decision) != (a.analysis_store is not None):
        ap.error('--analysis-report, --cpu-analysis or --decision requires --analysis-store')
    with journal.writer():
        if a.decision:
            from rlflow.store import Store
            from tools.r6_decision import handoff
            result=handoff(journal,a.event_key,Store(a.analysis_store.resolve()))
        elif a.cpu_analysis:
            from rlflow.store import Store
            from tools.r6_cpu_analysis import handoff
            result=handoff(journal,a.event_key,Store(a.analysis_store.resolve()))
        elif a.analysis_report is not None:
            from rlflow.store import Store
            from tools.r6_analysis_binding import record_analysis
            result=record_analysis(journal,a.event_key,file_pin(a.analysis_report),Store(a.analysis_store.resolve()))
        elif a.postrun_keys is not None:
            from tools.r6_postrun import complete_postruns
            result=complete_postruns(journal,a.event_key,a.postrun_keys)
        elif a.inventory_store is not None:
            from rlflow.store import Store
            from tools.r6_job_inventory import record_inventory
            result=record_inventory(journal,a.event_key,Store(a.inventory_store.resolve()))
        elif a.job_store is None:
            result=model_handoff(journal,a.event_key)
        else:
            from rlflow.store import Store
            from tools.r6_job_handoff import job_handoff
            result=job_handoff(journal,a.event_key,Store(a.job_store.resolve()))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
