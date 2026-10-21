"""Local R6 queue observation loop; never submits jobs or invokes a model."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rlflow.jobs import identity,owns
from tools.r6_recovery import RecoveryError
from tools.r6_postrun import owned_tree,stop_paths
from tools.r6_worker_processes import become_subreaper
from tools import r6_resource_queue as queue
from tools import r6_resource_sampler as sampler

ROOT=Path(__file__).resolve().parents[1]


def directory(journal,key,ordinal):return journal.directory/'resource-queue'/key/('sample-'+str(ordinal))


def cancel(journal,key,ordinal,reason,proc=None):
    """Bounded exact-identity recovery, including a stopped worker's children."""
    journal.require_writer();state=journal.read();claim=state['events'][key]['sample_claims'][str(ordinal)]
    if 'cancel' not in claim:
        claim['cancel']=dict(reason=reason,boot_id=identity(os.getpid())['boot_id'],started=time.monotonic(),known=[])
        journal.save(state)
    end=time.monotonic()+5;empty=0
    while True:
        if proc is not None:proc.poll()
        state=journal.read();claim=state['events'][key]['sample_claims'][str(ordinal)];record=claim['cancel']
        roots=[claim.get('process'),*record['known']]
        for name in ('worker.json','0.child.json','1.child.json'):
            path=directory(journal,key,ordinal)/name
            if path.exists():
                value=json.loads(path.read_text())
                if value['event_key']!=key or value['ordinal']!=ordinal:raise RecoveryError('RESOURCE_CANCEL_IDENTITY')
                roots.append(value['identity'])
        found=owned_tree(roots)
        for ident in found:
            if ident not in record['known']:record['known'].append(ident)
        if len(record['known'])>128:raise RecoveryError('RESOURCE_CANCEL_IDENTITY_BOUND')
        journal.save(state)
        sig=signal.SIGKILL if record['boot_id']!=identity(os.getpid())['boot_id'] or time.monotonic()-record['started']>=1 else signal.SIGTERM
        for ident in record['known']:
            if owns(ident):
                try:os.kill(ident['pid'],sig)
                except ProcessLookupError:pass
            if proc is None or ident['pid']!=proc.pid:
                try:os.waitpid(ident['pid'],os.WNOHANG)
                except ChildProcessError:pass
        if proc is not None:proc.poll()
        live=[i for i in record['known'] if owns(i)];empty=empty+1 if not live and not found else 0
        if empty>=2 or time.monotonic()>=end:
            state=journal.read();record=state['events'][key]['sample_claims'][str(ordinal)]['cancel']
            record.update(status='NO_KNOWN_OWNED_SURVIVORS' if empty>=2 else 'CLEANUP_BOUND_EXCEEDED',survivors=live,stable_empty_scans=empty)
            journal.save(state)
            if live:raise RecoveryError('RESOURCE_CANCEL_CLEANUP_BOUND')
            return
        time.sleep(.05)


def sample_handoff(journal,key,ordinal,executable,*,review=None,poll_seconds=.1):
    journal.require_writer()
    if type(poll_seconds) not in (int,float) or not 0<poll_seconds<=1:raise RecoveryError('RESOURCE_SAMPLE_POLL')
    state=journal.read();entry=queue.validate(journal,key,state)
    fresh=str(ordinal) not in entry.get('sample_claims',{})
    if fresh:claim=sampler.prepare(journal,key,ordinal,executable,review=review)
    else:
        claim=entry['sample_claims'][str(ordinal)]
        if claim['request']['executable']!=executable or claim['request']['review']!=review:
            raise RecoveryError('RESOURCE_SAMPLE_HANDOFF_DRIFT')
    folder=directory(journal,key,ordinal);proc=None
    if fresh:
        # Claim was persisted before any possible worker side effect. A restart
        # that sees this claim but no handle/receipt will never repeat Popen.
        become_subreaper()
        proc=subprocess.Popen([sys.executable,'-B','-m','tools.r6_resource_sampler',str(journal.directory.resolve()),key,str(ordinal)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.:tools/rlflow',CUDA_VISIBLE_DEVICES=''))
        ident=identity(proc.pid)
        if ident is not None:
            state=journal.read();state['events'][key]['sample_claims'][str(ordinal)]['process']=ident;journal.save(state)
    while True:
        if proc is not None:proc.poll()
        state=journal.read();claim=state['events'][key]['sample_claims'][str(ordinal)]
        path=folder/'worker.json'
        worker=None
        if path.exists():
            value=json.loads(path.read_text())
            if value['event_key']!=key or value['ordinal']!=ordinal:raise RecoveryError('RESOURCE_WORKER_IDENTITY')
            worker=value['identity']
            if claim.get('process') is not None and claim['process']!=worker:raise RecoveryError('RESOURCE_WORKER_PROCESS_DRIFT')
            if claim.get('process') is None and owns(worker):
                claim['process']=worker;journal.save(state)
        live=owns(claim.get('process')) or owns(worker)
        if not live and (folder/'receipt.json').exists():
            try:return sampler.consume(journal,key,ordinal)
            except RecoveryError:
                # Failed but clean receipts are already retained by consume.
                # Unverified descendants must still be reconciled locally.
                if not journal.read()['events'][key]['sample_claims'][str(ordinal)].get('receipt'):
                    cancel(journal,key,ordinal,'UNVERIFIED_RECEIPT_CLEANUP',proc)
                raise
            except (ValueError,KeyError,UnicodeError):
                cancel(journal,key,ordinal,'PARTIAL_OR_INVALID_RECEIPT',proc)
                raise RecoveryError('RESOURCE_SAMPLE_INVALID_RECEIPT_NO_RETRY')
        deadline=claim.get('deadline_monotonic')
        valid=type(deadline) in (int,float) and math.isfinite(deadline) and claim['controller']['boot_id']==identity(os.getpid())['boot_id']
        if claim.get('cancel') or stop_paths(journal) or not valid or time.monotonic()>=deadline:
            cancel(journal,key,ordinal,'STOP_OR_ORIGINAL_DEADLINE',proc)
            # A cooperative worker may leave a failure receipt during cleanup.
            # Preserve it if possible, but never synthesize one after SIGKILL.
            if (folder/'receipt.json').exists():
                try:sampler.consume(journal,key,ordinal)
                except (RecoveryError,ValueError,KeyError,UnicodeError):pass
            raise RecoveryError('RESOURCE_SAMPLE_CANCELLED_NO_RETRY')
        if not live:
            cancel(journal,key,ordinal,'MISSING_WORKER_AND_RECEIPT',proc)
            raise RecoveryError('RESOURCE_SAMPLE_AMBIGUOUS_NO_RETRY')
        time.sleep(min(poll_seconds,max(.001,deadline-time.monotonic())))


def collect_window(journal,key,executable,*,review=None):
    """Resume the exact window only; do not splice samples across interrupted gaps."""
    journal.require_writer()
    while True:
        entry=queue.validate(journal,key)
        if entry.get('window_record'):return queue.finish(journal,key)
        claims=entry.get('sample_claims',{})
        # Resume an appended sample whose receipt state commit was interrupted.
        pending=[int(n) for n,c in claims.items() if not c.get('receipt')]
        if pending:
            try:sample_handoff(journal,key,min(pending),executable,review=review)
            except RecoveryError as error:
                if str(error)!='RESOURCE_SAMPLE_STALE_PRESERVED':raise
                return queue.finish(journal,key,interrupted=True)
            continue
        if len(entry['samples'])==13:return queue.finish(journal,key)
        if stop_paths(journal):return queue.finish(journal,key,interrupted=True)
        if entry['samples']:
            last=entry['samples'][-1]['monotonic_ns'];age=(time.monotonic_ns()-last)/1e9
            if age>queue.POLICY['window_policy']['maximum_gap_seconds']:
                return queue.finish(journal,key,interrupted=True)
            delay=max(0,5-age)
            # This is a local CPU collection wait, never a model call.
            until=time.monotonic()+delay
            while time.monotonic()<until:
                if stop_paths(journal):return queue.finish(journal,key,interrupted=True)
                time.sleep(min(.5,until-time.monotonic()))
        sample_handoff(journal,key,len(entry['samples']),executable,review=review)


def wait_for_observation(journal,spec,executable,*,free_disk_bytes,queue_review=None,sampler_review=None):
    """Full windows + bounded local backoff. Returns observation only, no submit.

    Caller supplies current free-disk measurement; this function refreshes it at
    each slot. Original queue reservations are never implicitly refunded.
    """
    journal.require_writer()
    import shutil
    from rlflow.store import sha
    from tools.r6_recovery import checked
    value=json.loads(checked(spec).read_text());jid='J-'+sha({'id':value['id'],'spec':value})[:16]
    while True:
        state=journal.read()
        entries=[(k,e) for k,e in state['events'].items() if e['identity']['role']=='resource-window' and e['identity']['job_id']==jid]
        if entries:
            key,entry=max(entries,key=lambda pair:pair[1]['request']['ordinal'])
            queue.validate(journal,key)
            if entry['request']['spec']!=spec or entry['request']['review']!=queue_review:
                raise RecoveryError('RESOURCE_QUEUE_WAIT_BINDING')
            status=entry['queue_status']
            if status=='OBSERVATION_PASS':
                if queue.observation_fresh(journal,key):return queue.finish(journal,key)
                status='EXPIRED' # Control flow only; preserve the original PASS artifact.
            if status in {'INVALID_WINDOW','STOPPED'}:return queue.finish(journal,key)
            if status in {'BUSY','INTERRUPTED','EXPIRED'}:
                while True:
                    queue.account_wait(journal,key)
                    from tools.r6_budget import admissible
                    admissible(journal.read()['budget'])
                    if stop_paths(journal):raise RecoveryError('STOP_DURING_RESOURCE_BACKOFF')
                    current=queue.now();end=entry['finished_clock']
                    if current['boot_id']!=end['boot_id']:raise RecoveryError('QUEUE_BOOT_RECONCILIATION_REQUIRED')
                    remaining=60-(current['monotonic_ns']-end['monotonic_ns'])/1e9
                    if remaining<=0:break
                    time.sleep(min(15,remaining))
                ordinal=entry['request']['ordinal']+1
                key=queue.begin(journal,spec,ordinal=ordinal,free_disk_bytes=min(free_disk_bytes,shutil.disk_usage(journal.directory).free),review=queue_review)
        else:
            key=queue.begin(journal,spec,ordinal=0,free_disk_bytes=min(free_disk_bytes,shutil.disk_usage(journal.directory).free),review=queue_review)
        result=collect_window(journal,key,executable,review=sampler_review)
        if result['status']=='OBSERVATION_PASS' and not queue.observation_fresh(journal,key):continue
        if result['status'] not in {'BUSY','INTERRUPTED','EXPIRED'}:return result
