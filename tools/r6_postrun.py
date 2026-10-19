"""Two fixed R6 postrun queries with durable intent/receipt; never a live launcher."""
import argparse
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from rlflow import policy
from rlflow.jobs import identity, owns
from rlflow.store import atomic, canonical, sha
from tools.r6_recovery import Journal, RecoveryError, checked, file_pin
from tools.r6_worker_processes import become_subreaper, cleanup
from tools.r6_model_worker import output_size, write
from tools.r6_budget import admissible

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ('--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu',
           '--query-compute-apps=pid,process_name,used_gpu_memory')
CAP = 1024**2


def validate(journal, key):
    state = journal.read(); entry = state['events'][key]; request = entry['request']
    if (entry['kind'] != 'postrun' or sha(entry['identity']) != key
            or entry['identity']['input_sha256'] != sha(request)
            or set(request) != {'job_event_key','closure','inventory','executable','ordinal'}
            or type(request['ordinal']) is not int or request['ordinal'] not in (0,1)):
        raise RecoveryError('POSTRUN_REQUEST_IDENTITY')
    expected_event = 'POSTRUN:'+request['job_event_key']+':'+str(request['ordinal'])
    if (entry['identity']['event_id'] != expected_event or entry['identity']['role'] != 'postrun'
            or entry['identity']['goal_id'] != state['goal_id']):
        raise RecoveryError('POSTRUN_CANONICAL_EVENT')
    matching = [e for e in state['events'].values() if e['kind'] == 'postrun'
        and e['identity']['job_id'] == entry['identity']['job_id']
        and e['request'].get('ordinal') == request['ordinal']]
    if len(matching) != 1:raise RecoveryError('POSTRUN_DUPLICATE_ORDINAL')
    job = journal.canonical_job(request['job_event_key'],state=state)
    if (job['kind'] != 'job' or job.get('closure') != request['closure']
            or job.get('inventory') != request['inventory']
            or job['identity']['job_id'] != entry['identity']['job_id']):
        raise RecoveryError('POSTRUN_JOB_BINDING')
    close = json.loads(checked(request['closure']).read_text())
    inv = json.loads(checked(request['inventory']).read_text())
    if (close['job_id'] != entry['identity']['job_id'] or close['event_key'] != request['job_event_key']
            or inv['job_id'] != close['job_id'] or inv['closure'] != request['closure']):
        raise RecoveryError('POSTRUN_CLOSURE_IDENTITY')
    for pin in close['evidence'].values():checked(pin)
    if entry['resources'].get('output_actual_plus_unreleased_reservations_bytes',0) < CAP+65536:
        raise RecoveryError('POSTRUN_OUTPUT_RESERVATION')
    spec = json.loads(checked(job['request']['spec']).read_text())
    executable = checked(request['executable'])
    if spec['backend'] == 'real':
        if state['budget']['capacity_pass'] is not True:
            raise RecoveryError('POSTRUN_REAL_CAPACITY_REQUIRED')
        from training_workloads.unified_h1.config import resource_binding_pins
        pins = resource_binding_pins()
        if request['executable'] != {'path':pins['nvidia_smi_path'],'sha256':pins['nvidia_smi_sha256']}:
            raise RecoveryError('POSTRUN_FROZEN_EXECUTABLE')
    elif (spec['backend'] != 'runner_cpu' or state['goal_id'] != 'R6-CPU-TEST-FIXTURE'
            or not executable.is_relative_to(journal.directory.parent.resolve())):
        raise RecoveryError('POSTRUN_CPU_FIXTURE_BOUNDARY')
    return entry, executable


def stop_paths(journal):
    return journal.stopped() + policy.stopped(ROOT/'coordination/research') + [str(p) for p in
        (journal.directory.parent/'STOP',ROOT/'STOP',ROOT/'coordination/STOP',
         ROOT/'coordination/commander/STOP',ROOT/'coordination/executor/STOP') if p.exists()]


def run_query(journal, key):
    """Separate bounded worker persists its identity before the query side effect."""
    entry, executable = validate(journal,key); request = entry['request']
    admissible(journal.read()['budget'])
    if not entry.get('dispatch_claim') or checked(entry['dispatch_claim']['worker']) != Path(__file__).resolve():
        raise RecoveryError('POSTRUN_DISPATCH_CLAIM_REQUIRED')
    directory = journal.directory/'postrun'/key;directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise RecoveryError('POSTRUN_EXISTING_WORKER_NO_RETRY')
    fd = os.open(directory/'ONCE',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:os.write(fd,b'one postrun query\n');os.fsync(fd)
    finally:os.close(fd)
    worker = identity(os.getpid())
    write(directory/'worker.json',{'identity':worker,'event_key':key},CAP-65536)
    argv = [str(executable),QUERIES[request['ordinal']],'--format=csv,noheader,nounits']
    write(directory/'request.json',{'request':request,'argv':argv},CAP-65536)
    proc = None; child = None; error = None; code = -1; cleaned = None
    start = time.monotonic(); interrupted = []
    previous = {s:signal.signal(s,lambda signum,frame:interrupted.append(signum)) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        if stop_paths(journal):raise RecoveryError('STOP_BEFORE_POSTRUN')
        become_subreaper()
        with (directory/'stdout').open('xb') as stdout,(directory/'stderr').open('xb') as stderr,selectors.DefaultSelector() as selector:
            proc = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''))
            child = identity(proc.pid)
            if child is None:raise RecoveryError('POSTRUN_CHILD_EXIT_BEFORE_IDENTITY')
            write(directory/'child.json',{'identity':child,'event_key':key},CAP-65536)
            selector.register(proc.stdout,selectors.EVENT_READ,stdout)
            selector.register(proc.stderr,selectors.EVENT_READ,stderr)
            while selector.get_map() or proc.poll() is None:
                if interrupted or stop_paths(journal):raise RecoveryError('STOP_DURING_POSTRUN')
                if time.monotonic()-start > 15:raise RecoveryError('POSTRUN_TIMEOUT')
                for ready,_ in selector.select(.1):
                    data = os.read(ready.fileobj.fileno(),65536)
                    if not data:
                        selector.unregister(ready.fileobj);ready.fileobj.close();continue
                    if output_size(directory)+len(data) > CAP-65536:
                        raise RecoveryError('POSTRUN_OUTPUT_CAP_BEFORE_WRITE')
                    ready.data.write(data);ready.data.flush()
            proc.wait();code = proc.returncode
            if code != 0:raise RecoveryError('POSTRUN_NONZERO_EXIT')
    except BaseException as exc:error = repr(exc)
    finally:
        if proc is not None:
            try:cleaned = cleanup(proc);code = proc.returncode
            except BaseException as exc:error = str(error)+'; '+repr(exc)
        for sig,handler in previous.items():signal.signal(sig,handler)
    for name in ('stdout','stderr'):
        path = directory/name
        if path.exists():
            with path.open('rb') as stream:os.fsync(stream.fileno())
    result = dict(event_key=key,request_sha256=sha(request),process=worker,child_identity=child,
        exit_code=code if type(code)is int else -1,error=error,cleanup=cleaned,ordinal=request['ordinal'],
        job_id=entry['identity']['job_id'],elapsed_seconds=time.monotonic()-start,
        stdout=file_pin(directory/'stdout') if (directory/'stdout').exists() else None,
        stderr=file_pin(directory/'stderr') if (directory/'stderr').exists() else None,
        automatic_retry=False,scientific_acceptance=False)
    write(directory/'receipt.json',result,CAP)
    return 0 if code == 0 and error is None else 1


def verified_receipt(journal,key):
    entry,_ = validate(journal,key)
    receipt = json.loads(checked(entry['receipt']).read_text())
    if (receipt.get('event_key') != key or receipt.get('request_sha256') != sha(entry['request'])
            or receipt.get('job_id') != entry['identity']['job_id']
            or receipt.get('ordinal') != entry['request']['ordinal']):
        raise RecoveryError('POSTRUN_RECEIPT_IDENTITY')
    for name in ('stdout','stderr'):
        if receipt.get(name):checked(receipt[name])
    if (receipt.get('exit_code') != 0 or receipt.get('error') is not None
            or (receipt.get('cleanup') or {}).get('status') != 'VERIFIED_NO_OWNED_DESCENDANTS'):
        raise RecoveryError('POSTRUN_FAILED_PRESERVED')
    if (owns(receipt.get('process')) or owns(receipt.get('child_identity'))
            or any(owns(i) for i in receipt['cleanup'].get('observed_identities',[]))):
        raise RecoveryError('POSTRUN_OWNED_PROCESS_ALIVE')
    return receipt


def owned_tree(roots):
    """Reuse rlflow's coherent ancestry snapshots; never admit by numeric PID alone."""
    from rlflow.lifeguard import _process_table, _ancestry_chain
    first = _process_table(); second = _process_table(); third = _process_table()
    found = {}
    for owner in roots:
        if not owner or first.get(owner['pid'],{}).get('identity') != owner:continue
        for pid in first:
            chain = _ancestry_chain(pid,owner['pid'],first)
            if (chain and all(first[p] == second.get(p) == third.get(p) for p in chain)
                    and all(owns(first[p]['identity']) for p in chain)):
                ident = first[pid]['identity']
                if ident['pid'] != os.getpid():found[(ident['pid'],ident['starttime'])] = ident
    return list(found.values())


def cancel_bounded(journal,key,reason,proc=None,*,directory_kind="postrun",status_prefix="POSTRUN"):
    """Persist owned identities before signals. Never manufacture a query receipt.

    Five seconds maximum per recovery invocation; original cancellation start is
    retained on restart so repeated recovery does not restart the TERM grace.
    Unknown ownership remains ambiguous and cannot authorize another query.
    """
    state = journal.read(); entry = state['events'][key]
    if directory_kind not in {'postrun','cpu-analysis'} or status_prefix not in {'POSTRUN','CPU_ANALYSIS'}:
        raise RecoveryError('CANCEL_DIRECTORY_KIND')
    directory = journal.directory/directory_kind/key
    cancel = entry.get('controller_cancel')
    if cancel is None:
        cancel = dict(reason=reason,boot_id=identity(os.getpid())['boot_id'],
            started_monotonic=time.monotonic(),known=[],status='CANCELLING')
        entry['controller_cancel'] = cancel;journal.save(state)
    end = time.monotonic()+5; empty = 0
    while True:
        if proc is not None:proc.poll()
        state = journal.read(); entry = state['events'][key];cancel = entry['controller_cancel']
        roots = [entry.get('process')]
        for name in ('worker.json','child.json'):
            path = directory/name
            if path.exists():
                record = json.loads(path.read_text())
                if record['event_key'] != key:raise RecoveryError('POSTRUN_CANCEL_IDENTITY')
                roots.append(record['identity'])
        roots += cancel['known']
        discovered = owned_tree(roots)
        for ident in discovered:
            if ident not in cancel['known']:cancel['known'].append(ident)
        if len(cancel['known']) > 128:raise RecoveryError('POSTRUN_CLEANUP_IDENTITY_BOUND')
        journal.save(state)  # before any kill; crash cannot discard known ownership
        elapsed = time.monotonic()-cancel['started_monotonic']
        sig = signal.SIGKILL if cancel['boot_id'] != identity(os.getpid())['boot_id'] or elapsed >= 1 else signal.SIGTERM
        for ident in cancel['known']:
            if owns(ident):
                try:os.kill(ident['pid'],sig)
                except ProcessLookupError:pass
            if proc is None or ident['pid'] != proc.pid:
                try:os.waitpid(ident['pid'],os.WNOHANG)
                except ChildProcessError:pass
        if proc is not None:proc.poll()
        live = [i for i in cancel['known'] if owns(i)]
        empty = empty+1 if not live and not discovered else 0
        if empty >= 2 or time.monotonic() >= end:
            state = journal.read();cancel = state['events'][key]['controller_cancel']
            cancel.update(status='NO_KNOWN_OWNED_SURVIVORS' if empty >= 2 else 'CLEANUP_BOUND_EXCEEDED',
                survivors=live,stable_empty_scans=empty)
            state['status'] = status_prefix+'_CANCELLED_REQUIRES_RECONCILIATION';journal.save(state)
            if live:raise RecoveryError('POSTRUN_CLEANUP_BOUND_EXCEEDED')
            return
        time.sleep(.05)


def postrun_handoff(journal,key,*,poll_seconds=1):
    """Recover one exact query. An unbound dispatch claim is never retried."""
    journal.require_writer()
    if type(poll_seconds) not in (int,float) or not 0 < poll_seconds <= 60:
        raise RecoveryError('POSTRUN_POLL_INTERVAL')
    entry,_ = validate(journal,key)
    if entry.get('receipt'):return verified_receipt(journal,key)
    directory = journal.directory/'postrun'/key;proc = None
    if not entry.get('dispatch_claim'):
        journal.require_dispatch_budget()
        if stop_paths(journal):raise RecoveryError('STOP_BEFORE_POSTRUN')
        if directory.exists() or entry.get('process') or entry['status'] != 'INTENT':
            raise RecoveryError('POSTRUN_UNCLAIMED_SIDE_EFFECT')
        # Check normal cleanup at dispatch, not just a historical boolean.
        from tools.r6_job_handoff import no_model_workers
        state = journal.read();jobentry = state['events'][entry['request']['job_event_key']]
        # The prior close record binds Store output; checking every recorded
        # identity here prevents postrun while an owned process remains alive.
        close = json.loads(checked(jobentry['closure']).read_text())
        records = {name:json.loads(checked(pin).read_text()) for name,pin in close['evidence'].items()}
        ids = [records['lifeguard-ack.json']['owner'],records['lifeguard-ready.json']['identity'],
            records['process-identity.json'],*records['descendant-ledger.json']['descendants'],
            *records['descendant-cleanup.json']['discovered']]
        if any(owns(i) for i in ids):raise RecoveryError('POSTRUN_JOB_STILL_ALIVE')
        no_model_workers(journal)
        entry['dispatch_claim'] = {'controller':identity(os.getpid()),'worker':file_pin(Path(__file__)),
            'deadline_monotonic':time.monotonic()+45,'deadline_wall':time.time()+45}
        state['events'][key] = entry;state['status'] = 'POSTRUN_QUERY';journal.save(state)
        become_subreaper()
        proc = subprocess.Popen([sys.executable,'-B','-m','tools.r6_postrun',str(journal.directory.resolve()),key],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.:tools/rlflow',CUDA_VISIBLE_DEVICES=''))
        ident = identity(proc.pid)
        if ident is not None:journal.bind_process(key,ident)
    while True:
        if proc is not None:proc.poll()
        entry = journal.read()['events'][key]
        workerpath = directory/'worker.json'
        if entry['process'] is None and workerpath.exists():
            worker = json.loads(workerpath.read_text())
            if worker['event_key'] != key:raise RecoveryError('POSTRUN_WORKER_EVENT')
            if owns(worker['identity']):journal.bind_process(key,worker['identity'])
        action = journal.observe(key,receipt_path=directory/'receipt.json')['action']
        if action == 'ACCEPT_RECEIPT':
            journal.consume(key,directory/'receipt.json')
            return verified_receipt(journal,key)
        if action == 'ALREADY_CONSUMED':return verified_receipt(journal,key)
        if action == 'RECONCILE':
            cancel_bounded(journal,key,'MISSING_WORKER_OR_RECEIPT',proc)
            raise RecoveryError('POSTRUN_AMBIGUOUS_NO_RETRY')
        if action not in ('WAIT_OWNED_WORKER','WAIT_OWNED_EXIT','CANCEL_OWNED'):
            raise RecoveryError('POSTRUN_UNEXPECTED_OBSERVATION')
        claim = entry.get('dispatch_claim',{})
        deadline = claim.get('deadline_monotonic')
        valid_deadline = (type(deadline) in (int,float) and math.isfinite(deadline)
            and claim.get('controller',{}).get('boot_id') == identity(os.getpid())['boot_id'])
        if entry.get('controller_cancel') or stop_paths(journal) or not valid_deadline or time.monotonic() >= deadline:
            cancel_bounded(journal,key,'STOP_OR_DISPATCH_DEADLINE',proc)
            raise RecoveryError('POSTRUN_CANCELLED_NO_RETRY')
        time.sleep(min(poll_seconds,max(.001,deadline-time.monotonic())))


def register_postruns(journal, job_key, executable, *, free_disk_bytes):
    """Reserve both fixed ordinals before launch; changed requests cannot rerun one."""
    journal.guard()
    state = journal.read(); job = journal.canonical_job(job_key,state=state)
    if job['kind'] != 'job' or not job.get('closure') or not job.get('inventory'):
        raise RecoveryError('POSTRUN_CLOSED_INVENTORIED_JOB_REQUIRED')
    keys = []
    for ordinal in (0,1):
        request = dict(job_event_key=job_key,closure=job['closure'],inventory=job['inventory'],
            executable=executable,ordinal=ordinal)
        for prior in journal.read()['events'].values():
            if (prior['kind'] == 'postrun' and prior['identity']['job_id'] == job['identity']['job_id']
                    and prior['request'].get('ordinal') == ordinal and prior['request'] != request):
                raise RecoveryError('POSTRUN_ORDINAL_ALREADY_BOUND')
        event = dict(goal_id=state['goal_id'],event_id='POSTRUN:'+job_key+':'+str(ordinal),
            job_id=job['identity']['job_id'],input_sha256=sha(request),role='postrun')
        key,_ = journal.intent(event,kind='postrun',request=request,
            resources={'output_actual_plus_unreleased_reservations_bytes':CAP+65536},
            purpose=job['purpose'],free_disk_bytes=free_disk_bytes)
        validate(journal,key);keys.append(key)
    return keys


def complete_postruns(journal, job_key, keys, *, poll_seconds=1):
    journal.require_writer()
    if len(keys) != 2 or len(set(keys)) != 2:raise RecoveryError('POSTRUN_TWO_EXACT_EVENTS')
    for ordinal,key in enumerate(keys):
        entry,_ = validate(journal,key)
        if entry['request']['ordinal'] != ordinal or entry['request']['job_event_key'] != job_key:
            raise RecoveryError('POSTRUN_ORDINAL_ORDER')
    for key in keys:postrun_handoff(journal,key,poll_seconds=poll_seconds)
    state = journal.read(); job = journal.canonical_job(job_key,state=state)
    result = dict(schema_version='r6-postrun-summary/1',job_event_key=job_key,job_id=job['identity']['job_id'],
        closure=job['closure'],inventory=job['inventory'],postrun_queries_used=2,
        receipts=[state['events'][key]['receipt'] for key in keys],
        postrun_complete=True,scientific_acceptance=False)
    path = journal.directory/'jobs'/job_key/'postrun-summary.json'
    if job.get('postrun'):
        if json.loads(checked(job['postrun']).read_text()) != result:raise RecoveryError('POSTRUN_SUMMARY_DRIFT')
        return result
    elif path.exists():
        if json.loads(path.read_text()) != result:raise RecoveryError('POSTRUN_SUMMARY_DRIFT')
    else:
        data = canonical(result)+b'\n'
        if len(data) > 65536:raise RecoveryError('POSTRUN_SUMMARY_CAP')
        atomic(path,data)
    job['postrun'] = file_pin(path)
    state['status'] = ('STOPPED_AFTER_OWNED_CLEANUP' if stop_paths(journal) or job.get('stop_seen')
        else 'JOB_BOUND_EXCEEDED' if job.get('deadline_seen') or job.get('inventory_violations') or state['budget'].get('observed_cap_excess')
        else 'JOB_EVIDENCE_INCOMPLETE' if not json.loads(checked(job['inventory']).read_text())['output_inventory_complete']
        else 'POSTRUN_COMPLETE_PENDING_ANALYSIS')
    journal.save(state)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('event_key');args=parser.parse_args()
    raise SystemExit(run_query(Journal(args.directory.resolve()),args.event_key))
