"""One exact, bounded two-query sample worker. No CUDA, simulation or submit."""
import argparse
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

from rlflow.jobs import identity,owns
from rlflow.store import sha
from tools.r6_recovery import Journal,RecoveryError,checked,file_pin
from tools.r6_model_worker import write,output_size
from tools.r6_worker_processes import become_subreaper,cleanup
from tools.r6_postrun import QUERIES,stop_paths
from tools.r6_budget import admissible
from tools import r6_resource_queue as queue

CAP=48*1024
LOG_CAP=16*1024


def prepare(journal,key,ordinal,executable,*,review=None):
    journal.guard()
    if stop_paths(journal):raise RecoveryError('STOP_BEFORE_RESOURCE_SAMPLE')
    state=journal.read();entry=queue.validate(journal,key,state)
    if type(ordinal) is not int or not 0<=ordinal<13:raise RecoveryError('RESOURCE_SAMPLE_ORDINAL')
    request=dict(queue_event=key,queue_request_sha256=sha(entry['request']),ordinal=ordinal,
        executable=executable,worker=file_pin(Path(__file__)),review=review)
    spec=json.loads(checked(entry['request']['spec']).read_text());exe=checked(executable)
    if spec['backend']=='real':
        if state['budget']['capacity_pass'] is not True:raise RecoveryError('RESOURCE_SAMPLE_CAPACITY')
        from training_workloads.unified_h1.config import resource_binding_pins
        pins=resource_binding_pins()
        accepted=json.loads(checked(review).read_text()) if review else {}
        if (executable!={'path':pins['nvidia_smi_path'],'sha256':pins['nvidia_smi_sha256']}
                or accepted.get('decision')!='PASS_R6_RESOURCE_SAMPLER'
                or accepted.get('worker')!=request['worker'] or accepted.get('must_fix')!=[]):
            raise RecoveryError('RESOURCE_SAMPLE_EXACT_REVIEW')
    elif state['goal_id']!='R6-CPU-TEST-FIXTURE' or not exe.is_relative_to(journal.directory.parent.resolve()):
        raise RecoveryError('RESOURCE_SAMPLE_CPU_FIXTURE')
    claims=entry.setdefault('sample_claims',{});name=str(ordinal)
    if name in claims:
        if claims[name]['request']!=request:raise RecoveryError('RESOURCE_SAMPLE_REPLAY_DRIFT')
        return claims[name]
    if state.get('current_job') is not None or state.get('current_model_event') is not None:
        raise RecoveryError('RESOURCE_SAMPLE_NOT_IDLE')
    from tools.r6_job_handoff import no_model_workers
    no_model_workers(journal)
    if any(e['kind']=='cpu' and k!=key and e.get('dispatch_claim') and not e.get('receipt') for k,e in state['events'].items()):
        raise RecoveryError('RESOURCE_SAMPLE_OTHER_CPU_UNRESOLVED')
    if entry['queue_status']!='COLLECTING' or ordinal!=len(entry['samples']):raise RecoveryError('RESOURCE_SAMPLE_SEQUENCE')
    if any(not c.get('receipt') for c in claims.values()):raise RecoveryError('RESOURCE_SAMPLE_UNRESOLVED')
    _,elapsed=queue.tick(entry)
    if elapsed+60>=queue.SLOT_SECONDS:raise RecoveryError('RESOURCE_SAMPLE_INSUFFICIENT_WINDOW_TIME')
    journal.require_dispatch_budget()
    claims[name]=dict(request=request,request_sha256=sha(request),controller=identity(os.getpid()),
        deadline_monotonic=time.monotonic()+60,process=None,receipt=None)
    journal.save(state);return claims[name]


def parse(gpu_text,apps_text,stamp):
    """Strict CSV conversion; V2 validates identity, memory, apps and full window."""
    from training_workloads.scale_003.resources import _integer
    rows=[line for line in gpu_text.splitlines() if line.strip()]
    if len(rows)!=1:raise ValueError('RESOURCE_GPU_ROW_COUNT')
    fields=[v.strip() for v in rows[0].split(',')]
    if len(fields)!=6:raise ValueError('RESOURCE_GPU_FIELDS')
    total=_integer(fields[3]);used=_integer(fields[4]);apps=[]
    text=apps_text.strip()
    if text and text!='No running processes found':
        for line in text.splitlines():
            values=[v.strip() for v in line.split(',')]
            if len(values)!=3:raise ValueError('RESOURCE_APP_FIELDS')
            apps.append(dict(pid=int(values[0]),name=values[1],used_memory_mib=_integer(values[2])))
    return dict(monotonic_ns=stamp,gpu=dict(index=int(fields[0]),name=fields[1],driver=fields[2],
        memory_total_mib=total,memory_used_mib=used,free_memory_mib=total-used,utilization_percent=_integer(fields[5])),compute_apps=apps)


def run(journal,key,ordinal):
    entry=queue.validate(journal,key)
    if entry['queue_status']!='COLLECTING' or ordinal!=len(entry['samples']):raise RecoveryError('RESOURCE_SAMPLE_SEQUENCE')
    claim=entry['sample_claims'][str(ordinal)];request=claim['request']
    if (claim['request_sha256']!=sha(request) or request['queue_event']!=key or request['ordinal']!=ordinal
            or request['queue_request_sha256']!=sha(entry['request']) or checked(request['worker'])!=Path(__file__).resolve()):
        raise RecoveryError('RESOURCE_SAMPLE_CLAIM')
    state=journal.read();admissible(state['budget'])
    if request['review'] is not None:checked(request['review'])
    if (claim['controller']['boot_id']!=identity(os.getpid())['boot_id']
            or time.monotonic()>=claim['deadline_monotonic']):raise RecoveryError('RESOURCE_SAMPLE_CLAIM_EXPIRED')
    spec=json.loads(checked(entry['request']['spec']).read_text())
    if spec['backend']=='real' and state['budget']['capacity_pass'] is not True:raise RecoveryError('RESOURCE_SAMPLE_CAPACITY')
    directory=journal.directory/'resource-queue'/key/('sample-'+str(ordinal))
    directory.mkdir(parents=True,exist_ok=False)
    worker=identity(os.getpid());write(directory/'worker.json',dict(identity=worker,event_key=key,ordinal=ordinal),CAP)
    write(directory/'request.json',request,CAP)
    interrupted=[];previous={s:signal.signal(s,lambda sig,frame:interrupted.append(sig)) for s in (signal.SIGTERM,signal.SIGINT)}
    started=time.monotonic();records=[];error=None;sample=None;outputs=[]
    try:
        become_subreaper()
        for number,query in enumerate(QUERIES):
            if interrupted or stop_paths(journal):raise RecoveryError('STOP_DURING_RESOURCE_SAMPLE')
            _,elapsed=queue.tick(entry)
            if elapsed>=queue.SLOT_SECONDS or time.monotonic()-started>=30:raise RecoveryError('RESOURCE_SAMPLE_DEADLINE')
            exe=checked(request['executable']);proc=None;code=-1;cleaned=None;child=None;query_error=None
            out=directory/(str(number)+'.stdout');err=directory/(str(number)+'.stderr');start=time.monotonic()
            try:
                with out.open('xb') as stdout,err.open('xb') as stderr,selectors.DefaultSelector() as selector:
                    proc=subprocess.Popen([str(exe),query,'--format=csv,noheader,nounits'],stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''))
                    child=identity(proc.pid)
                    if child is None:raise RecoveryError('RESOURCE_QUERY_EXIT_BEFORE_IDENTITY')
                    write(directory/(str(number)+'.child.json'),dict(identity=child,event_key=key,ordinal=ordinal),CAP)
                    selector.register(proc.stdout,selectors.EVENT_READ,stdout);selector.register(proc.stderr,selectors.EVENT_READ,stderr)
                    logged=0
                    while selector.get_map() or proc.poll() is None:
                        if interrupted or stop_paths(journal):raise RecoveryError('STOP_DURING_RESOURCE_QUERY')
                        if time.monotonic()-start>15 or time.monotonic()-started>30:raise RecoveryError('RESOURCE_QUERY_TIMEOUT')
                        for ready,_ in selector.select(.1):
                            data=os.read(ready.fileobj.fileno(),4096)
                            if not data:selector.unregister(ready.fileobj);ready.fileobj.close();continue
                            if logged+len(data)>LOG_CAP or output_size(directory)+len(data)>CAP-16384:
                                raise RecoveryError('RESOURCE_QUERY_OUTPUT_CAP_BEFORE_WRITE')
                            ready.data.write(data);ready.data.flush();logged+=len(data)
                    proc.wait();code=proc.returncode
            except BaseException as exc:query_error=repr(exc)
            finally:
                if proc is not None:
                    try:cleaned=cleanup(proc);code=proc.returncode
                    except BaseException as exc:query_error=str(query_error)+'; '+repr(exc)
            for path in (out,err):
                with path.open('rb') as stream:os.fsync(stream.fileno())
            records.append(dict(query=query,child_identity=child,exit_code=code,error=query_error,cleanup=cleaned,
                stdout=file_pin(out),stderr=file_pin(err)))
            if query_error is not None or code!=0:raise RecoveryError('RESOURCE_QUERY_FAILED')
            outputs.append(out.read_text())
        sample=parse(*outputs,time.monotonic_ns())
    except BaseException as exc:error=repr(exc)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    result=dict(queue_event=key,ordinal=ordinal,request_sha256=sha(request),process=worker,
        exit_code=0 if error is None else 1,error=error,queries=records,sample=sample,
        started_monotonic=started,finished_monotonic=time.monotonic(),
        elapsed_seconds=time.monotonic()-started,automatic_retry=False,launch_admission=False)
    write(directory/'receipt.json',result,CAP)
    return result['exit_code']


def consume(journal,key,ordinal):
    journal.require_writer();state=journal.read();entry=queue.validate(journal,key,state);claim=entry['sample_claims'][str(ordinal)]
    path=journal.directory/'resource-queue'/key/('sample-'+str(ordinal))/'receipt.json'
    if not path.exists():raise RecoveryError('RESOURCE_SAMPLE_RECEIPT_MISSING_NO_RETRY')
    receipt=json.loads(path.read_text())
    worker=json.loads((path.parent/'worker.json').read_text())
    if worker!={'identity':receipt['process'],'event_key':key,'ordinal':ordinal}:
        raise RecoveryError('RESOURCE_SAMPLE_WORKER_BINDING')
    checked(claim['request']['worker']);checked(claim['request']['executable'])
    if (receipt['queue_event']!=key or receipt['ordinal']!=ordinal or receipt['request_sha256']!=claim['request_sha256']
            or owns(receipt['process'])):raise RecoveryError('RESOURCE_SAMPLE_RECEIPT_IDENTITY_OR_LIVE')
    pin=file_pin(path)
    if claim.get('receipt') and claim['receipt']!=pin:raise RecoveryError('RESOURCE_SAMPLE_RECEIPT_DRIFT')
    if not isinstance(receipt['queries'],list) or len(receipt['queries'])>2:raise RecoveryError('RESOURCE_SAMPLE_QUERY_COUNT')
    outputs=[]
    for index,row in enumerate(receipt['queries']):
        if (row['query']!=QUERIES[index]
                or (row['cleanup'] or {}).get('status')!='VERIFIED_NO_OWNED_DESCENDANTS'
                or any(owns(i) for i in [row['child_identity'],*row['cleanup'].get('observed_identities',[])])):
            raise RecoveryError('RESOURCE_SAMPLE_QUERY_NOT_CLEAN')
        if (checked(row['stderr'])!=path.parent/(str(index)+'.stderr')
                or checked(row['stdout'])!=path.parent/(str(index)+'.stdout')):
            raise RecoveryError('RESOURCE_SAMPLE_OUTPUT_BINDING')
        if receipt['exit_code']==0 and receipt['error'] is None:
            outputs.append(checked(row['stdout']).read_text())
    stamps=[receipt.get('started_monotonic'),receipt.get('finished_monotonic')]
    if (not all(type(t) in (int,float) and math.isfinite(t) for t in stamps)
            or stamps[1]<stamps[0] or stamps[1]>claim['deadline_monotonic'] or claim.get('cancel')):
        claim['receipt']=pin;claim['sample_failed']=True;journal.save(state)
        raise RecoveryError('RESOURCE_SAMPLE_LATE_OR_CANCELLED')
    if receipt['exit_code']!=0 or receipt['error'] is not None:
        claim['receipt']=pin;claim['sample_failed']=True;journal.save(state)
        raise RecoveryError('RESOURCE_SAMPLE_FAILED_PRESERVED')
    if stop_paths(journal):
        claim['receipt']=pin;claim['sample_rejected_after_stop']=True;journal.save(state)
        raise RecoveryError('STOP_RESOURCE_SAMPLE_PRESERVED')
    if len(receipt['queries'])!=2 or any(r['exit_code']!=0 or r['error'] is not None for r in receipt['queries']):
        raise RecoveryError('RESOURCE_SAMPLE_QUERY_FAILED')
    if parse(*outputs,receipt['sample']['monotonic_ns'])!=receipt['sample']:raise RecoveryError('RESOURCE_SAMPLE_PARSE_DRIFT')
    if claim.get('sample_stale'):raise RecoveryError('RESOURCE_SAMPLE_STALE_PRESERVED')
    if claim.get('receipt'):
        if entry['samples'][ordinal]!=receipt['sample']:raise RecoveryError('RESOURCE_SAMPLE_COMMITTED_DRIFT')
        return receipt['sample']
    clock=queue.now()
    if clock['boot_id']!=entry['queue_clock']['boot_id']:raise RecoveryError('RESOURCE_SAMPLE_BOOT_RECONCILIATION')
    if clock['monotonic_ns']-receipt['sample']['monotonic_ns']>queue.POLICY['maximum_sample_age_seconds']*10**9:
        claim['receipt']=pin;claim['sample_stale']=True;journal.save(state)
        raise RecoveryError('RESOURCE_SAMPLE_STALE_PRESERVED')
    queue.append_sample(journal,key,ordinal,receipt['sample'])
    state=journal.read();state['events'][key]['sample_claims'][str(ordinal)]['receipt']=pin;journal.save(state)
    return receipt['sample']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('key');p.add_argument('ordinal',type=int)
    a=p.parse_args();raise SystemExit(run(Journal(a.directory.resolve()),a.key,a.ordinal))
