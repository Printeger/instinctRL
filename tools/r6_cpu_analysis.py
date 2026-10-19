"""Bounded CPU replay after exact job closure. No model calls or live dispatch."""
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

from rlflow.jobs import identity, owns
from rlflow.store import sha
from tools.r6_recovery import Journal, RecoveryError, checked, file_pin
from tools.r6_analysis_binding import analysis_context, record_analysis
from tools.r6_model_worker import write, output_size
from tools.r6_postrun import stop_paths, cancel_bounded
from tools.r6_worker_processes import become_subreaper, cleanup
from tools.r6_budget import admissible

ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path('/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11')
LOG_CAP=1024**2
RESERVATION=20*1024**2
SOURCES=('tools/analyze_r6_evaluation.py','tools/r6_cpu_analysis.py',
 'tools/r6_analysis_binding.py','tools/r6_recovery.py','tools/r6_postrun.py',
 'tools/r6_worker_processes.py','tools/r6_model_worker.py',
 'training_workloads/unified_h1/robustness_analysis.py','training_workloads/unified_h1/robustness.py',
 'training_workloads/unified_h1/robustness_measurement_replay.py',
 'training_workloads/unified_h1/robustness_physics_replay.py',
 'training_workloads/unified_h1/robustness_audit.py','training_workloads/unified_h1/force_archive.py',
 'training_workloads/unified_h1/range_archive.py','isaaclab-training/instinctlab/observations/robustness.py')


def environment():
    # Explicit tensor runtime, CPU only. The controller remains in its own venv.
    return dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
        PYTHONPATH='.:tools/rlflow:isaaclab-training',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')


def validate(journal,key):
    state=journal.read();entry=state['events'][key];request=entry['request']
    if (entry['kind']!='cpu' or sha(entry['identity'])!=key or entry['identity']['input_sha256']!=sha(request)
            or set(request)!={'job_event_key','spec','closure','inventory','postrun','python','program','sources','timeout_seconds'}):
        raise RecoveryError('CPU_ANALYSIS_REQUEST_IDENTITY')
    job=journal.canonical_job(request['job_event_key'],state=state)
    expected=dict(goal_id=state['goal_id'],event_id='CPU_ANALYSIS:'+request['job_event_key'],
        job_id=job['identity']['job_id'],input_sha256=sha(request),role='cpu-analysis')
    if entry['identity']!=expected:raise RecoveryError('CPU_ANALYSIS_CANONICAL_EVENT')
    matches=[e for e in state['events'].values() if e['kind']=='cpu' and e['identity']['job_id']==expected['job_id']
        and e['identity']['role']=='cpu-analysis']
    if len(matches)!=1:raise RecoveryError('CPU_ANALYSIS_ALIAS')
    if (request['spec']!=job['request']['spec'] or any(request[n]!=job.get(n) for n in ('closure','inventory','postrun'))):
        raise RecoveryError('CPU_ANALYSIS_JOB_BINDING')
    for n in ('spec','closure','inventory','postrun','python','program'):checked(request[n])
    if type(request['timeout_seconds']) is not int or not 1<=request['timeout_seconds']<=3600:
        raise RecoveryError('CPU_ANALYSIS_TIMEOUT_BOUND')
    if entry['resources']!={'output_actual_plus_unreleased_reservations_bytes':RESERVATION}:
        raise RecoveryError('CPU_ANALYSIS_RESERVATION')
    if request['sources']!=[file_pin(ROOT/p) for p in SOURCES]:raise RecoveryError('CPU_ANALYSIS_SOURCE_DRIFT')
    spec=json.loads(checked(request['spec']).read_text())
    if spec.get('backend')=='real':
        if request['python']!=file_pin(PYTHON) or request['program']!=file_pin(ROOT/SOURCES[0]):
            raise RecoveryError('CPU_ANALYSIS_EXECUTABLE')
    elif (spec.get('backend')!='runner_cpu' or state['goal_id']!='R6-CPU-TEST-FIXTURE'
            or not checked(request['program']).is_relative_to(journal.directory.parent.resolve())):
        raise RecoveryError('CPU_ANALYSIS_FIXTURE_BOUNDARY')
    auth=json.loads(checked(spec['pins']['authorization']).read_text())
    if spec['h1']['mode']!='research-r6-evaluation' or auth['mode']!=spec['h1']['mode']:
        raise RecoveryError('CPU_ANALYSIS_MODE')
    params=auth['parameters']['r6'];checked(params['protocol'])
    output=Path(spec['h1']['attempt'])/'results/research-r6-evaluation'
    directory=journal.directory/'cpu-analysis'/key
    if output.resolve()!=output or any(output==p or output in p.parents or p in output.parents for p in (directory.resolve(),)):
        raise RecoveryError('CPU_ANALYSIS_OUTPUT_SEPARATION')
    argv=[str(checked(request['python'])),'-B',str(checked(request['program'])),str(output),params['protocol']['path'],
        str(directory.resolve()/'result'),'--run-id',params['run_id'],'--seed',str(params['seed']),'--split',params['split']]
    return entry,argv


def register_analysis(journal,job_key,store,*,free_disk_bytes,python=None,program=None,timeout_seconds=3600):
    journal.guard();state=journal.read();job=journal.canonical_job(job_key,state=state)
    analysis_context(job,job_key,store)
    request=dict(job_event_key=job_key,spec=job['request']['spec'],closure=job['closure'],inventory=job['inventory'],
        postrun=job['postrun'],python=file_pin(PYTHON) if python is None else python,
        program=file_pin(ROOT/SOURCES[0]) if program is None else program,
        sources=[file_pin(ROOT/p) for p in SOURCES],timeout_seconds=timeout_seconds)
    for entry in state['events'].values():
        if (entry['kind']=='cpu' and entry['identity']['job_id']==job['identity']['job_id']
                and entry['identity']['role']=='cpu-analysis' and entry['request']!=request):
            raise RecoveryError('CPU_ANALYSIS_ALREADY_BOUND')
    event=dict(goal_id=state['goal_id'],event_id='CPU_ANALYSIS:'+job_key,job_id=job['identity']['job_id'],
        input_sha256=sha(request),role='cpu-analysis')
    key,_=journal.intent(event,kind='cpu',request=request,
        resources={'output_actual_plus_unreleased_reservations_bytes':RESERVATION},purpose=job['purpose'],free_disk_bytes=free_disk_bytes)
    validate(journal,key)
    return key


def run(journal,key):
    entry,argv=validate(journal,key);request=entry['request'];admissible(journal.read()['budget'])
    if not entry.get('dispatch_claim') or checked(entry['dispatch_claim']['worker'])!=Path(__file__).resolve():
        raise RecoveryError('CPU_ANALYSIS_DISPATCH_REQUIRED')
    directory=journal.directory/'cpu-analysis'/key;directory.mkdir(parents=True,exist_ok=False)
    worker=identity(os.getpid());write(directory/'worker.json',dict(identity=worker,event_key=key),LOG_CAP)
    write(directory/'request.json',dict(request=request,argv=argv),LOG_CAP)
    proc=None;child=None;error=None;code=-1;cleaned=None;outputs={}
    start=time.monotonic();interrupted=[]
    previous={s:signal.signal(s,lambda signum,frame:interrupted.append(signum)) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        if stop_paths(journal):raise RecoveryError('STOP_BEFORE_CPU_ANALYSIS')
        become_subreaper()
        with (directory/'stdout').open('xb') as stdout,(directory/'stderr').open('xb') as stderr,selectors.DefaultSelector() as selector:
            proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                start_new_session=True,cwd=ROOT,env=environment())
            child=identity(proc.pid)
            if child is None:raise RecoveryError('CPU_ANALYSIS_CHILD_EXIT_BEFORE_IDENTITY')
            write(directory/'child.json',dict(identity=child,event_key=key),LOG_CAP)
            selector.register(proc.stdout,selectors.EVENT_READ,stdout);selector.register(proc.stderr,selectors.EVENT_READ,stderr)
            while selector.get_map() or proc.poll() is None:
                if interrupted or stop_paths(journal):raise RecoveryError('STOP_DURING_CPU_ANALYSIS')
                if time.monotonic()-start>request['timeout_seconds']:raise RecoveryError('CPU_ANALYSIS_TIMEOUT')
                for ready,_ in selector.select(.1):
                    data=os.read(ready.fileobj.fileno(),65536)
                    if not data:selector.unregister(ready.fileobj);ready.fileobj.close();continue
                    if output_size(directory)+len(data)>LOG_CAP-65536:raise RecoveryError('CPU_ANALYSIS_LOG_CAP_BEFORE_WRITE')
                    ready.data.write(data);ready.data.flush()
            proc.wait();code=proc.returncode
            if code!=0:raise RecoveryError('CPU_ANALYSIS_NONZERO_EXIT')
    except BaseException as exc:error=repr(exc)
    finally:
        if proc is not None:
            try:cleaned=cleanup(proc);code=proc.returncode
            except BaseException as exc:error=str(error)+'; '+repr(exc)
        for sig,handler in previous.items():signal.signal(sig,handler)
    try:
        for name,limit in (('report.json',16*1024**2),('summary.json',48*1024)):
            path=directory/'result'/name
            if not path.is_file() or path.is_symlink() or path.stat().st_size>limit:raise RecoveryError('CPU_ANALYSIS_RESULT_MISSING_OR_BOUND')
            outputs[name]=file_pin(path)
    except BaseException as exc:error=str(error)+'; '+repr(exc)
    for name in ('stdout','stderr'):
        path=directory/name
        if path.exists():
            with path.open('rb') as stream:os.fsync(stream.fileno())
    result=dict(event_key=key,request_sha256=sha(request),process=worker,child_identity=child,
        exit_code=code if type(code)is int else -1,error=error,cleanup=cleaned,
        job_id=entry['identity']['job_id'],elapsed_seconds=time.monotonic()-start,outputs=outputs,
        stdout=file_pin(directory/'stdout') if (directory/'stdout').exists() else None,
        stderr=file_pin(directory/'stderr') if (directory/'stderr').exists() else None,
        automatic_retry=False,model_calls=0,scientific_acceptance=False)
    write(directory/'receipt.json',result,LOG_CAP)
    return 0 if code==0 and error is None else 1


def verified_receipt(journal,key):
    entry,_=validate(journal,key);r=json.loads(checked(entry['receipt']).read_text())
    if (r.get('event_key')!=key or r.get('request_sha256')!=sha(entry['request'])
            or r.get('job_id')!=entry['identity']['job_id']):raise RecoveryError('CPU_ANALYSIS_RECEIPT_IDENTITY')
    if (r.get('exit_code')!=0 or r.get('error') is not None
            or (r.get('cleanup') or {}).get('status')!='VERIFIED_NO_OWNED_DESCENDANTS'):
        raise RecoveryError('CPU_ANALYSIS_FAILED_PRESERVED')
    if any(owns(i) for i in [r.get('process'),r.get('child_identity'),*r['cleanup'].get('observed_identities',[])]):
        raise RecoveryError('CPU_ANALYSIS_PROCESS_ALIVE')
    for n in ('stdout','stderr'):checked(r[n])
    directory=journal.directory.resolve()/'cpu-analysis'/key/'result'
    for name,limit in (('report.json',16*1024**2),('summary.json',48*1024)):
        path=checked(r['outputs'][name])
        if path!=directory/name or path.stat().st_size>limit:raise RecoveryError('CPU_ANALYSIS_RESULT_BINDING')
    report=json.loads(checked(r['outputs']['report.json']).read_text())
    summary=json.loads(checked(r['outputs']['summary.json']).read_text())
    if summary!=report['summary']:raise RecoveryError('CPU_ANALYSIS_SUMMARY_DRIFT')
    return r


def handoff(journal,key,store,*,poll_seconds=1):
    journal.require_writer()
    if type(poll_seconds) not in (int,float) or not 0<poll_seconds<=60:raise RecoveryError('CPU_ANALYSIS_POLL')
    entry,_=validate(journal,key);job_key=entry['request']['job_event_key']
    state=journal.read();analysis_context(journal.canonical_job(job_key,state=state),job_key,store)
    if state.get('current_job') is not None or state.get('current_model_event') is not None:raise RecoveryError('CPU_ANALYSIS_NOT_IDLE')
    directory=journal.directory/'cpu-analysis'/key;proc=None
    if not entry.get('receipt') and not entry.get('dispatch_claim'):
        journal.require_dispatch_budget()
        if stop_paths(journal):raise RecoveryError('STOP_BEFORE_CPU_ANALYSIS')
        from tools.r6_job_handoff import no_model_workers
        no_model_workers(journal)
        for other_key,e in state['events'].items():
            if e['kind']=='cpu' and other_key!=key and e.get('dispatch_claim') and not e.get('receipt'):
                raise RecoveryError('CPU_ANALYSIS_OTHER_EVENT_UNRESOLVED')
        if directory.exists() or entry.get('process') or entry['status']!='INTENT':raise RecoveryError('CPU_ANALYSIS_UNCLAIMED_SIDE_EFFECT')
        duration=entry['request']['timeout_seconds']+30
        entry['dispatch_claim']=dict(controller=identity(os.getpid()),worker=file_pin(Path(__file__)),
            deadline_monotonic=time.monotonic()+duration,deadline_wall=time.time()+duration)
        state['events'][key]=entry;journal.save(state)
        become_subreaper()
        proc=subprocess.Popen([sys.executable,'-B','-m','tools.r6_cpu_analysis',str(journal.directory.resolve()),key],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.:tools/rlflow',CUDA_VISIBLE_DEVICES=''))
        ident=identity(proc.pid)
        if ident is not None:journal.bind_process(key,ident)
    while True:
        if proc is not None:proc.poll()
        entry=journal.read()['events'][key]
        workerpath=directory/'worker.json'
        if entry['process'] is None and workerpath.exists():
            worker=json.loads(workerpath.read_text())
            if worker['event_key']!=key:raise RecoveryError('CPU_ANALYSIS_WORKER_IDENTITY')
            if owns(worker['identity']):journal.bind_process(key,worker['identity'])
        action=journal.observe(key,receipt_path=directory/'receipt.json')['action']
        if action in ('ACCEPT_RECEIPT','ALREADY_CONSUMED'):
            if action=='ACCEPT_RECEIPT':journal.consume(key,directory/'receipt.json')
            r=verified_receipt(journal,key)
            return record_analysis(journal,job_key,r['outputs']['report.json'],store)
        claim=entry.get('dispatch_claim',{});deadline=claim.get('deadline_monotonic')
        valid=type(deadline) in (int,float) and math.isfinite(deadline) and claim.get('controller',{}).get('boot_id')==identity(os.getpid())['boot_id']
        reason=('MISSING_WORKER_OR_RECEIPT' if action=='RECONCILE' else
            'STOP_OR_DISPATCH_DEADLINE' if entry.get('controller_cancel') or stop_paths(journal) or not valid or time.monotonic()>=deadline else None)
        if reason:
            cancel_bounded(journal,key,reason,proc,directory_kind='cpu-analysis',status_prefix='CPU_ANALYSIS')
            raise RecoveryError('CPU_ANALYSIS_AMBIGUOUS_NO_RETRY' if action=='RECONCILE' else 'CPU_ANALYSIS_CANCELLED_NO_RETRY')
        if action not in ('WAIT_OWNED_WORKER','WAIT_OWNED_EXIT','CANCEL_OWNED'):raise RecoveryError('CPU_ANALYSIS_OBSERVATION')
        time.sleep(min(poll_seconds,max(.001,deadline-time.monotonic())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('event_key')
    args=parser.parse_args();raise SystemExit(run(Journal(args.directory.resolve()),args.event_key))
