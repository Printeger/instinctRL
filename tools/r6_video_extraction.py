"""Controller-owned video extraction with bounded CPU worker recovery and cleanup."""
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
from tools.r6_model_worker import write, output_size
from tools.r6_postrun import stop_paths, cancel_bounded
from tools.r6_worker_processes import become_subreaper, cleanup
from tools.r6_budget import admissible

from tools import r6_video_selection as selection
from tools import r6_profile_preparation as profile
from tools import prepare_r6_video_trajectories as extractor
from tools.r6_decision_packet import persist_exact
from rlflow.store import canonical

ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path('/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11')
LOG_CAP=1024**2
RESERVATION=18*1024**2
ROLE='cpu-video-extraction'
SOURCES=('tools/r6_video_extraction.py','tools/prepare_r6_video_trajectories.py','tools/r6_video_trajectory.py',
 'tools/r6_video_selection.py','training_workloads/unified_h1/robustness_audit.py',
 'training_workloads/unified_h1/force_archive.py','training_workloads/unified_h1/logging.py',
 'training_workloads/unified_h1/robustness.py')


def environment():
    return dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
        PYTHONPATH='.:tools/rlflow:isaaclab-training',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')


def validate(journal,key):
    state=journal.read();entry=state['events'][key];q=entry['request']
    if set(q)!={'controller_key','selection','python','program','sources','timeout_seconds'}:
        raise RecoveryError('CPU_VIDEO_REQUEST')
    expected=dict(goal_id=state['goal_id'],event_id='CPU_VIDEO:'+q['controller_key'],job_id='FINAL-MEDIA',input_sha256=sha(q),role=ROLE)
    if entry['identity']!=expected or sha(expected)!=key:raise RecoveryError('CPU_VIDEO_IDENTITY')
    profile.reservation(state,key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':RESERVATION},'confirmation_delivery')
    if q['sources']!=[file_pin(ROOT/p) for p in SOURCES]:raise RecoveryError('CPU_VIDEO_SOURCE_DRIFT')
    if type(q['timeout_seconds'])is not int or not 1<=q['timeout_seconds']<=3600:raise RecoveryError('CPU_VIDEO_TIMEOUT_BOUND')
    for field in ('python','program'):checked(q[field])
    if state['goal_id']=='R6-CPU-TEST-FIXTURE':
        if q['program']!=file_pin(extractor.__file__) and not checked(q['program']).is_relative_to(journal.directory.parent.resolve()):
            raise RecoveryError('CPU_VIDEO_FIXTURE_PROGRAM')
    elif q['python']!=file_pin(PYTHON) or q['program']!=file_pin(extractor.__file__):raise RecoveryError('CPU_VIDEO_PROGRAM')
    p=journal.directory/'video-inputs'/key/'selection.json'
    selection_pin=file_pin(p)
    if json.loads(checked(selection_pin).read_text())!=q['selection']:raise RecoveryError('CPU_VIDEO_SELECTION_DRIFT')
    for clip in q['selection']['clips']:
        for pin in clip['raw'].values():checked(pin)
    argv=[q['python']['path'],'-B',q['program']['path'],str(p),'--sha256',selection_pin['sha256'],str(journal.directory/'cpu-video'/key/'result')]
    return entry,argv


def register(journal,controller_key,store,*,free_disk_bytes,python=None,program=None,timeout_seconds=1800):
    profile.guard(journal);state=journal.read();controller=state['events'][controller_key]
    if controller['identity']['role']!='r6-controller' or controller['request']['store_root']!=str(store.root.resolve()):
        raise RecoveryError('CPU_VIDEO_CONTROLLER')
    final=controller['cursor']['final_collection']
    if final['status']!='FINAL_BATCH_COLLECTED_REQUIRES_INDEPENDENT_AUDIT':raise RecoveryError('CPU_VIDEO_COLLECTION')
    selected=selection.prepare(journal,final['collection']['batch'],final['completed_jobs'],store)
    if len(canonical(selected))>256*1024:raise RecoveryError('CPU_VIDEO_INPUT_CAP')
    request=dict(controller_key=controller_key,selection=selected,python=file_pin(PYTHON) if python is None else python,
        program=file_pin(extractor.__file__) if program is None else program,
        sources=[file_pin(ROOT/p) for p in SOURCES],timeout_seconds=timeout_seconds)
    prior=[e for e in state['events'].values() if e['identity'].get('event_id')=='CPU_VIDEO:'+controller_key]
    if len(prior)>1 or any(e['request']!=request for e in prior):raise RecoveryError('CPU_VIDEO_ALREADY_BOUND')
    event=dict(goal_id=state['goal_id'],event_id='CPU_VIDEO:'+controller_key,job_id='FINAL-MEDIA',input_sha256=sha(request),role=ROLE)
    key,_=journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':RESERVATION},
        purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    persist_exact(journal.directory/'video-inputs'/key/'selection.json',canonical(selected)+b'\n')
    validate(journal,key)
    return key

def run(journal,key):
    entry,argv=validate(journal,key);request=entry['request'];admissible(journal.read()['budget'])
    if not entry.get('dispatch_claim') or checked(entry['dispatch_claim']['worker'])!=Path(__file__).resolve():
        raise RecoveryError('CPU_VIDEO_DISPATCH_REQUIRED')
    directory=journal.directory/'cpu-video'/key;directory.mkdir(parents=True,exist_ok=False)
    worker=identity(os.getpid());write(directory/'worker.json',dict(identity=worker,event_key=key),LOG_CAP)
    write(directory/'request.json',dict(request=request,argv=argv),LOG_CAP)
    proc=None;child=None;error=None;code=-1;cleaned=None;outputs={}
    start=time.monotonic();interrupted=[]
    previous={s:signal.signal(s,lambda signum,frame:interrupted.append(signum)) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        if stop_paths(journal):raise RecoveryError('STOP_BEFORE_CPU_VIDEO')
        become_subreaper()
        with (directory/'stdout').open('xb') as stdout,(directory/'stderr').open('xb') as stderr,selectors.DefaultSelector() as selector:
            proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                start_new_session=True,cwd=ROOT,env=environment())
            child=identity(proc.pid)
            if child is None:raise RecoveryError('CPU_VIDEO_CHILD_EXIT_BEFORE_IDENTITY')
            write(directory/'child.json',dict(identity=child,event_key=key),LOG_CAP)
            selector.register(proc.stdout,selectors.EVENT_READ,stdout);selector.register(proc.stderr,selectors.EVENT_READ,stderr)
            while selector.get_map() or proc.poll() is None:
                if interrupted or stop_paths(journal):raise RecoveryError('STOP_DURING_CPU_VIDEO')
                if time.monotonic()-start>request['timeout_seconds']:raise RecoveryError('CPU_VIDEO_TIMEOUT')
                for ready,_ in selector.select(.1):
                    data=os.read(ready.fileobj.fileno(),65536)
                    if not data:selector.unregister(ready.fileobj);ready.fileobj.close();continue
                    if output_size(directory)+len(data)>LOG_CAP-65536:raise RecoveryError('CPU_VIDEO_LOG_CAP_BEFORE_WRITE')
                    ready.data.write(data);ready.data.flush()
            proc.wait();code=proc.returncode
            if code!=0:raise RecoveryError('CPU_VIDEO_NONZERO_EXIT')
    except BaseException as exc:error=repr(exc)
    finally:
        if proc is not None:
            try:cleaned=cleanup(proc);code=proc.returncode
            except BaseException as exc:error=str(error)+'; '+repr(exc)
        for sig,handler in previous.items():signal.signal(sig,handler)
    try:
        for name,limit in (('result.json',256*1024),):
            path=directory/'result'/name
            if not path.is_file() or path.is_symlink() or path.stat().st_size>limit:raise RecoveryError('CPU_VIDEO_RESULT_MISSING_OR_BOUND')
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
    from tools.r6_worker_processes import cleanup_verified
    entry,_=validate(journal,key);r=json.loads(checked(entry['receipt']).read_text())
    if r.get('event_key')!=key or r.get('request_sha256')!=sha(entry['request']) or r.get('job_id')!='FINAL-MEDIA':raise RecoveryError('CPU_VIDEO_RECEIPT')
    if r.get('exit_code')!=0 or r.get('error') is not None or not cleanup_verified(r.get('cleanup',{})):raise RecoveryError('CPU_VIDEO_FAILED_PRESERVED')
    if any(owns(i) for i in [r.get('process'),r.get('child_identity'),*r['cleanup'].get('observed_identities',[])]):raise RecoveryError('CPU_VIDEO_PROCESS_ALIVE')
    for name in ('stdout','stderr'):checked(r[name])
    directory=journal.directory/'cpu-video'/key/'result'
    result_path=checked(r['outputs']['result.json'])
    if result_path!=directory/'result.json' or result_path.stat().st_size>256*1024:raise RecoveryError('CPU_VIDEO_RESULT_PATH')
    result=json.loads(result_path.read_text());q=entry['request'];selected=q['selection']
    if (result['status']!='RECORDED_TRAJECTORIES_EXTRACTED_NOT_RENDERED'
        or result['selection']!=file_pin(journal.directory/'video-inputs'/key/'selection.json')
        or result['physics_reintegrated']is not False or result['dispatch_authorized']is not False
        or result['scientific_acceptance']is not False or result['fps']!=10
        or result['total_frames']!=selected['total_frames'] or len(result['trajectories'])!=len(selected['clips'])):
        raise RecoveryError('CPU_VIDEO_RESULT_BINDING')
    if q['program']==file_pin(extractor.__file__):
        from tools import r6_video_trajectory
        if result['extractor']!=file_pin(extractor.__file__) or result['decoder']!=file_pin(r6_video_trajectory.__file__):raise RecoveryError('CPU_VIDEO_DECODER_BINDING')
    if output_size(directory)>extractor.CAP:raise RecoveryError('CPU_VIDEO_OUTPUT_CAP')
    for i,row in enumerate(result['trajectories']):
        path=checked(row['trajectory']);clip=selected['clips'][i]
        if path!=directory/f'clip-{i:02d}.json' or row['index']!=i or row['frame_count']!=clip['frame_count']:raise RecoveryError('CPU_VIDEO_CLIP_PATH')
        value=json.loads(path.read_text())
        if value['clip']!=clip or len(value['frames'])!=clip['frame_count'] or value['physics_reintegrated']is not False:raise RecoveryError('CPU_VIDEO_CLIP_BINDING')
    return r


def handoff(journal,key,store,*,poll_seconds=1):
    journal.require_writer()
    if type(poll_seconds) not in (int,float) or not 0<poll_seconds<=60:raise RecoveryError('CPU_VIDEO_POLL')
    entry,_=validate(journal,key)
    state=journal.read()
    if state.get('current_job') is not None or state.get('current_model_event') is not None:raise RecoveryError('CPU_VIDEO_NOT_IDLE')
    directory=journal.directory/'cpu-video'/key;proc=None
    if not entry.get('receipt') and not entry.get('dispatch_claim'):
        journal.require_dispatch_budget()
        if stop_paths(journal):raise RecoveryError('STOP_BEFORE_CPU_VIDEO')
        from tools.r6_job_handoff import no_model_workers
        no_model_workers(journal)
        for other_key,e in state['events'].items():
            if e['kind']=='cpu' and other_key!=key and ((e.get('dispatch_claim') and not e.get('receipt'))
                    or any(not c.get('receipt') for c in e.get('sample_claims',{}).values())):
                raise RecoveryError('CPU_VIDEO_OTHER_EVENT_UNRESOLVED')
        if directory.exists() or entry.get('process') or entry['status']!='INTENT':raise RecoveryError('CPU_VIDEO_UNCLAIMED_SIDE_EFFECT')
        duration=entry['request']['timeout_seconds']+30
        entry['dispatch_claim']=dict(controller=identity(os.getpid()),worker=file_pin(Path(__file__)),
            deadline_monotonic=time.monotonic()+duration,deadline_wall=time.time()+duration)
        state['events'][key]=entry;journal.save(state)
        become_subreaper()
        proc=subprocess.Popen([sys.executable,'-B','-m','tools.r6_video_extraction',str(journal.directory.resolve()),key],
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
            if worker['event_key']!=key:raise RecoveryError('CPU_VIDEO_WORKER_IDENTITY')
            if owns(worker['identity']):journal.bind_process(key,worker['identity'])
        action=journal.observe(key,receipt_path=directory/'receipt.json')['action']
        if action in ('ACCEPT_RECEIPT','ALREADY_CONSUMED'):
            if action=='ACCEPT_RECEIPT':journal.consume(key,directory/'receipt.json')
            r=verified_receipt(journal,key)
            return dict(status='VIDEO_TRAJECTORIES_READY_NOT_RENDERED',event_key=key,receipt=journal.read()['events'][key]['receipt'],
                result=r['outputs']['result.json'],scientific_acceptance=False,dispatch_authorized=False)
        claim=entry.get('dispatch_claim',{});deadline=claim.get('deadline_monotonic')
        valid=type(deadline) in (int,float) and math.isfinite(deadline) and claim.get('controller',{}).get('boot_id')==identity(os.getpid())['boot_id']
        reason=('MISSING_WORKER_OR_RECEIPT' if action=='RECONCILE' else
            'STOP_OR_DISPATCH_DEADLINE' if entry.get('controller_cancel') or stop_paths(journal) or not valid or time.monotonic()>=deadline else None)
        if reason:
            cancel_bounded(journal,key,reason,proc,directory_kind='cpu-video',status_prefix='CPU_VIDEO')
            raise RecoveryError('CPU_VIDEO_AMBIGUOUS_NO_RETRY' if action=='RECONCILE' else 'CPU_VIDEO_CANCELLED_NO_RETRY')
        if action not in ('WAIT_OWNED_WORKER','WAIT_OWNED_EXIT','CANCEL_OWNED'):raise RecoveryError('CPU_VIDEO_OBSERVATION')
        time.sleep(min(poll_seconds,max(.001,deadline-time.monotonic())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('event_key')
    args=parser.parse_args();raise SystemExit(run(Journal(args.directory.resolve()),args.event_key))
