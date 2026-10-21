"""Controller-owned video validation with bounded CPU worker recovery and cleanup."""
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

from tools import r6_profile_preparation as profile
from tools import validate_r6_videos as extractor
from tools.r6_decision_packet import persist_exact
from rlflow.store import canonical

ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path('/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11')
LOG_CAP=1024**2
RESERVATION=18*1024**2
ROLE='cpu-video-validation'
SOURCES=('tools/r6_video_validation.py','tools/validate_r6_videos.py','tools/r6_video_presentation.py',
 'tools/r6_video_trajectory.py','tools/r6_postrun.py')


def require(ok,reason):
    if not ok:raise RecoveryError('CPU_VIDEO_VALIDATION_'+reason)


def read(pin):return json.loads(checked(pin).read_text())


def describe(journal,controller_key,store):
    from tools.r6_job_inventory import record_inventory
    from tools import r6_postrun as postrun, r6_render_job as provenance
    profile.guard(journal);state=journal.read();controller=state['events'][controller_key]
    require(controller['identity']['role']=='r6-controller' and controller['request']['store_root']==str(store.root.resolve()),'CONTROLLER')
    cursor=controller['cursor'];outcome=cursor['render_job']
    require(outcome['status']=='RENDER_JOB_CLOSED_REQUIRES_VIDEO_VALIDATION','CLOSED_RENDER')
    key=outcome['event_key'];job=journal.canonical_job(key)
    require(job['purpose']=='confirmation_delivery' and job['closure']==outcome['closure'] and not job.get('inventory_violations'),'JOB')
    spec=read(job['request']['spec']);auth=read(spec['pins']['authorization']);p=auth['parameters']
    require(spec['h1']['mode']==auth['mode']=='research-r6-video-replay'
            and spec['h1']['attempt']==auth['attempt'],'MODE')
    provenance.validate(journal,p,'confirmation_delivery')
    material=read(cursor['render_materialization'])
    require(material['spec']==job['request']['spec'] and material['authorization']==spec['pins']['authorization'],'EXACT_PLAN')
    inventory=record_inventory(journal,key,store)
    require(inventory['output_inventory_complete']is True,'INVENTORY')
    summary=read(job['postrun'])
    require(summary['job_event_key']==key and summary['closure']==job['closure'] and summary['inventory']==job['inventory']
            and summary['postrun_complete']is True and summary['postrun_queries_used']==2 and len(summary['receipts'])==2,'POSTRUN')
    for ordinal,pin in enumerate(summary['receipts']):
        matches=[(k,e) for k,e in state['events'].items() if e['kind']=='postrun' and e.get('receipt')==pin]
        require(len(matches)==1,'POSTRUN_RECEIPT')
        k,e=matches[0];postrun.validate(journal,k)
        require(e['request']['job_event_key']==key and e['request']['ordinal']==ordinal,'POSTRUN_ORDER')
        postrun.verified_receipt(journal,k)
    root=Path(auth['attempt'])/'results'/auth['mode']
    return dict(render_manifest=file_pin(root/'media/RUN_MANIFEST.json'),extraction=p['extraction'],
        renderer=p['implementation']['renderer'],source_snapshot=p['source_snapshot'],
        inventory=job['inventory'],closure=job['closure'],postrun=job['postrun'],spec=job['request']['spec'])


def validate(journal,key):
    state=journal.read();entry=state['events'][key];q=entry['request']
    require(set(q)=={'controller_key','inputs','python','program','sources','timeout_seconds'},'REQUEST')
    expected=dict(goal_id=state['goal_id'],event_id='CPU_VIDEO_VALIDATION:'+q['controller_key'],job_id='FINAL-VIDEO-VALIDATION',input_sha256=sha(q),role=ROLE)
    require(entry['identity']==expected and sha(expected)==key,'IDENTITY')
    profile.reservation(state,key,'cpu',{'output_actual_plus_unreleased_reservations_bytes':RESERVATION},'confirmation_delivery')
    require(q['sources']==[file_pin(ROOT/p) for p in SOURCES],'SOURCE_DRIFT')
    require(type(q['timeout_seconds'])is int and 1<=q['timeout_seconds']<=3600,'TIMEOUT')
    for field in ('python','program'):checked(q[field])
    if state['goal_id']=='R6-CPU-TEST-FIXTURE':
        require(q['program']==file_pin(extractor.__file__) or checked(q['program']).is_relative_to(journal.directory.parent.resolve()),'FIXTURE_PROGRAM')
    else:require(q['python']==file_pin(PYTHON) and q['program']==file_pin(extractor.__file__),'PROGRAM')
    pin=file_pin(journal.directory/'video-validation-inputs'/key/'inputs.json')
    require(read(pin)==q['inputs'],'INPUT_DRIFT')
    if q['program']==file_pin(extractor.__file__):extractor.inputs(pin)
    else:
        for v in q['inputs'].values():
            if isinstance(v,dict):checked(v)
    return entry,[q['python']['path'],'-B',q['program']['path'],pin['path'],'--sha256',pin['sha256'],str(journal.directory/'cpu-video-validation'/key/'result')]


def register(journal,controller_key,store,*,free_disk_bytes,python=None,program=None,timeout_seconds=1800):
    profile.guard(journal);q=describe(journal,controller_key,store)
    require(len(canonical(q))<=256*1024,'INPUT_CAP');state=journal.read()
    request=dict(controller_key=controller_key,inputs=q,python=file_pin(PYTHON) if python is None else python,
        program=file_pin(extractor.__file__) if program is None else program,sources=[file_pin(ROOT/p) for p in SOURCES],timeout_seconds=timeout_seconds)
    name='CPU_VIDEO_VALIDATION:'+controller_key
    prior=[e for e in state['events'].values() if e['identity'].get('event_id')==name]
    require(len(prior)<=1 and all(e['request']==request for e in prior),'ALREADY_BOUND')
    event=dict(goal_id=state['goal_id'],event_id=name,job_id='FINAL-VIDEO-VALIDATION',input_sha256=sha(request),role=ROLE)
    key,_=journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':RESERVATION},purpose='confirmation_delivery',free_disk_bytes=free_disk_bytes)
    persist_exact(journal.directory/'video-validation-inputs'/key/'inputs.json',canonical(q)+b'\n')
    validate(journal,key);return key

def environment():
    return dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
        PYTHONPATH='.:tools/rlflow:isaaclab-training',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')


def run(journal,key):
    entry,argv=validate(journal,key);request=entry['request'];admissible(journal.read()['budget'])
    if not entry.get('dispatch_claim') or checked(entry['dispatch_claim']['worker'])!=Path(__file__).resolve():
        raise RecoveryError('CPU_VIDEO_VALIDATION_DISPATCH_REQUIRED')
    directory=journal.directory/'cpu-video-validation'/key;directory.mkdir(parents=True,exist_ok=False)
    worker=identity(os.getpid());write(directory/'worker.json',dict(identity=worker,event_key=key),LOG_CAP)
    write(directory/'request.json',dict(request=request,argv=argv),LOG_CAP)
    proc=None;child=None;error=None;code=-1;cleaned=None;outputs={}
    start=time.monotonic();interrupted=[]
    previous={s:signal.signal(s,lambda signum,frame:interrupted.append(signum)) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        if stop_paths(journal):raise RecoveryError('STOP_BEFORE_CPU_VIDEO_VALIDATION')
        become_subreaper()
        with (directory/'stdout').open('xb') as stdout,(directory/'stderr').open('xb') as stderr,selectors.DefaultSelector() as selector:
            proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                start_new_session=True,cwd=ROOT,env=environment())
            child=identity(proc.pid)
            if child is None:raise RecoveryError('CPU_VIDEO_VALIDATION_CHILD_EXIT_BEFORE_IDENTITY')
            write(directory/'child.json',dict(identity=child,event_key=key),LOG_CAP)
            selector.register(proc.stdout,selectors.EVENT_READ,stdout);selector.register(proc.stderr,selectors.EVENT_READ,stderr)
            while selector.get_map() or proc.poll() is None:
                if interrupted or stop_paths(journal):raise RecoveryError('STOP_DURING_CPU_VIDEO_VALIDATION')
                if time.monotonic()-start>request['timeout_seconds']:raise RecoveryError('CPU_VIDEO_VALIDATION_TIMEOUT')
                for ready,_ in selector.select(.1):
                    data=os.read(ready.fileobj.fileno(),65536)
                    if not data:selector.unregister(ready.fileobj);ready.fileobj.close();continue
                    if output_size(directory)+len(data)>LOG_CAP-65536:raise RecoveryError('CPU_VIDEO_VALIDATION_LOG_CAP_BEFORE_WRITE')
                    ready.data.write(data);ready.data.flush()
            proc.wait();code=proc.returncode
            if code!=0:raise RecoveryError('CPU_VIDEO_VALIDATION_NONZERO_EXIT')
    except BaseException as exc:error=repr(exc)
    finally:
        if proc is not None:
            try:cleaned=cleanup(proc);code=proc.returncode
            except BaseException as exc:error=str(error)+'; '+repr(exc)
        for sig,handler in previous.items():signal.signal(sig,handler)
    try:
        for name,limit in (('result.json',256*1024),):
            path=directory/'result'/name
            if not path.is_file() or path.is_symlink() or path.stat().st_size>limit:raise RecoveryError('CPU_VIDEO_VALIDATION_RESULT_MISSING_OR_BOUND')
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
    entry,_=validate(journal,key);r=read(entry['receipt'])
    require(r.get('event_key')==key and r.get('request_sha256')==sha(entry['request']) and r.get('job_id')=='FINAL-VIDEO-VALIDATION','RECEIPT')
    require(r.get('exit_code')==0 and r.get('error')is None and cleanup_verified(r.get('cleanup',{})),'FAILED_PRESERVED')
    require(not any(owns(i) for i in [r.get('process'),r.get('child_identity'),*r['cleanup'].get('observed_identities',[])]),'PROCESS_ALIVE')
    for name in ('stdout','stderr'):checked(r[name])
    directory=journal.directory/'cpu-video-validation'/key/'result'
    path=checked(r['outputs']['result.json']);require(path==directory/'result.json' and path.stat().st_size<=256*1024,'RESULT_PATH')
    result=read(r['outputs']['result.json'])
    require(result['status']=='VIDEOS_CPU_VALIDATED_REQUIRES_VISUAL_AUDIT' and result['inputs']==file_pin(journal.directory/'video-validation-inputs'/key/'inputs.json')
        and result['scientific_acceptance']is False and result['dispatch_authorized']is False,'RESULT')
    expected={'result.json','VIDEO_MANIFEST.json','watch.html','README.txt'}
    if entry['request']['program']==file_pin(extractor.__file__):
        manifest=read(result['manifest'])
        require(len(manifest['visual_samples'])==result['video_count'],'VISUAL_COUNT')
        for i,sample in enumerate(manifest['visual_samples']):
            require(sample['index']==i and checked(sample['image'])==directory/f'visual-{i:02d}.jpg','VISUAL_PATH')
            expected.add(f'visual-{i:02d}.jpg')
    require(set(p.name for p in directory.iterdir())==expected and output_size(directory)<=extractor.CAP,'OUTPUT_CAP')
    for field,name in (('manifest','VIDEO_MANIFEST.json'),('viewer','watch.html'),('instructions','README.txt')):require(checked(result[field])==directory/name,'ARTIFACT_PATH')
    if entry['request']['program']==file_pin(extractor.__file__):
        require(result['validator']==file_pin(extractor.__file__) and result['presentation']==file_pin(extractor.presentation.__file__),'VALIDATOR')
    return r


def handoff(journal,key,store,*,poll_seconds=1):
    journal.require_writer()
    if type(poll_seconds) not in (int,float) or not 0<poll_seconds<=60:raise RecoveryError('CPU_VIDEO_VALIDATION_POLL')
    entry,_=validate(journal,key)
    state=journal.read()
    if state.get('current_job') is not None or state.get('current_model_event') is not None:raise RecoveryError('CPU_VIDEO_VALIDATION_NOT_IDLE')
    directory=journal.directory/'cpu-video-validation'/key;proc=None
    if not entry.get('receipt') and not entry.get('dispatch_claim'):
        journal.require_dispatch_budget()
        if stop_paths(journal):raise RecoveryError('STOP_BEFORE_CPU_VIDEO_VALIDATION')
        from tools.r6_job_handoff import no_model_workers
        no_model_workers(journal)
        for other_key,e in state['events'].items():
            if e['kind']=='cpu' and other_key!=key and ((e.get('dispatch_claim') and not e.get('receipt'))
                    or any(not c.get('receipt') for c in e.get('sample_claims',{}).values())):
                raise RecoveryError('CPU_VIDEO_VALIDATION_OTHER_EVENT_UNRESOLVED')
        if directory.exists() or entry.get('process') or entry['status']!='INTENT':raise RecoveryError('CPU_VIDEO_VALIDATION_UNCLAIMED_SIDE_EFFECT')
        duration=entry['request']['timeout_seconds']+30
        entry['dispatch_claim']=dict(controller=identity(os.getpid()),worker=file_pin(Path(__file__)),
            deadline_monotonic=time.monotonic()+duration,deadline_wall=time.time()+duration)
        state['events'][key]=entry;journal.save(state)
        become_subreaper()
        proc=subprocess.Popen([sys.executable,'-B','-m','tools.r6_video_validation',str(journal.directory.resolve()),key],
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
            if worker['event_key']!=key:raise RecoveryError('CPU_VIDEO_VALIDATION_WORKER_IDENTITY')
            if owns(worker['identity']):journal.bind_process(key,worker['identity'])
        action=journal.observe(key,receipt_path=directory/'receipt.json')['action']
        if action in ('ACCEPT_RECEIPT','ALREADY_CONSUMED'):
            if action=='ACCEPT_RECEIPT':journal.consume(key,directory/'receipt.json')
            r=verified_receipt(journal,key)
            return dict(status='VIDEOS_CPU_VALIDATED_REQUIRES_VISUAL_AUDIT',event_key=key,receipt=journal.read()['events'][key]['receipt'],
                result=r['outputs']['result.json'],scientific_acceptance=False,dispatch_authorized=False)
        claim=entry.get('dispatch_claim',{});deadline=claim.get('deadline_monotonic')
        valid=type(deadline) in (int,float) and math.isfinite(deadline) and claim.get('controller',{}).get('boot_id')==identity(os.getpid())['boot_id']
        reason=('MISSING_WORKER_OR_RECEIPT' if action=='RECONCILE' else
            'STOP_OR_DISPATCH_DEADLINE' if entry.get('controller_cancel') or stop_paths(journal) or not valid or time.monotonic()>=deadline else None)
        if reason:
            cancel_bounded(journal,key,reason,proc,directory_kind='cpu-video-validation',status_prefix='CPU_VIDEO_VALIDATION')
            raise RecoveryError('CPU_VIDEO_VALIDATION_AMBIGUOUS_NO_RETRY' if action=='RECONCILE' else 'CPU_VIDEO_VALIDATION_CANCELLED_NO_RETRY')
        if action not in ('WAIT_OWNED_WORKER','WAIT_OWNED_EXIT','CANCEL_OWNED'):raise RecoveryError('CPU_VIDEO_VALIDATION_OBSERVATION')
        time.sleep(min(poll_seconds,max(.001,deadline-time.monotonic())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('event_key')
    args=parser.parse_args();raise SystemExit(run(Journal(args.directory.resolve()),args.event_key))
