"""Single serial direct-run supervisor. No model calls and no legacy dispatcher."""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from tools.r6_direct.import_legacy import ROOT,OLD,NEW,pin,process_identity,stop_paths
from tools.r6_direct.io import atomic

LOG_CAP=64*1024**2


def reserve(ledger, run_id, resources):
    if any(r['id']==run_id for r in ledger['new_reservations']):raise ValueError('DIRECT_DUPLICATE_RESERVATION')
    totals={k:ledger['historical_charged'][k]+ledger['new_used'][k] for k in ledger['caps']}
    for row in ledger['new_reservations']:
        for k,v in row['resources'].items():totals[k]+=v
    for k,v in resources.items():
        if k not in totals or type(v) not in (int,float) or v<0:raise ValueError('DIRECT_RESOURCE_AMOUNT')
        totals[k]+=v
    for k,total in totals.items():
        if total>ledger['caps'][k]-ledger['protected_confirmation_delivery_repair'].get(k,0):
            raise ValueError('DIRECT_PROTECTED_BUDGET:'+k)
    ledger['new_reservations'].append(dict(id=run_id,purpose='development',resources=resources,status='UNRELEASED'))
    return ledger


def descendants(parent):
    rows={}
    for item in Path('/proc').iterdir():
        if not item.name.isdecimal():continue
        try:
            fields=(item/'stat').read_text().rsplit(')',1)[1].split()
            rows[int(item.name)]=(int(fields[1]),fields[0])
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    owned={parent}; changed=True
    while changed:
        before=len(owned);owned.update(pid for pid,(ppid,_) in rows.items() if ppid in owned);changed=len(owned)!=before
    return [i for pid in owned-{parent} if (i:=process_identity(pid)) is not None]


def terminate_owned(identities, sig):
    signaled=[]
    for identity in identities:
        if process_identity(identity['pid'])!=identity:continue
        try:
            fd=os.pidfd_open(identity['pid'])
            try:
                if process_identity(identity['pid'])==identity:
                    signal.pidfd_send_signal(fd,sig);signaled.append(identity)
            finally:os.close(fd)
        except ProcessLookupError:pass
    return signaled


def supervise(argv, *, cwd, env, directory, deadline_seconds, stop, before_release=None, workload_state=None):
    """CPU-testable exact-child runner with subreaper cleanup and bounded logs."""
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        raise RuntimeError('DIRECT_SUPERVISOR_REQUIRES_PIDFD_USE_SYSTEM_PYTHON')
    if descendants(os.getpid()):raise RuntimeError('DIRECT_SUPERVISOR_HAS_OTHER_CHILDREN')
    if any((Path(directory)/name).exists() for name in ('stdout.log','stderr.log','receipt.json','state.json')):
        raise FileExistsError('DIRECT_RUN_EVIDENCE_EXISTS')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'subreaper')
    directory=Path(directory); selector=selectors.DefaultSelector(); streams={}; sizes={}
    parent=process_identity(os.getpid()); child=None; error=None; seen={}; started=time.monotonic(); closing_since=None
    try:
        for name in ('stdout','stderr'):
            streams[name]=(directory/(name+'.log')).open('xb');sizes[name]=0
        child=subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        identity=process_identity(child.pid)
        if identity is None:raise RuntimeError('DIRECT_CHILD_DIED_BEFORE_RELEASE')
        seen[(identity['pid'],identity['starttime'])]=identity
        atomic(directory/'state.json',dict(status='RUNNING',parent=parent,child=identity,argv=argv,started_unix=time.time()))
        if before_release is not None:before_release()
        if stop():raise RuntimeError('STOP')
        child.stdin.write(b'1');child.stdin.close()
        for pipe,name in ((child.stdout,'stdout'),(child.stderr,'stderr')):
            os.set_blocking(pipe.fileno(),False);selector.register(pipe,selectors.EVENT_READ,name)
        while True:
            for item in descendants(os.getpid()):seen[(item['pid'],item['starttime'])]=item
            if stop():raise RuntimeError('STOP')
            if workload_state is not None and Path(workload_state).exists():
                state=json.loads(Path(workload_state).read_text())
                if state.get('status') in ('PASS_PENDING_CLOSE','FAIL_PENDING_CLOSE'):
                    if closing_since is None:closing_since=time.monotonic()
                    if time.monotonic()-closing_since>60:raise TimeoutError('DIRECT_CLEANUP_TIMEOUT')
            if time.monotonic()-started>deadline_seconds:raise TimeoutError('DIRECT_WALL_TIMEOUT')
            for key,_ in selector.select(.25):
                data=os.read(key.fileobj.fileno(),65536)
                if not data:selector.unregister(key.fileobj);continue
                name=key.data; remaining=LOG_CAP-sizes[name]
                streams[name].write(data[:remaining]);streams[name].flush();sizes[name]+=min(len(data),remaining)
                if len(data)>remaining:raise RuntimeError('DIRECT_LOG_CAP')
            if child.poll() is not None and not selector.get_map():break
    except BaseException as exc:
        error=repr(exc)
    finally:
        if child is not None:
            # Every adopted descendant of this dedicated subreaper belongs to
            # this one run. Identity+pidfd prevents signaling reused PIDs.
            for item in descendants(os.getpid()):seen[(item['pid'],item['starttime'])]=item
            terminate_owned(list(seen.values()),signal.SIGTERM)
            end=time.monotonic()+3
            while descendants(os.getpid()) and time.monotonic()<end:time.sleep(.05)
            for item in descendants(os.getpid()):seen[(item['pid'],item['starttime'])]=item
            terminate_owned(list(seen.values()),signal.SIGKILL)
            child.wait(timeout=10)
            end=time.monotonic()+5
            while time.monotonic()<end:
                try:
                    pid,_=os.waitpid(-1,os.WNOHANG)
                    if pid==0:time.sleep(.05)
                except ChildProcessError:break
            remaining=descendants(os.getpid())
        else:remaining=[]
        selector.close()
        if child is not None:
            for pipe in (child.stdin,child.stdout,child.stderr):
                if pipe is not None and not pipe.closed:pipe.close()
        for stream in streams.values():stream.flush();os.fsync(stream.fileno());stream.close()
        receipt=dict(status='CLOSED' if not remaining else 'CLEANUP_INCOMPLETE',error=error,
            exit_code=None if child is None else child.returncode,parent=parent,
            child=None if child is None else identity,owned_processes=list(seen.values()),remaining=remaining,
            wall_seconds=time.monotonic()-started,log_bytes=sizes)
        atomic(directory/'receipt.json',receipt);atomic(directory/'state.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);args=parser.parse_args()
    def interrupted(signum, frame):raise KeyboardInterrupt('DIRECT_SIGNAL:'+str(signum))
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,interrupted)
    plan=json.loads(args.inputs.read_text()); input_pin=pin(args.inputs)
    if stop_paths():raise RuntimeError('STOP')
    run_id=plan['run_id'];directory=Path(plan['output_root'])
    if directory!=NEW/'runs'/run_id or '/' in run_id or not run_id.startswith('R6-DIRECT-'):raise ValueError('DIRECT_OUTPUT_ROOT')
    snapshot=Path(plan['snapshot'])
    source=json.loads(Path(plan['source']['path']).read_text())
    if pin(plan['source']['path'])!=plan['source'] or pin(plan['contract']['path'])!=plan['contract']:raise ValueError('DIRECT_INPUT_HASH')
    if subprocess.check_output(['git','-C',str(snapshot),'rev-parse','HEAD'],text=True).strip()!=source['source_commit']:raise ValueError('DIRECT_SNAPSHOT')
    contract=json.loads(Path(plan['contract']['path']).read_text())
    executable=source['execution']['python_path']
    from tools.r6_direct.source import checked
    wrapper=str(checked(plan['ipc_wrapper']))
    argv=[executable,'-B','-m','tools.r6_direct.worker','--inputs',str(args.inputs.resolve())]
    env={k:v for k,v in os.environ.items() if k in ('HOME','PATH','LANG','LC_ALL','TZ','DISPLAY')}
    env.update(PYTHONPATH=str(snapshot)+':'+str(snapshot/'isaaclab-training'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='0',
               ISAACSIM_ASSET_ROOT=source['execution']['isaacsim_asset_root'],OMNI_KIT_ACCEPT_EULA='YES')
    with (OLD/'batch/writer.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        legacy=json.loads((OLD/'batch/state.json').read_text());old=legacy.get('controller_identity')
        if legacy['current_job'] or legacy['current_model_event']:raise RuntimeError('DIRECT_OLD_JOB_ACTIVE')
        if old and (actual:=process_identity(old['pid'])) and all(old[k]==v for k,v in actual.items()):raise RuntimeError('DIRECT_OLD_CONTROLLER_ACTIVE')
        imported=json.loads((NEW/'LEGACY_BUDGET_IMPORT.json').read_text())
        if pin(OLD/'batch/state.json')!=imported['sources'][0]:raise RuntimeError('DIRECT_LEGACY_LEDGER_DRIFT')
        ledger_path=NEW/'RESOURCE_LEDGER.json';ledger=json.loads(ledger_path.read_text())
        # The existing unused bounded physical volumes stay fully charged in
        # the import. Binding a new subdirectory does not release any bytes.
        storage=[r for r in imported['reservation_mapping'] if r['original']['identity']['event_id']=='STORAGE:J-d737390980235144']
        if len(storage)!=1:raise RuntimeError('DIRECT_STORAGE_IMPORT')
        prior=[r for r in ledger['new_reservations'] if r.get('legacy_storage_event')==storage[0]['event_key']]
        if prior:
            predecessor=plan.get('predecessor_receipt')
            if not predecessor or prior[-1].get('receipt')!=predecessor:raise RuntimeError('DIRECT_STORAGE_PREDECESSOR')
            for row in prior:
                receipt=json.loads(checked(row['receipt']).read_text())
                if receipt['status']!='CLOSED' or receipt['remaining']:raise RuntimeError('DIRECT_PRIOR_CLEANUP')
                for owner in [receipt['parent'],*receipt['owned_processes']]:
                    if process_identity(owner['pid'])==owner:raise RuntimeError('DIRECT_PRIOR_STILL_LIVE')
        if contract['parameters']['output_cap_bytes']!=storage[0]['charged']['output_actual_plus_unreleased_reservations_bytes']:raise ValueError('DIRECT_STORAGE_CAP')
        subprocess.run(argv+['--check'],cwd=snapshot,env=env,check=True,timeout=180)
        for key in ('attempt_dir','native_dir'):
            if Path(plan[key]).exists():raise FileExistsError(plan[key])
        directory.mkdir(parents=True,exist_ok=False)
        resources=dict(live_starts=1,simulator_initializations=1,nontraining_transitions=contract['budgets']['nontraining_transitions'],
            training_transitions=0,allocated_gpu_seconds=3790,live_wall_seconds=3790,queue_seconds=180)
        reserve(ledger,run_id,resources)
        ledger['new_reservations'][-1]['legacy_storage_event']=storage[0]['event_key']
        ledger['new_reservations'][-1]['legacy_storage_new_binding']={k:plan[k] for k in ('attempt_dir','native_dir')}
        ledger['status']='RUNNING';atomic(ledger_path,ledger)
        for key in ('attempt_dir','native_dir'):Path(plan[key]).mkdir(exist_ok=False)
        ipc=Path(plan['native_dir'])/'ipc';ipc.mkdir(exist_ok=False)
        argv=[wrapper,'--die-with-parent','--bind','/','/',
              '--bind',str(ipc),'/dev/shm','--',*argv]
        atomic(directory/'RUN_INPUTS.json',dict(plan=plan,input=input_pin,argv=argv,environment=env,execution_mode='local-direct'))
        atomic(directory/'launch.json',dict(parent=process_identity(os.getpid()),input=input_pin))
        result=supervise(argv,cwd=snapshot,env=env,directory=directory,deadline_seconds=3780,
            stop=lambda:bool(stop_paths()) or (directory/'STOP').exists(),
            before_release=lambda:None,workload_state=Path(plan['attempt_dir'])/'state.json')
        ledger['status']='WAITING_FOR_ANALYSIS';ledger['new_reservations'][-1]['receipt']=pin(directory/'receipt.json')
        atomic(ledger_path,ledger)
        if result['error'] or result['exit_code']!=0 or result['remaining']:sys.exit(1)


if __name__=='__main__':main()
