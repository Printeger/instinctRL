"""One bounded cuInit trace; no simulator, policy step, optimizer or model call."""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import signal
import subprocess
import hashlib
import time
from tools.r6_direct.import_legacy import NEW,OLD,pin,process_identity,stop_paths
from tools.r6_direct.io import atomic
from tools.r6_direct.source import checked
from tools.r6_direct.run import reserve,supervise


def prelaunch(directory, gate, plan, plan_pin, run_gate):
    """Close a failed admission attempt even when no CUDA child was created."""
    from training_workloads.unified_h1.config import resource_binding_pins
    started=time.monotonic()
    try:
        return run_gate(gate,mode='research-r6-evaluation',attempt_id=plan['run_id'],
                        authorization_sha256=plan_pin['sha256'],
                        resource_pins=resource_binding_pins(),
                        stop_paths=[NEW/'STOP',directory/'STOP'])
    except BaseException as error:
        atomic(directory/'receipt.json',dict(status='CLOSED',phase='PRELAUNCH',
            parent=process_identity(os.getpid()),owned_processes=[],remaining=[],child=None,
            exit_code=1,error=repr(error),wall_seconds=time.monotonic()-started,
            cuda_child_started=False,training_transitions=0,optimizer_updates=0))
        raise


def child(plan):
    if sys.stdin.buffer.read(1)!=b'1':raise RuntimeError('UNRELEASED')
    native=Path(plan['native_dir'])
    if Path('/dev/shm').stat().st_ino!=(native/'ipc').stat().st_ino:raise RuntimeError('PRIVATE_IPC_REQUIRED')
    from training_workloads.unified_h1.write_boundary import prepare,enforce
    devices=[p for p in ('/dev/null','/dev/zero','/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools') if Path(p).exists()]
    policy=prepare([str(native),'/dev/shm'],devices)
    # CUDA names freshly created threads through task/<tid>/comm. Bind only
    # this process's task subtree: no global /proc write or disk permission.
    if plan.get('self_task_metadata') is True:
        policy['entries'].append((f'/proc/{os.getpid()}/task',2|(1<<14)))
    enforce(policy)
    if 'probability_replay' in plan:
        from tools.r6_direct.probability_replay import replay
        result=replay(plan['probability_replay'])
        atomic(native/'probability_result.json',result)
        if result['result']['status']!='PASS':raise RuntimeError('PROBABILITY_REPLAY_FAILED_SEE_RESULT')
        return
    library=ctypes.CDLL('libcuda.so.1');result=library.cuInit(0)
    name=ctypes.c_char_p();library.cuGetErrorName(result,ctypes.byref(name))
    atomic(native/'cuda_result.json',dict(cuInit=result,error_name=None if name.value is None else name.value.decode(),
        policy_transitions=0,simulator_initializations=0,optimizer_updates=0,execution_mode='local-direct-diagnostic'))
    print('cuInit',result,name.value,flush=True)
    sys.exit(0 if result==0 else 1)


def main():
    from tools.r6_direct.dependencies import install
    install()
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--child',action='store_true');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    if args.child:child(plan);return
    if pin(__file__)!=plan['script']:raise RuntimeError('DIAGNOSTIC_SOURCE_DRIFT')
    for key in ('strace','bwrap','python','predecessor_receipt'):checked(plan[key])
    snapshot=Path(plan['snapshot']);source=json.loads(checked(plan['helper_source']).read_text())
    if subprocess.check_output(['git','-C',str(snapshot),'rev-parse','HEAD'],text=True).strip()!=source['source_commit']:raise RuntimeError('HELPER_COMMIT_DRIFT')
    for name,digest in source['sources'].items():
        if hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('HELPER_SOURCE_DRIFT:'+name)
    from tools.r6_direct.source import audit_modules
    audit_modules()
    receipt=json.loads(checked(plan['predecessor_receipt']).read_text())
    if receipt['status']!='CLOSED' or receipt['remaining']:raise RuntimeError('PRIOR_CLEANUP')
    for identity in [receipt['parent'],*receipt['owned_processes']]:
        if process_identity(identity['pid'])==identity:raise RuntimeError('PRIOR_PROCESS_ALIVE')
    from tools.r6_direct.lifecycle import writer,imported_budget
    with writer():
        if stop_paths():raise RuntimeError('STOP')
        directory=Path(plan['output_root']);native=Path(plan['native_dir'])
        if directory.exists() or native.exists():raise FileExistsError('DIAGNOSTIC_ALREADY_STARTED')
        ledger_path=NEW/'RESOURCE_LEDGER.json';ledger=json.loads(ledger_path.read_text())
        imported_budget(ledger)
        reserve(ledger,plan['run_id'],dict(allocated_gpu_seconds=240,live_wall_seconds=240,live_starts=1,queue_seconds=180),resource_policy=plan.get('resource_policy'))
        ledger['new_reservations'][-1]['output_within_existing_native_volume']=plan['native_volume']
        from training_workloads.unified_h1.native_volume import validate
        binding=validate(plan['native_volume'])
        if native.parent!=Path(binding['target']):raise RuntimeError('NATIVE_BINDING')
        if args.check:
            print('CPU_DIAGNOSTIC_INPUTS_PASS');return
        ledger['status']='RUNNING_DIAGNOSTIC'
        atomic(ledger_path,ledger)
        directory.mkdir();native.mkdir();(native/'ipc').mkdir();(native/'tmp').mkdir();(native/'cache').mkdir()
        from tools.r6_direct.admission import run_gate
        gate=native/'prelaunch';gate.mkdir()
        atomic(directory/'intent.json',dict(plan=plan,parent=process_identity(os.getpid())))
        try:
            prelaunch(directory,gate,plan,pin(args.plan),run_gate)
        except BaseException:
            ledger['new_reservations'][-1]['receipt']=pin(directory/'receipt.json')
            ledger['status']='WAITING_FOR_ANALYSIS';atomic(ledger_path,ledger)
            raise
        env={k:v for k,v in os.environ.items() if k in ('HOME','PATH','LANG','LC_ALL')}
        env['PYTHONPATH']=str(snapshot)+':'+str(snapshot/'isaaclab-training')
        env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
            TMPDIR=str(native/'tmp'),CUDA_CACHE_PATH=str(native/'cache'))
        argv=[str(checked(plan['bwrap'])),'--die-with-parent','--bind','/','/','--dev-bind','/dev','/dev',
            '--bind',str(native/'ipc'),'/dev/shm','--',str(checked(plan['strace'])),'-f','-qq','-s','256',
            '-e','trace=%file,ioctl,prctl,mmap,mprotect','-o',str(native/'syscalls.log'),
            plan['python']['path'],'-B',plan['script']['path'],'--plan',str(args.plan.resolve()),'--child']
        atomic(directory/'RUN_INPUTS.json',dict(plan=plan,argv=argv,environment=env))
        resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
        result=supervise(argv,cwd=plan['snapshot'],env=env,directory=directory,deadline_seconds=30,
            stop=lambda:bool(stop_paths()) or (directory/'STOP').exists())
        ledger['new_reservations'][-1]['receipt']=pin(directory/'receipt.json');ledger['status']='WAITING_FOR_ANALYSIS';atomic(ledger_path,ledger)
        sys.exit(0 if result['exit_code']==0 and not result['error'] else 1)


if __name__=='__main__':
    def interrupted(signum, frame):raise KeyboardInterrupt('DIRECT_SIGNAL:'+str(signum))
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,interrupted)
    main()
