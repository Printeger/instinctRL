"""CPU-only exact R6 snapshot, charged before creation; never grants live access.

Reuse the detached local-clone/config-overlay approach from prepare_r5_w_snapshot.
Interrupted creation is preserved for reconciliation, never automatically retried.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from rlflow.jobs import identity, owns
from rlflow.store import atomic, canonical, sha
from tools.r6_recovery import Journal, RecoveryError, checked, file_pin
from training_workloads.unified_h1.config import R6_CONFIG_SCHEMA, R6_TRAINING_CONFIG_SCHEMA, R6_CANDIDATE_CONFIG_SCHEMA, R6_FINAL_CONFIG_SCHEMA, required_source_sets

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/experiments/r6-robustness-auto'
RESERVATION=2*1024**3


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],timeout=60)


def size(root):
    return sum(p.lstat().st_size for p in root.rglob('*') if p.is_file() or p.is_symlink())


def preflight(config_path,commit):
    config_path=Path(config_path)
    if config_path.is_symlink() or config_path.resolve()!=config_path or BASE not in config_path.parents:
        raise RecoveryError('SNAPSHOT_CONFIG_PATH')
    commit=git(ROOT,'rev-parse',commit+'^{commit}').decode().strip()
    data=git(ROOT,'show',commit+':'+str(config_path.relative_to(ROOT)))
    if config_path.read_bytes()!=data:raise RecoveryError('SNAPSHOT_CONFIG_NOT_COMMITTED')
    config=json.loads(data)
    if config['schema_version'] not in (R6_CONFIG_SCHEMA,R6_TRAINING_CONFIG_SCHEMA,R6_CANDIDATE_CONFIG_SCHEMA,R6_FINAL_CONFIG_SCHEMA):raise RecoveryError('SNAPSHOT_R6_CONFIG_REQUIRED')
    target=Path(config['workload']['path'])
    if (target.resolve()!=target or target.is_symlink() or target.parent!=ROOT/'coordination/research/workspaces'
            or not target.name.startswith('R6-') or config['product']['path']!=str(target)):
        raise RecoveryError('SNAPSHOT_TARGET_PATH')
    for section,sources in zip(('product','workload'),required_source_sets(config['schema_version'])):
        spec=config[section]
        if spec['candidate_commit']!='SELF' or spec['candidate_tree']!='SELF' or spec['branch']!='' or set(spec['source_sha256'])!=sources:
            raise RecoveryError('SNAPSHOT_SOURCE_MANIFEST')
        git(ROOT,'merge-base','--is-ancestor',spec['base_commit'],commit)
        for name,digest in spec['source_sha256'].items():
            import hashlib
            if hashlib.sha256(git(ROOT,'show',commit+':'+name)).hexdigest()!=digest:
                raise RecoveryError('SNAPSHOT_SOURCE_HASH: '+name)
    if git(ROOT,'show',commit+':tools/prepare_r6_snapshot.py')!=Path(__file__).read_bytes():
        raise RecoveryError('SNAPSHOT_TOOL_NOT_COMMITTED')
    return config,target,commit


def execution_inventory(target):
    """Every checkout entry, including ignored files and Git metadata."""
    import stat
    entries={}
    for path in [target,*sorted(target.rglob('*'))]:
        info=path.lstat();relative=str(path.relative_to(target))
        if stat.S_ISLNK(info.st_mode):raise RecoveryError('SNAPSHOT_SYMLINK_ENTRY: '+relative)
        if info.st_mode & 0o222:raise RecoveryError('SNAPSHOT_NOT_READ_ONLY')
        if stat.S_ISREG(info.st_mode):
            entries[relative]=dict(type='file',bytes=info.st_size,mode=stat.S_IMODE(info.st_mode),sha256=file_pin(path)['sha256'])
        elif stat.S_ISDIR(info.st_mode):entries[relative]=dict(type='directory',mode=stat.S_IMODE(info.st_mode))
        else:raise RecoveryError('SNAPSHOT_SPECIAL_ENTRY: '+relative)
    return entries


def verify_frozen(result,config):
    import hashlib
    target=Path(result['snapshot'])
    if 'execution_inventory' not in result:raise RecoveryError('SNAPSHOT_INVENTORY_REQUIRED')
    expected=json.loads(checked(result['execution_inventory']).read_text())
    if execution_inventory(target)!=expected:raise RecoveryError('SNAPSHOT_INVENTORY_DRIFT')
    if (target.is_symlink() or not target.is_dir() or
            any(p.stat().st_mode & 0o222 for p in [target,*target.rglob('*')] if not p.is_symlink())):
        raise RecoveryError('SNAPSHOT_NOT_READ_ONLY')
    if git(target,'rev-parse','HEAD').decode().strip()!=result['source_commit']:
        raise RecoveryError('SNAPSHOT_REPLAY_COMMIT')
    if hashlib.sha256(git(target,'diff','--binary',result['source_commit'])).hexdigest()!=result['snapshot_diff_sha256']:
        raise RecoveryError('SNAPSHOT_REPLAY_DIFF')
    for section in ('product','workload'):
        for name,digest in config[section]['source_sha256'].items():
            checked(dict(path=str(target/name),sha256=digest))
    checked(dict(path=str(target/'training_workloads/unified_h1/candidate_config.json'),sha256=result['config']['sha256']))


def prepare(journal,config_path,commit,*,purpose="development"):
    if purpose not in {"development","confirmation_delivery","repair"}:raise RecoveryError("SNAPSHOT_PURPOSE")
    journal.guard();config,target,commit=preflight(config_path,commit)
    state=journal.read()
    if state.get('current_job') or state.get('current_model_event') or any(owns(e.get('process')) for e in state['events'].values()):
        raise RecoveryError('SNAPSHOT_REQUIRES_IDLE')
    request=dict(source_commit=commit,config=file_pin(config_path),target=str(target),tool=file_pin(__file__),reserved_bytes=RESERVATION)
    if purpose!='development':request['purpose']=purpose
    event=dict(goal_id=state['goal_id'],event_id='SOURCE_SNAPSHOT:'+target.name,job_id='CPU-SNAPSHOT:'+target.name,input_sha256=sha(request),role='source-snapshot')
    key=sha(event)
    aliases=[k for k,e in state['events'].items() if e['identity']['role']=='source-snapshot' and e['request']['target']==str(target)]
    if aliases and aliases!=[key]:raise RecoveryError('SNAPSHOT_TARGET_ALREADY_BOUND')
    if key in state['events']:
        old=state['events'][key]
        if old.get('receipt'):
            result=json.loads(checked(old['receipt']).read_text())
            if result['status']!='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY':raise RecoveryError('SNAPSHOT_PRIOR_FAILURE_NO_RETRY')
            verify_frozen(result,config)
            return result
        raise RecoveryError('SNAPSHOT_INCOMPLETE_NO_RETRY')
    if target.exists():raise RecoveryError('SNAPSHOT_TARGET_EXISTS')
    # Conservative pre-copy bound: tracked checkout bytes + complete local Git
    # storage + 64MiB metadata margin; no sparse omission of execution sources.
    rows=git(ROOT,'ls-tree','-rl',commit).decode().splitlines()
    checkout_bytes=sum(int(row.split()[3]) for row in rows if row.split()[3]!='-')
    gitdir=Path(git(ROOT,'rev-parse','--absolute-git-dir').decode().strip())
    if checkout_bytes+size(gitdir)+64*1024**2>RESERVATION:raise RecoveryError('SNAPSHOT_CAPACITY_BEFORE_CLONE')
    key,_=journal.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':RESERVATION},purpose=purpose,free_disk_bytes=shutil.disk_usage(ROOT).free)
    state=journal.read();state['events'][key]['dispatch_claim']=dict(controller=identity(os.getpid()),source_commit=commit);journal.save(state)
    result=dict(event_key=key,input_sha256=sha(request),source_commit=commit,snapshot=str(target),reserved_bytes=RESERVATION,GPU_queries=0,live_starts=0,status='FAILED_PRESERVED_NO_RETRY')
    try:
        journal.guard()
        subprocess.run(['git','clone','--no-hardlinks','--no-checkout','--local',str(ROOT),str(target)],check=True,timeout=60,capture_output=True)
        journal.guard()
        git(target,'-c','core.hooksPath=/dev/null','checkout','--detach',commit)
        (target/'training_workloads/unified_h1/candidate_config.json').write_bytes(checked(request['config']).read_bytes())
        journal.guard()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(target)+':'+str(target/'isaaclab-training'))
        checked(dict(path=config['execution']['python_path'],sha256=config['execution']['python_sha256']))
        run=subprocess.run([config['execution']['python_path'],'-B','-c','import json; from training_workloads.unified_h1.config import load_config; print(json.dumps(load_config()["resolved_identity"],sort_keys=True))'],cwd=target,env=env,capture_output=True,text=True,timeout=60)
        result.update(config_stdout=run.stdout,config_stderr=run.stderr,config_exit_code=run.returncode)
        if run.returncode:raise RecoveryError('SNAPSHOT_CONFIG_CPU_FAILED')
        resolved=json.loads(run.stdout);tree=git(target,'rev-parse','HEAD^{tree}').decode().strip()
        expected={s+'_'+k:v for s in ('product','workload') for k,v in [('commit',commit),('tree',tree)]}|{'config_sha256':request['config']['sha256']}
        if resolved!=expected:raise RecoveryError('SNAPSHOT_RESOLVED_IDENTITY')
        result.update(source_identity=resolved,config=request['config'],snapshot_tree=tree)
        import hashlib
        result['snapshot_diff_sha256']=hashlib.sha256(git(target,'diff','--binary',commit)).hexdigest()
        journal.guard()
        result['observed_file_bytes']=size(target)
        if result['observed_file_bytes']>RESERVATION:raise RecoveryError('SNAPSHOT_OUTPUT_CAP')
        for p in sorted(target.rglob('*'),key=lambda p:len(p.parts),reverse=True):
            if not p.is_symlink():p.chmod(p.stat().st_mode & ~0o222)
        target.chmod(target.stat().st_mode & ~0o222)
        inventory_path=journal.directory/'snapshots'/key/'execution-inventory.json'
        inventory_bytes=canonical(execution_inventory(target))+b'\n'
        if len(inventory_bytes)>16*1024**2 or result['observed_file_bytes']+len(inventory_bytes)>RESERVATION:
            raise RecoveryError('SNAPSHOT_INVENTORY_OUTPUT_CAP')
        atomic(inventory_path,inventory_bytes)
        result['execution_inventory']=file_pin(inventory_path)
        result['inventory_bytes']=len(inventory_bytes)
        result.update(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',read_only=True)
    except Exception as error:
        result.update(error_type=type(error).__name__,error=str(error))
    if target.exists():result['observed_file_bytes']=size(target)
    output=journal.directory/'snapshots'/key/'receipt.json'
    atomic(output,canonical(result)+b'\n')
    state=journal.read();state['events'][key].update(receipt=file_pin(output),status=result['status'])
    from tools.r6_budget import observe
    state['budget']=observe(state['budget'],event_key=event,resources={'output_actual_plus_unreleased_reservations_bytes':result.get('observed_file_bytes',0)+result.get('inventory_bytes',0)},evidence=file_pin(output))
    journal.save(state)
    if result['status']!='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY':raise RecoveryError('SNAPSHOT_FAILED_PRESERVED: '+str(output))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('config',type=Path);ap.add_argument('--commit',required=True)
    args=ap.parse_args()
    journal=Journal(BASE/'batch',stop_paths=[BASE/'STOP',ROOT/'STOP',ROOT/'coordination/STOP',ROOT/'coordination/commander/STOP',ROOT/'coordination/executor/STOP'])
    with journal.writer():print(json.dumps(prepare(journal,args.config,args.commit),sort_keys=True))
