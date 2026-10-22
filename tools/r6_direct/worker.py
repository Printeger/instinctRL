"""Single direct baseline worker, supervised by tools.r6_direct.run.

No legacy authorization or dispatcher is constructed. AppLauncher follows the
fresh resource window; scientific evaluation is the existing shared core.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

from tools.r6_direct.io import atomic
from tools.r6_direct.source import checked, validate, audit_modules
from tools.r6_direct.import_legacy import process_identity, stop_paths, pin


def namespace_host_root_uid():
    """Map host root only for the launcher's exact one-UID user namespace.

    The supervisor has already validated these root-owned volumes outside the
    namespace before spawning bwrap. Legacy callers retain uid=0 checks.
    """
    mapping=[tuple(map(int,row.split())) for row in Path('/proc/self/uid_map').read_text().splitlines()]
    if mapping==[(0,0,4294967295)]:return 0
    if mapping!=[(1000,1000,1)] or os.getuid()!=1000:raise RuntimeError('DIRECT_UNEXPECTED_USER_NAMESPACE')
    return int(Path('/proc/sys/kernel/overflowuid').read_text())


def preflight(plan):
    from training_workloads.unified_h1.direct_contract import validate_runtime_contract
    source=validate(json.loads(checked(plan['source']).read_text()))
    contract=validate_runtime_contract(json.loads(checked(plan['contract']).read_text()))
    if plan['run_id']!=contract['parameters']['experiment_id']:raise ValueError('DIRECT_RUN_ID')
    from training_workloads.unified_h1.native_volume import validate as native_validate
    from training_workloads.unified_h1.result_volume import validate as result_validate
    native=native_validate(plan['native_volume'],host_root_uid=namespace_host_root_uid())
    result=result_validate(plan['result_volume'],output_cap_bytes=contract['parameters']['output_cap_bytes'],host_root_uid=namespace_host_root_uid())
    if Path(plan['native_dir'])!=Path(native['target'])/plan['run_id']:
        raise ValueError('DIRECT_NATIVE_DESTINATION')
    if Path(plan['attempt_dir'])!=Path(result['target'])/plan['run_id']:
        raise ValueError('DIRECT_RESULT_DESTINATION')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('ISAACSIM_ASSET_ROOT')!=source['execution']['isaacsim_asset_root']:
        raise ValueError('DIRECT_RUNTIME_ENVIRONMENT')
    if stop_paths():raise RuntimeError('STOP')
    if shutil.disk_usage('/home/mint/rl_dev/instinctRL-dev').free<30*1024**3:raise RuntimeError('DIRECT_DISK_FLOOR')
    from training_workloads.unified_h1.extension_inputs import validate as extensions_validate
    extensions_validate(plan['extension_inputs'])
    return source,contract,native,result


def run(plan, input_pin):
    # Durable parent intent and a live exact parent are mandatory. stdin release
    # occurs only after the parent has recorded this child's start identity.
    claim=json.loads((Path(plan['output_root'])/'launch.json').read_text())
    parent=process_identity(os.getppid())
    # Bubblewrap keeps a small monitor between supervisor and worker. Verify
    # the live supervisor and its exact ancestor relationship, not a PID alone.
    expected=claim['parent'];ancestor=os.getppid();chain=[]
    while ancestor>1 and len(chain)<4:
        observed=process_identity(ancestor)
        if observed is None:break
        chain.append(observed)
        if observed==expected:break
        ancestor=int(Path(f'/proc/{ancestor}/stat').read_text().rsplit(')',1)[1].split()[1])
    if expected not in chain or claim['input']!=input_pin:raise RuntimeError('DIRECT_PARENT_IDENTITY')
    if sys.stdin.buffer.read(1)!=b'1':raise RuntimeError('DIRECT_UNRELEASED_CHILD')
    source,contract,native,result=preflight(plan)
    attempt=Path(plan['attempt_dir']); native_root=Path(plan['native_dir'])
    output=attempt/'results'/contract['mode']; gate=attempt/'prelaunch'
    output.mkdir(parents=True,exist_ok=False); gate.mkdir(exist_ok=False)
    if not native_root.is_dir() or set(p.name for p in native_root.iterdir())!={'ipc'}:
        raise RuntimeError('DIRECT_NATIVE_PRECREATION')
    if Path('/dev/shm').stat().st_ino!=(native_root/'ipc').stat().st_ino or Path('/dev/shm').stat().st_dev!=(native_root/'ipc').stat().st_dev:
        raise RuntimeError('DIRECT_PRIVATE_IPC_MOUNT')
    for name in ('tmp','xdg_cache','cuda_cache','warp_cache'):(native_root/name).mkdir()
    for name,folder in (('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('XDG_CACHE_HOME','xdg_cache'),('CUDA_CACHE_PATH','cuda_cache'),('WARP_CACHE_PATH','warp_cache')):
        os.environ[name]=str(native_root/folder)
    from training_workloads.unified_h1.write_boundary import prepare,enforce
    devices=[p for p in ('/dev/null','/dev/zero','/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools') if Path(p).exists()]
    policy=prepare([str(attempt),str(native_root),'/dev/shm'],devices)
    atomic(attempt/'write_boundary.json',policy)
    enforce(policy)
    # Existing kernel-bounded volumes contain all native and scientific writes.
    from training_workloads.scale_003.persistence import install_output_write_quota
    install_output_write_quota(attempt,result['payload_cap_bytes'])
    from training_workloads.unified_h1.robustness_prelaunch import run_gate
    from training_workloads.unified_h1.config import resource_binding_pins
    pins=resource_binding_pins()
    from tools.r6_direct.import_legacy import ROOT,OLD,NEW
    stops=[ROOT/'STOP',ROOT/'coordination/STOP',ROOT/'coordination/research/STOP',
           ROOT/'coordination/commander/STOP',ROOT/'coordination/executor/STOP',OLD/'STOP',OLD/'batch/STOP',NEW/'STOP',Path(plan['output_root'])/'STOP']
    run_gate(gate,mode=contract['mode'],attempt_id=plan['run_id'],
             authorization_sha256=input_pin['sha256'],resource_pins=pins,stop_paths=stops)
    atomic(gate/'direct_identity.json',{'execution_mode':'local-direct','input':input_pin,
        'note':'Shared resource helper labels the input digest authorization_sha256; no legacy authorization exists.'})
    from training_workloads.unified_h1.native_volume import settings
    from training_workloads.unified_h1.extension_inputs import validate as extension_validate,user_config_bytes
    settings_map=settings({'target':str(native_root)})
    for key in ('/crashreporter/dumpDir','/rtx/shaderDb/shaderCachePath','/rtx/shaderDb/driverShaderCachePath',
                '/rtx-transient/resourcemanager/localTextureCachePath','/app/extensions/registryCacheFull',
                '/app/extensions/registryCache','/structuredLog/logDirectory','/exts/omni.kit.pipapi/envPath',
                '/exts/omni.kit.registry.nucleus/cachePath'):
        Path(settings_map[key]).mkdir(exist_ok=False)
    Path(settings_map['/app/userConfigPath']).write_bytes(user_config_bytes(plan['extension_inputs']))
    asset=source['execution']['isaacsim_asset_root']
    settings_map.update({'/persistent/isaac/asset_root/default':asset,'/persistent/isaac/asset_root/cloud':asset})
    kit=' '.join('--'+k+'='+str(v).lower() if type(v)is bool else '--'+k+'='+str(v) for k,v in settings_map.items())
    kit+=' --ext-folder '+extension_validate(plan['extension_inputs'])
    kit+=' --/log/file=/proc/self/fd/1'
    atomic(attempt/'kit_settings.json',settings_map)
    app=env=backend=None; started=time.monotonic(); error=None
    from training_workloads.unified_h1.budget import ExecutionCounters
    counters=ExecutionCounters();counters.reserve_live_start(1);counters.reserve_simulator_initialization(1)
    atomic(attempt/'state.json',{'status':'INITIALIZING','identity':process_identity(os.getpid()),'counters':counters.snapshot()})
    try:
        from isaaclab.app import AppLauncher
        launcher=AppLauncher(headless=True,enable_cameras=False,fast_shutdown=False,kit_args=kit);app=launcher.app
        from training_workloads.scale_003.resources import validate_graceful_shutdown_configuration,RuntimeResourceGuard
        validate_graceful_shutdown_configuration(app,resource_pins=pins)
        import carb.settings
        actual=carb.settings.get_settings()
        for key,value in settings_map.items():
            if actual.get(key)!=value:raise RuntimeError('DIRECT_NATIVE_SETTING:'+key)
        from instinctlab.envs.unified_h1_direct_env import UnifiedH1DirectEnv,UnifiedH1DirectEnvCfg
        env=UnifiedH1DirectEnv(UnifiedH1DirectEnvCfg())
        ticks=env.h1_initialization_physics_ticks
        counters.record_initialization_ticks(ticks,contract['budgets']['initialization_ticks'])
        from training_workloads.unified_h1.backend import UnifiedH1LiveBackend
        class Guard(RuntimeResourceGuard):
            def check(self,**kw):
                if stop_paths() or (Path(plan['output_root'])/'STOP').exists():raise RuntimeError('STOP')
                if shutil.disk_usage('/home/mint/rl_dev/instinctRL-dev').free<30*1024**3:raise RuntimeError('DIRECT_DISK_FLOOR')
                return super().check(**kw)
        backend=UnifiedH1LiveBackend(run_contract=contract,attempt=attempt,app=app,environment=env,
            resource_guard=Guard(gate),counters=counters,process_started_monotonic=started)
        backend.initialization_measurement=dict(status='OBSERVED',observed_physics_ticks=ticks,
            counter_source=env.h1_initialization_tick_source,fresh_physical_reset=True)
        from training_workloads.unified_h1.robustness_contract import baseline,evaluate_baseline
        backend.initial_preflight()
        atomic(attempt/'state.json',{'status':'EVALUATING','identity':process_identity(os.getpid()),'counters':counters.snapshot()})
        evaluate_baseline(backend,parameters=contract['parameters'],bindings=baseline(contract['parameters']))
        if counters.nontraining_transitions!=contract['budgets']['nontraining_transitions']:raise RuntimeError('DIRECT_FINAL_COUNT')
        backend.preserve('PASS_PENDING_CLOSE')
    except BaseException as exc:
        error=repr(exc)
        if backend is not None:backend.preserve('FAIL_PENDING_CLOSE')
        raise
    finally:
        atomic(attempt/'CPU_MODULES.json',audit_modules())
        atomic(attempt/'state.json',{'status':'FAIL_PENDING_CLOSE' if error else 'PASS_PENDING_CLOSE',
            'error':error,'counters':counters.snapshot(),'identity':process_identity(os.getpid())})
        # Outer runner independently limits cleanup time and reaps descendants.
        if env is not None:env.close()
        if app is not None:app.close(wait_for_replicator=False,skip_cleanup=True)
    atomic(attempt/'state.json',{'status':'COMPLETE','counters':counters.snapshot(),'identity':process_identity(os.getpid())})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--check',action='store_true')
    args=parser.parse_args(); input_pin=pin(args.inputs); plan=json.loads(args.inputs.read_text())
    if args.check:
        preflight(plan);print('PASS_CPU_DIRECT_PREFLIGHT_NO_GPU');return
    run(plan,input_pin)


if __name__=='__main__':main()
