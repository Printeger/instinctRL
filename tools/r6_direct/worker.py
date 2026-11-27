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
    if contract['operation']=='train':
        from tools.r6_direct.source import training_identity
        if contract['parameters']['source_identity']!=training_identity(source):
            raise ValueError('DIRECT_TRAINING_SOURCE_IDENTITY')
    if plan['run_id']!=contract['parameters']['experiment_id']:raise ValueError('DIRECT_RUN_ID')
    from training_workloads.unified_h1.native_volume import validate as native_validate
    from training_workloads.unified_h1.result_volume import validate as result_validate
    if plan.get('system_native') is True:
        from tools.r6_direct.storage import system_native
        native_path=system_native(plan)
        native=dict(target=str(native_path))
    else:
        native=native_validate(plan['native_volume'],host_root_uid=namespace_host_root_uid(),
            minimum_free_bytes=max(256*1024**2,plan.get('native_minimum_free_bytes',0)))
        native_path=Path(native['target'])/plan['run_id']
    if plan.get('system_results'):
        from tools.r6_direct.storage import system_result
        result=system_result(plan,contract,before_launch=not Path(plan['attempt_dir']).exists())
    else:
        result=result_validate(plan['result_volume'],output_cap_bytes=contract['parameters']['output_cap_bytes'],
        minimum_free_bytes=0 if 'result_payload_cap_bytes' in plan else 256*1024**2,
        host_root_uid=namespace_host_root_uid())
    # Generic newly provisioned volumes have no legacy fixed-layout payload.
    # Derive available payload from verified filesystem capacity, retaining
    # control headroom; never infer it from the requested backing-image size.
    if 'payload_cap_bytes' not in result:
        result=dict(result,payload_cap_bytes=max(0,result['free_bytes']-64*1024**2))
    if 'result_payload_cap_bytes' in plan and not plan.get('system_results'):
        # Reuse a nonempty prepaid volume without claiming its original empty
        # capacity. Kernel volume checks above remain, and each new attempt gets
        # a smaller enforced write budget backed by currently free capacity.
        cap=plan['result_payload_cap_bytes']
        if type(cap) is not int or not 256*1024**2<=cap<=result['payload_cap_bytes']:
            raise ValueError('DIRECT_RESULT_PAYLOAD_CAP')
        stats=os.statvfs(result['target'])
        if result['free_bytes']<cap+64*1024**2 or stats.f_favail<4096:
            raise ValueError('DIRECT_RESULT_REMAINING_CAPACITY')
        result=dict(result,payload_cap_bytes=cap)
    if contract['operation']=='train' and contract['parameters']['r6_training_plan']['storage']['payload_bytes']>result['payload_cap_bytes']:
        raise ValueError('DIRECT_TRAINING_PAYLOAD_CAPACITY')
    if Path(plan['native_dir'])!=native_path:
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
    output_roots=[str(attempt),'/dev/shm']
    if not native_root.is_relative_to(attempt):
        output_roots.append(str(native_root))
    policy=prepare(output_roots,devices)
    # Validated by CUDA-DIAGNOSTIC-02: CUDA names new threads through comm.
    # Only this process's task metadata receives write/truncate permission.
    policy['entries'].append((f'/proc/{os.getpid()}/task',2|(1<<14)))
    atomic(attempt/'write_boundary.json',policy)
    enforce(policy)
    # Keep the scientific file budget as well as the supervisor's joint disk cap.
    from training_workloads.scale_003.persistence import install_output_write_quota
    install_output_write_quota(attempt,min(result['payload_cap_bytes'],1024**3) if contract['operation']=='render' else result['payload_cap_bytes'])
    from tools.r6_direct.admission import run_gate
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
    if contract['operation']=='render':
        from tools.r6_direct.render import startup_settings
        settings_map.update(startup_settings(native_root))
        for name in ('cache','omni_global_cache'):
            Path(settings_map['/app/tokens/'+name]).mkdir(exist_ok=False)
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
    app=env=backend=resource_guard=None; started=time.monotonic(); error=None
    from training_workloads.unified_h1.budget import ExecutionCounters
    counters=ExecutionCounters();counters.reserve_live_start(1);counters.reserve_simulator_initialization(1)
    atomic(attempt/'state.json',{'status':'INITIALIZING','identity':process_identity(os.getpid()),'counters':counters.snapshot()})
    try:
        from isaaclab.app import AppLauncher
        launcher=AppLauncher(headless=True,enable_cameras=contract['operation']=='render',fast_shutdown=False,kit_args=kit);app=launcher.app
        from training_workloads.scale_003.resources import (
            JournaledRuntimeResourceGuard, RuntimeResourceGuard, validate_graceful_shutdown_configuration,
        )
        from training_workloads.unified_h1.dynamic_assay import FRONT_STUDY_PREFIX, NEXT_LONG_PREFIX, U1792_STUDY_PREFIX
        from training_workloads.unified_h1.command_replay import (
            BRAKE_EXECUTION, CONFIRMED_STANDOFF_EXECUTION, CONFIRMED_TUBE_EXECUTIONS, ENDURANCE_PREFIXES,
            EDGE_CONTINUATION_EXECUTION, FINITE_EDGE_EXECUTIONS, LONG_PREFIX, NEIGHBOR_HISTORY_EXECUTIONS,
            ALONG_INTERCEPT_EXECUTIONS, CROSSING_FALLBACK_EXECUTIONS, CROSSING_STOP_CONTINUATION_EXECUTION, CROSSING_STOP_EXECUTIONS,
            EARLY_DIAGONAL_EXECUTIONS, FINITE_INTERCEPT_EXECUTIONS,
            INTERCEPT_CHOICE_EXECUTIONS, OPENING_STEERING_EXECUTIONS, RESERVE_DIRECTION_EXECUTIONS,
            ROTATION_TUBE_EXECUTIONS, TANGENT_MAGNITUDE_EXECUTION,
            VERIFIED_RESUME_EXECUTION, VERIFIED_ZERO_EXECUTIONS,
            NOVEL_PREFIX, REGRESSION_IDS,
            SETTLED_OFFSET_STOP_EXECUTIONS, STEP_MS,
            WALL_TANGENT_EXECUTIONS,
            load_lane_streams, matches as command_replay,
            route_layouts,
        )
        from training_workloads.unified_h1.pipe_authority_probe import MIRRORED_PIPE_RUNS
        validate_graceful_shutdown_configuration(app,resource_pins=pins)
        import carb.settings
        actual=carb.settings.get_settings()
        for key,value in settings_map.items():
            if actual.get(key)!=value:raise RuntimeError('DIRECT_NATIVE_SETTING:'+key)
        if contract['operation']=='render':
            import carb.tokens
            from tools.r6_direct.render import validate_cache_tokens
            validate_cache_tokens(settings_map,carb.tokens.get_tokens_interface())
        pipe_probe = plan['run_id'] in MIRRORED_PIPE_RUNS
        replay = command_replay(plan['run_id'])
        command_assay = replay or plan['run_id'] in REGRESSION_IDS
        journaled = command_assay or plan['run_id'].startswith((NEXT_LONG_PREFIX, U1792_STUDY_PREFIX, FRONT_STUDY_PREFIX)) or pipe_probe

        class Guard(JournaledRuntimeResourceGuard if journaled else RuntimeResourceGuard):
            def check(self,**kw):
                if stop_paths() or (Path(plan['output_root'])/'STOP').exists():raise RuntimeError('STOP')
                if shutil.disk_usage('/home/mint/rl_dev/instinctRL-dev').free<30*1024**3:raise RuntimeError('DIRECT_DISK_FLOOR')
                return super().check(**kw)
        resource_guard = Guard(gate)
        if contract['operation']=='render':
            from tools.r6_direct.render import run as render
            atomic(attempt/'state.json',{'status':'RENDERING',
                'identity':process_identity(os.getpid()),'counters':counters.snapshot()})
            render(plan,contract,source,output,app,counters,resource_guard)
        else:
            from instinctlab.envs.unified_h1_direct_env import UnifiedH1DirectEnv,UnifiedH1DirectEnvCfg
            from training_workloads.unified_h1.checkpoint_compatibility import pinned_json
            from training_workloads.unified_h1.robustness_training_method import (
                ESCAPE_REPLAY_GAIN_CANDIDATE, ESCAPE_SINGLE_PASS_CANDIDATE, REFERENCE_REPLAY_CANDIDATES,
                HOVER_REFLEX_CANDIDATE,
                RECOVERY_CANDIDATES, action_option_version,
            )
            operation = contract['operation']
            params = contract['parameters']
            moving = operation == 'dynamic'
            option_version = None
            if operation == 'train':
                recipe = pinned_json(params['r6_training_plan']['recipe'])
                option_version = action_option_version(recipe['candidate_id'])
                moving = option_version is not None or recipe['candidate_id'] in (
                    'R6-CANDIDATE-INSTINCT-GPU-R2', 'R6-CANDIDATE-INSTINCT-GPU-R3', HOVER_REFLEX_CANDIDATE)
            if operation == 'evaluate':
                selection = params['evaluation_selection']
                if 'execution_override' in selection:
                    option_version = selection['execution_override']
                elif selection['training_plan'] is not None:
                    recipe = pinned_json(selection['training_plan']['recipe'])
                    option_version = action_option_version(recipe['candidate_id'])
                else:
                    candidate = pinned_json(selection['comparison'])['candidate_id']
                    if candidate in RECOVERY_CANDIDATES:
                        option_version = action_option_version(candidate)
            cfg = UnifiedH1DirectEnvCfg(h1_moving_sphere=moving)
            if plan['run_id'].startswith((*ENDURANCE_PREFIXES, NOVEL_PREFIX, LONG_PREFIX)):
                duration = load_lane_streams(params['stream'])[1][0]['duration_ms']
                cfg.episode_length_s = (duration + 2*STEP_MS)/1000.
            from training_workloads.unified_h1.dynamic_assay import physical_layout, three_case, OPTION_RUN, RECOVERY_PREFIX
            holdout = operation == 'dynamic' and physical_layout(plan['run_id'])
            recovery_training = operation == 'train' and recipe['candidate_id'] in RECOVERY_CANDIDATES
            if holdout:
                from training_workloads.unified_h1.escape_option_env import EscapeHoldoutEnv
                env = EscapeHoldoutEnv(cfg, layout_seed=params['seed'], mirrored_pipe_pairs=pipe_probe,
                                       mirrored_trunk_pairs=three_case(plan['run_id']) == 'FRONT-CLEARANCE',
                                       command_routes=route_layouts(plan['run_id']))
            elif recovery_training:
                from training_workloads.unified_h1.escape_option_env import EscapeRecoveryTrainingEnv
                env = EscapeRecoveryTrainingEnv(cfg)
                env.set_escape_option(True, version=option_version)
                env.single_pass_tube_movers = recipe['candidate_id'] in (ESCAPE_SINGLE_PASS_CANDIDATE,
                    ESCAPE_REPLAY_GAIN_CANDIDATE, *REFERENCE_REPLAY_CANDIDATES)
            elif option_version is not None or plan['run_id'] == OPTION_RUN:
                from training_workloads.unified_h1.escape_option_env import EscapeOptionEnv
                env = EscapeOptionEnv(cfg)
                if option_version is not None:
                    env.set_escape_option(True, version=option_version)
            else:
                env = UnifiedH1DirectEnv(cfg)
            ticks=env.h1_initialization_physics_ticks
            counters.record_initialization_ticks(ticks,contract['budgets']['initialization_ticks'])
            from training_workloads.unified_h1.backend import UnifiedH1LiveBackend
            backend=UnifiedH1LiveBackend(run_contract=contract,attempt=attempt,app=app,environment=env,
                resource_guard=resource_guard,counters=counters,process_started_monotonic=started)
            backend.initialization_measurement=dict(status='OBSERVED',observed_physics_ticks=ticks,
                counter_source=env.h1_initialization_tick_source,fresh_physical_reset=True)
            from training_workloads.unified_h1.robustness_contract import baseline,evaluate_baseline
            if holdout:
                if route_layouts(plan['run_id']):
                    env.holdout_preflight(backend.output, scene='room-corner-trunks-street-corridor-tube')
                else:
                    env.holdout_preflight(backend.output)
            elif recovery_training:
                env.holdout_preflight(backend.output, scene='training-tubes-and-combined-walls')
            else:
                backend.initial_preflight()
            from instinctlab.control.escape_recovery import MOUNT_PITCH_DEGREES
            if option_version in MOUNT_PITCH_DEGREES:
                from training_workloads.unified_h1.lidar_calibration import record as record_mount

                atomic(backend.output/'lidar-calibration.json', record_mount(env, backend.output, 'execution'))
            if command_assay or plan['run_id'].startswith(RECOVERY_PREFIX):
                import torch
                from training_workloads.scale_003.resources import RUNTIME_MIN_FREE_MEMORY_BYTES

                # Preflight can leave unused caching-allocator blocks resident.
                # Release them before the first normal resource guard and retain readings.
                torch.cuda.synchronize()
                before = torch.cuda.mem_get_info()[0]
                reserved = torch.cuda.memory_reserved()
                torch.cuda.empty_cache()
                free = []
                for _ in range(11):
                    if stop_paths() or (Path(plan['output_root'])/'STOP').exists():
                        raise RuntimeError('STOP')
                    free.append(torch.cuda.mem_get_info()[0])
                    if free[-1] >= RUNTIME_MIN_FREE_MEMORY_BYTES:
                        break
                    time.sleep(1)
                atomic(attempt/'preflight_cuda_cache_release.json', dict(
                    before_free_bytes=before, after_free_bytes=free, before_reserved_bytes=reserved,
                    after_reserved_bytes=torch.cuda.memory_reserved()))
            operation=contract['operation']
            atomic(attempt/'state.json',{'status':'TRAINING' if operation=='train' else 'EVALUATING',
                'identity':process_identity(os.getpid()),'counters':counters.snapshot()})
            if operation=='train':
                backend.train_r6()
                for counter,budget in (('training_transitions','training_transitions'),
                        ('real_optimizer_updates','optimizer_updates'),('real_checkpoint_writes','checkpoint_writes')):
                    if getattr(counters,counter)!=contract['budgets'][budget]:raise RuntimeError('DIRECT_FINAL_COUNT:'+counter)
            elif operation=='baseline':
                evaluate_baseline(backend,parameters=contract['parameters'],bindings=baseline(contract['parameters']))
            elif operation == 'dynamic':
                from training_workloads.unified_h1.dynamic_assay import run as run_dynamic
                run_dynamic(backend, contract['parameters'])
            else:
                from training_workloads.unified_h1 import robustness_candidate_contract,robustness_final_contract
                evaluator=robustness_final_contract if operation=='final' else robustness_candidate_contract
                evaluator.evaluate_registered(backend)
                if plan['run_id'] in REGRESSION_IDS:
                    tube = option_version != BRAKE_EXECUTION
                    use_neighbors = option_version in NEIGHBOR_HISTORY_EXECUTIONS
                    standoff = .45 if option_version == CONFIRMED_STANDOFF_EXECUTION else 0.
                    settled_stop = option_version in SETTLED_OFFSET_STOP_EXECUTIONS
                    tangent = option_version in WALL_TANGENT_EXECUTIONS
                    inner_tube = .41 if option_version in CONFIRMED_TUBE_EXECUTIONS else 0.
                    detour = option_version in FINITE_EDGE_EXECUTIONS
                    edge_continuation = option_version == EDGE_CONTINUATION_EXECUTION
                    tangent_magnitude = option_version == TANGENT_MAGNITUDE_EXECUTION
                    aligns_rotation = option_version in ROTATION_TUBE_EXECUTIONS
                    resumes = option_version == VERIFIED_RESUME_EXECUTION
                    zero_stop = option_version in VERIFIED_ZERO_EXECUTIONS
                    intercept_choice = option_version in INTERCEPT_CHOICE_EXECUTIONS
                    early_diagonal = option_version in EARLY_DIAGONAL_EXECUTIONS
                    reserve_direction = option_version in RESERVE_DIRECTION_EXECUTIONS
                    crossing_fallback = option_version in CROSSING_FALLBACK_EXECUTIONS
                    opening_steering = option_version in OPENING_STEERING_EXECUTIONS
                    along_intercept = option_version in ALONG_INTERCEPT_EXECUTIONS
                    finite_intercept = option_version in FINITE_INTERCEPT_EXECUTIONS
                    crossing_stop = option_version in CROSSING_STOP_EXECUTIONS
                    crossing_stop_continuation = option_version == CROSSING_STOP_CONTINUATION_EXECUTION
                    expected = (.15, .60 if tube else 0., tube, use_neighbors, standoff, inner_tube, aligns_rotation)
                    profiles = {(p.config.ics.clearance_margin_m, p.config.ics.motion_tube_radius_m,
                                 p.config.ics.motion_tube_confirm_closure, p.config.ics.motion_tube_neighbor_history,
                                 p.config.ics.confirmed_standoff_m, p.config.ics.confirmed_tube_radius_m,
                                 p.config.ics.confirmed_tube_align_rotation)
                                for p in env._control_pipelines}
                    if (profiles != {expected} or env.escape_option_version != option_version
                            or env.lidar_mount_pitch_degrees != 25 or env.escape_option.settled_offset_stop != settled_stop
                            or env.escape_option.wall_tangent != tangent or env.escape_option.finite_edge != detour
                            or env.escape_option.edge_continuation != edge_continuation
                            or env.escape_option.tangent_magnitude != tangent_magnitude
                            or env.escape_option.verified_resume != resumes or env.escape_option.verified_zero != zero_stop
                            or env.escape_option.intercept_choice != intercept_choice
                            or env.escape_option.receding_stop != intercept_choice
                            or env.escape_option.early_diagonal != early_diagonal
                            or env.escape_option.reserve_direction != reserve_direction
                            or env.escape_option.zero_recheck != crossing_fallback
                            or env.escape_option.reverse_diagonal != crossing_fallback
                            or env.escape_option.opening_steering != opening_steering
                            or env.escape_option.along_intercept != along_intercept
                            or env.escape_option.finite_intercept != finite_intercept
                            or env.escape_option.crossing_stop != crossing_stop
                            or env.escape_option.crossing_stop_continuation != crossing_stop_continuation):
                        raise ValueError('COMMAND_REGRESSION_EXECUTION_READBACK')
                    execution = dict(execution=option_version,
                           checkpoint=selection['checkpoint'], mount_pitch_degrees=env.lidar_mount_pitch_degrees,
                           clearance_margin_m=expected[0], motion_tube_radius_m=expected[1], confirm_motion=expected[2])
                    if use_neighbors:
                        execution['uses_neighbor_history'] = True
                    if standoff:
                        execution['confirmed_standoff_m'] = standoff
                    if settled_stop:
                        execution['settled_offset_stop'] = True
                    if tangent:
                        execution['wall_tangent'] = True
                    if inner_tube:
                        execution['confirmed_tube_radius_m'] = inner_tube
                    if aligns_rotation:
                        execution['confirmed_tube_align_rotation'] = True
                    if detour:
                        execution['finite_edge'] = True
                    if edge_continuation:
                        execution['edge_continuation'] = True
                    if tangent_magnitude:
                        execution['tangent_magnitude'] = True
                    if resumes:
                        execution['verified_resume'] = True
                    if zero_stop:
                        execution['verified_zero'] = True
                    if intercept_choice:
                        execution.update(intercept_choice=True, receding_stop=True)
                    if early_diagonal:
                        execution['early_diagonal'] = True
                    if reserve_direction:
                        execution['reserve_direction'] = True
                    if crossing_fallback:
                        execution.update(zero_recheck=True, reverse_diagonal=True)
                    if opening_steering:
                        execution['opening_steering'] = True
                    if along_intercept:
                        execution['along_intercept'] = True
                    if finite_intercept:
                        execution['finite_intercept'] = True
                    if crossing_stop:
                        execution['crossing_stop'] = True
                    if crossing_stop_continuation:
                        execution['crossing_stop_continuation'] = True
                    atomic(backend.output/'command-execution.json', execution)
            if counters.nontraining_transitions!=contract['budgets']['nontraining_transitions']:raise RuntimeError('DIRECT_FINAL_COUNT')
            if journaled:
                resource_guard.seal()
            backend.preserve('PASS_PENDING_CLOSE')
    except BaseException as exc:
        error=repr(exc)
        if resource_guard is not None and journaled:
            try:
                resource_guard.seal()
            except BaseException as seal_error:
                exc.add_note('Runtime resource summary could not be written: '+repr(seal_error))
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
    from tools.r6_direct.dependencies import install
    install()
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--check',action='store_true')
    args=parser.parse_args(); input_pin=pin(args.inputs); plan=json.loads(args.inputs.read_text())
    if args.check:
        preflight(plan);print('PASS_CPU_DIRECT_PREFLIGHT_NO_GPU');return
    run(plan,input_pin)


if __name__=='__main__':main()
