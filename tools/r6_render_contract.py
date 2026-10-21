"""Exact render parameters and resource accounting; no Store issuance or launch.

Replay consumes a simulator start and allocated GPU time even though it executes
zero policy/physics transitions. Production dispatch remains a separate gate.
"""
import json
import re
from pathlib import Path

from tools.r6_recovery import checked, file_pin, RecoveryError
from training_workloads.unified_h1 import robustness_contract as base
from training_workloads.unified_h1 import native_volume, result_volume, extension_inputs
from training_workloads.unified_h1 import robustness_prelaunch
from training_workloads.scale_003 import resources as launch_resources

MODE = 'research-r6-video-replay'
PAYLOAD = 1024**3
FIELDS = {'experiment_id', 'master_authority', 'extraction', 'source_snapshot',
          'native_volume', 'result_volume', 'extension_inputs', 'output_cap_bytes',
          'wall_seconds', 'implementation', 'independent_review'}
ZERO = ('training_transitions', 'nontraining_transitions', 'optimizer_updates',
        'checkpoint_writes', 'initialization_ticks', 'explicit_resets', 'automatic_resets')


def require(ok, reason):
    if not ok:
        raise RecoveryError('RENDER_CONTRACT_' + reason)


def sources():
    from tools import render_r6_isaac_video, r6_video_presentation, run_r6_render
    from tools import r6_recovery, r6_budget, r6_video_trajectory
    from rlflow import store
    pins = {name: file_pin(module.__file__) for name, module in (
        ('contract', __import__(__name__, fromlist=[''])),
        ('renderer', render_r6_isaac_video), ('presentation', r6_video_presentation), ('entry', run_r6_render),
        ('launcher', launch_resources), ('prelaunch', robustness_prelaunch),
        ('recovery', r6_recovery), ('budget', r6_budget), ('trajectory', r6_video_trajectory))}
    # The replay tools import these CPU helpers. Pin the complete local package,
    # rather than depending on an ambient mutable control checkout at runtime.
    package = Path(store.__file__).parent
    pins.update({'rlflow:'+str(p.relative_to(package)): file_pin(p) for p in sorted(package.rglob('*.py'))})
    return pins


def budget(wall_seconds):
    require(type(wall_seconds) is int and 1 <= wall_seconds <= 3600, 'WALL_SECONDS')
    return dict(live_process_starts=1, simulator_initializations=1,
                wall_seconds=wall_seconds, **{name: 0 for name in ZERO})


def exact_budget(value, wall_seconds):
    require(isinstance(value, dict) and value == budget(wall_seconds)
            and all(type(v) is int for v in value.values()), 'BUDGET')


def parameters(value, *, attempt, require_review=True):
    require(isinstance(value, dict) and set(value) == FIELDS, 'FIELDS')
    p = value
    require(isinstance(p['experiment_id'], str)
            and re.fullmatch(r'R6-[A-Z0-9-]{1,90}', p['experiment_id']), 'EXPERIMENT_ID')
    budget(p['wall_seconds'])
    master = json.loads(checked(p['master_authority']).read_text())
    require(master.get('id') == 'R6-ROBUSTNESS-AUTO-20261019'
            and master.get('status') == 'REGISTERED_USER_DELEGATION', 'MASTER')
    for name, digest in (('user_authority', base.USER_SHA), ('requirements', base.REQUIREMENTS_SHA)):
        # Use the same original authority fields as the existing R6 contracts.
        pin = master[name]
        checked(pin); require(pin['sha256'] == digest, 'USER_AUTHORITY')
    require(p['implementation'] == sources(), 'SOURCE_DRIFT')
    source = Path(p['source_snapshot'])
    require(source.is_absolute() and source.resolve() == source and not source.is_symlink()
            and source.is_dir(), 'SNAPSHOT_PATH')
    for pin in p['implementation'].values():
        require(checked(pin).is_relative_to(source), 'IMPLEMENTATION_OUTSIDE_SNAPSHOT')
        require(not checked(pin).stat().st_mode & 0o222, 'WRITABLE_IMPLEMENTATION')
    extraction = json.loads(checked(p['extraction']).read_text())
    require(extraction.get('status') == 'RECORDED_TRAJECTORIES_EXTRACTED_NOT_RENDERED'
            and extraction.get('fps') == 10 and extraction.get('physics_reintegrated') is False
            and type(extraction.get('total_frames')) is int
            and 1 <= extraction['total_frames'] <= 14*330, 'EXTRACTION')
    from tools.r6_video_presentation import load
    for clip in load(p['extraction']):
        recorded = clip['decoded']['clip']
        checked(recorded['checkpoint'])
        for relative in ('isaaclab-training/instinctlab/task/scene_h1.py',
                         'isaaclab-training/instinctlab/assets/taslab_uav/taslab_uav.usd'):
            require(file_pin(Path(recorded['source_snapshot'])/relative)['sha256']
                    == file_pin(source/relative)['sha256'], 'SCENE_SOURCE')
    target = Path(attempt)/'results'
    require(Path(attempt).is_absolute() and Path(attempt).resolve() == Path(attempt), 'ATTEMPT')
    require(type(p['output_cap_bytes']) is int, 'OUTPUT_CAP')
    native_volume.validate(p['native_volume'])
    binding = result_volume.validate(p['result_volume'], output_cap_bytes=p['output_cap_bytes'],
                                     expected_target=str(target), minimum_free_bytes=PAYLOAD)
    require(binding.get('payload_cap_bytes') == PAYLOAD, 'FIXED_RENDER_PAYLOAD')
    require(p['output_cap_bytes'] == PAYLOAD+96*1024**2+native_volume.MAXIMUM+result_volume.CONTROL_ALLOWANCE,
            'AGGREGATE_CAP')
    extension_inputs.validate(p['extension_inputs'])
    if require_review:
        payload = {k: v for k, v in p.items() if k != 'independent_review'}
        base.accepted_review(p['independent_review'], 'PASS_R6_RENDER_PLAN',
                             payload_sha256=base.object_sha(payload))
    return p


def authorization(pin, attempt, config):
    auth = json.loads(checked(pin).read_text())
    require(set(auth) == {'schema_version', 'mode', 'attempt', 'parameters', 'budgets', 'config'}, 'AUTH_FIELDS')
    require(auth['schema_version'] == 'r6-video-replay-authorization/1' and auth['mode'] == MODE
            and auth['attempt'] == str(attempt), 'AUTH_IDENTITY')
    p = parameters(auth['parameters'], attempt=attempt)
    config_path = Path(p['source_snapshot'])/'training_workloads/unified_h1/candidate_config.json'
    require(checked(auth['config']) == config_path
            and json.loads(config_path.read_text()) == {k:v for k,v in config.items() if k!='resolved_identity'}
            and config.get('resolved_identity', {}).get('config_sha256') == auth['config']['sha256'], 'CONFIG_BINDING')
    exact_budget(auth['budgets'], p['wall_seconds'])
    return dict(auth, authorization_sha256=pin['sha256'])


def resources(spec, auth, *, prepaid=False):
    require(spec.get('backend') == 'real' and spec.get('h1', {}).get('mode') == MODE
            and auth.get('mode') == MODE, 'MODE')
    p = auth['parameters']
    exact_budget(auth.get('budgets'), p['wall_seconds'])
    limit = p['wall_seconds']+10
    require(spec['resources'] == dict(starts=1, wall_seconds=limit, gpu_seconds=limit, transitions=0)
            and all(type(v) is int for v in spec['resources'].values()), 'SPEC_RESOURCES')
    require(type(p['output_cap_bytes']) is int and p['output_cap_bytes'] ==
            PAYLOAD+96*1024**2+native_volume.MAXIMUM+result_volume.CONTROL_ALLOWANCE, 'AGGREGATE_CAP')
    return dict(live_starts=1, simulator_initializations=1,
                live_wall_seconds=limit+180, allocated_gpu_seconds=limit+180,
                nontraining_transitions=0, training_transitions=0, training_checkpoint_writes=0,
                output_actual_plus_unreleased_reservations_bytes=0 if prepaid else p['output_cap_bytes'])


def launch(authorization, attempt, config, owners):
    """Caller must already own an exact Store job and bounded cleanup watchdog."""
    require(authorization.get('mode') == MODE, 'MODE')
    p = parameters(authorization['parameters'], attempt=attempt)
    exact_budget(authorization.get('budgets'), p['wall_seconds'])
    require(not owners.get('app') and not owners.get('launcher'), 'ALREADY_LAUNCHED')
    from functools import partial
    from training_workloads.unified_h1.config import resource_binding_pins, validate_prelaunch_environment
    validate_prelaunch_environment(config)
    pins = resource_binding_pins()
    base_dir = Path(p['master_authority']['path']).parent
    root = base_dir.parents[2]
    stops = (base_dir/'STOP', base_dir/'batch/STOP', root/'STOP', root/'coordination/STOP',
             root/'coordination/commander/STOP', root/'coordination/executor/STOP')
    require(not any(path.exists() for path in stops), 'STOP')
    output = Path(attempt)/'results'/'prelaunch'
    output.mkdir(exist_ok=False)
    gate = partial(robustness_prelaunch.run_gate, resource_pins=pins, stop_paths=stops)
    observation, launcher, app = launch_resources.start_after_coexistence_gate(
        output, mode=MODE, attempt_id=Path(attempt).name,
        authorization_sha256=authorization['authorization_sha256'], gate_runner=gate,
        resource_pins=pins, asset_root=Path(config['execution']['isaacsim_asset_root']),
        native_volume_pin=p['native_volume'], extension_inputs_pin=p['extension_inputs'])
    # Register before any readback that may fail, so caller's finally owns cleanup.
    owners.update(launcher=launcher, app=app)
    launch_resources.verify_r6_native_paths(output, p['native_volume'])
    launch_resources.validate_graceful_shutdown_configuration(app, resource_pins=pins)
    return observation
