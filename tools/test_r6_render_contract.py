"""CPU-only render admission, charges and cleanup ownership; no GPU queries."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools import r6_render_contract as m
from tools.r6_job_resources import resources
from tools.r6_recovery import RecoveryError, file_pin
from tools.test_r6_native_volume_binding import wired
from training_workloads.unified_h1 import config


def inputs():
    p = dict(wall_seconds=600, output_cap_bytes=m.PAYLOAD+96*1024**2+m.native_volume.MAXIMUM+m.result_volume.CONTROL_ALLOWANCE)
    auth = dict(mode=m.MODE, parameters=p, budgets=m.budget(600))
    spec = dict(backend='real', h1=dict(mode=m.MODE),
                resources=dict(starts=1, wall_seconds=610, gpu_seconds=610, transitions=0))
    return spec, auth


def test_render_charges_gpu_start_and_initialization_with_zero_transitions():
    spec, auth = inputs()
    actual = resources(spec, auth)
    assert actual == dict(live_starts=1, simulator_initializations=1,
        live_wall_seconds=790, allocated_gpu_seconds=790, nontraining_transitions=0,
        training_transitions=0, training_checkpoint_writes=0,
        output_actual_plus_unreleased_reservations_bytes=auth['parameters']['output_cap_bytes'])
    assert resources(spec, auth, prepaid=True) == dict(actual, output_actual_plus_unreleased_reservations_bytes=0)


@pytest.mark.parametrize('field', m.ZERO)
@pytest.mark.parametrize('value', [1, False])
def test_any_physics_or_learning_budget_is_rejected(field, value):
    spec, auth = inputs(); auth['budgets'][field] = value
    with pytest.raises(RecoveryError, match='BUDGET'): resources(spec, auth)


@pytest.mark.parametrize('fault', ['cpu', 'gpu', 'wall', 'transitions', 'output'])
def test_resource_mismatch_cannot_hide_rendering(fault):
    spec, auth = inputs()
    if fault == 'cpu': spec['backend'] = 'runner_cpu'
    if fault == 'gpu': spec['resources']['gpu_seconds'] = 0
    if fault == 'wall': spec['resources']['wall_seconds'] = 600
    if fault == 'transitions': spec['resources']['transitions'] = 1
    if fault == 'output': auth['parameters']['output_cap_bytes'] -= 4096
    with pytest.raises(RecoveryError): resources(spec, auth)


def test_real_launcher_argument_builder_enables_cameras_and_keeps_quota(wired):
    wired.start(m.MODE)
    call = wired.calls[0]
    assert call['enable_cameras'] is True and call['fast_shutdown'] is False
    assert '--/log/file=/proc/self/fd/1' in call['kit_args']
    assert '--/rtx/shaderDb/shaderCachePath='+str(wired.target/'shader') in call['kit_args']
    wired.gate.assert_called_once()


def test_render_without_hard_volume_never_reaches_gate(tmp_path):
    gate = Mock()
    with pytest.raises(ValueError, match='hard native'):
        m.launch_resources.start_after_coexistence_gate(tmp_path, mode=m.MODE,
            attempt_id='CPU', authorization_sha256='a'*64, gate_runner=gate)
    gate.assert_not_called()


def test_renderer_cannot_create_its_own_unadmitted_app(tmp_path):
    from tools.render_r6_isaac_video import render
    with pytest.raises(RecoveryError, match='ADMITTED_APP_REQUIRED'):
        render({}, tmp_path/'output', source_snapshot=tmp_path, asset_root=tmp_path,
               app=None, check=lambda: None)
    assert not (tmp_path/'output').exists()


@pytest.fixture
def parameters(tmp_path, monkeypatch):
    def pin(name, data):
        p=tmp_path/name; p.write_text(json.dumps(data)); return file_pin(p)
    master = Path(m.__file__).resolve().parents[1]/'docs/experiments/r6-robustness-auto/MASTER_AUTHORITY_V2.json'
    if not master.exists():
        master = Path(m.__file__).resolve().parents[1]/'docs/experiments/r6-robustness-auto/MASTER_AUTHORITY.json'
    impl = pin('source.py', {'fixture': 'not executable'}); Path(impl['path']).chmod(0o444)
    monkeypatch.setattr(m, 'sources', lambda: {'fixture': impl})
    monkeypatch.setattr(m.native_volume, 'validate', lambda *a, **k: {})
    monkeypatch.setattr(m.result_volume, 'validate', lambda *a, **k: {'payload_cap_bytes':m.PAYLOAD})
    monkeypatch.setattr(m.extension_inputs, 'validate', lambda *a, **k: None)
    from tools import r6_video_presentation
    monkeypatch.setattr(r6_video_presentation, 'load', lambda p: [])  # No scientific inputs in this admission fixture.
    _, auth = inputs()
    p = dict(auth['parameters'], experiment_id='R6-RENDER-CPU-FIXTURE', master_authority=file_pin(master),
        extraction=pin('extraction.json', dict(status='RECORDED_TRAJECTORIES_EXTRACTED_NOT_RENDERED',
            fps=10, physics_reintegrated=False, total_frames=200)), source_snapshot=str(tmp_path),
        native_volume={'fixture':True}, result_volume={'fixture':True}, extension_inputs={'fixture':True},
        implementation=m.sources(), independent_review=None)
    payload = {k:v for k,v in p.items() if k!='independent_review'}
    p['independent_review'] = pin('review.json', dict(decision='PASS_R6_RENDER_PLAN', must_fix=[],
        reviewer='explicit-cpu-review-substitute', payload_sha256=m.base.object_sha(payload)))
    return p


def test_exact_review_and_output_parameters(parameters, tmp_path):
    assert m.parameters(parameters, attempt=tmp_path/'attempt') == parameters
    changed = copy.deepcopy(parameters); changed['wall_seconds'] += 1
    with pytest.raises(PermissionError, match='input mismatch'): m.parameters(changed, attempt=tmp_path/'attempt')
    Path(parameters['implementation']['fixture']['path']).chmod(0o644)
    with pytest.raises(RecoveryError, match='WRITABLE'): m.parameters(parameters, attempt=tmp_path/'attempt')


def test_authorization_binds_loaded_config_and_exact_attempt(parameters, tmp_path):
    config_path=tmp_path/'training_workloads/unified_h1/candidate_config.json'
    config_path.parent.mkdir(parents=True); config_path.write_text('{"fixture":"CPU only"}')
    config_pin=file_pin(config_path)
    loaded=dict(fixture='CPU only', resolved_identity={'config_sha256':config_pin['sha256']})
    attempt=tmp_path/'attempt'
    auth=dict(schema_version='r6-video-replay-authorization/1', mode=m.MODE, attempt=str(attempt),
        parameters=parameters, budgets=m.budget(600), config=config_pin)
    path=tmp_path/'authorization.json'; path.write_text(json.dumps(auth)); pin=file_pin(path)
    assert m.authorization(pin, attempt, loaded)['authorization_sha256']==pin['sha256']
    with pytest.raises(RecoveryError, match='AUTH_IDENTITY'): m.authorization(pin, tmp_path/'other', loaded)
    with pytest.raises(RecoveryError, match='CONFIG_BINDING'): m.authorization(pin, attempt, dict(loaded, fixture='changed'))


def test_app_registered_for_cleanup_before_failed_readback(parameters, tmp_path, monkeypatch):
    attempt=tmp_path/'attempt'; (attempt/'results').mkdir(parents=True)
    monkeypatch.setattr(config, 'validate_prelaunch_environment', lambda c: None)
    monkeypatch.setattr(config, 'resource_binding_pins', lambda: {})
    app=object(); launcher=object(); observations=[]
    def start(*a, **kwargs):
        observations.append(kwargs)
        return {'status':'PASS'}, launcher, app
    monkeypatch.setattr(m.launch_resources, 'start_after_coexistence_gate', start)
    monkeypatch.setattr(m.launch_resources, 'verify_r6_native_paths', Mock(side_effect=RuntimeError('readback failure')))
    owners={}
    auth=dict(mode=m.MODE, parameters=parameters, budgets=m.budget(600), authorization_sha256='a'*64)
    with pytest.raises(RuntimeError, match='readback failure'):
        m.launch(auth, attempt, {'execution':{'isaacsim_asset_root':'/CPU_NOT_ASSETS'}}, owners)
    assert owners == dict(app=app, launcher=launcher)
    assert observations[0]['gate_runner'].func is m.robustness_prelaunch.run_gate
    assert observations[0]['mode'] == m.MODE


@pytest.mark.parametrize('failure', ['launch_readback', 'render', 'persist', 'close', 'none'])
def test_entry_preserves_failure_and_always_closes_owned_app(parameters, tmp_path, monkeypatch, failure):
    from tools import run_r6_render as entry, render_r6_isaac_video as renderer
    from training_workloads.scale_003 import persistence
    from types import SimpleNamespace
    monkeypatch.setattr(persistence, '_output_quota', None)
    attempt=tmp_path/'attempt'; (attempt/'results').mkdir(parents=True)
    app=SimpleNamespace(close=Mock())
    if failure=='close': app.close.side_effect=RuntimeError('close failure')
    def launch(auth, path, config, owners):
        owners.update(app=app, launcher='CPU_SUBSTITUTE')
        if failure=='launch_readback': raise RuntimeError('readback failure')
    monkeypatch.setattr(m, 'launch', launch)
    monkeypatch.setattr(m.launch_resources, 'RuntimeResourceGuard', lambda p: SimpleNamespace(check=lambda **kw: None))
    # Exercise entry lifecycle, but no watchdog fork or Kit shutdown in this fixture.
    monkeypatch.setattr(entry, '_bounded_owner_call', lambda fn,*a,**k: fn())
    def render(extraction, output, **kwargs):
        kwargs['check']()
        assert kwargs['app'] is app
        if failure=='render': raise RuntimeError('render failure')
        output.mkdir(); (output/'RUN_MANIFEST.json').write_text('{}')
        return {}
    monkeypatch.setattr(renderer, 'render', render)
    write=entry.atomic_json_replace
    def persist(root, name, value):
        if failure=='persist' and name=='status_before_cleanup.json': raise OSError('quota full')
        return write(root, name, value)
    monkeypatch.setattr(entry, 'atomic_json_replace', persist)
    auth=dict(mode=m.MODE, parameters=parameters, budgets=m.budget(600), authorization_sha256='a'*64)
    result=entry.run(auth, attempt, {'execution':{'isaacsim_asset_root':'/CPU_NOT_ASSETS'}})
    app.close.assert_called_once()
    assert result['status']==('PASS' if failure=='none' else 'FAIL')
    assert result['counters']['nontraining_transitions']==0
    assert result['counters']['simulator_initializations']==1
    with pytest.raises(RecoveryError, match='FRESH_ATTEMPT'): entry.run(auth, attempt, {})
