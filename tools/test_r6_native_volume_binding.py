"""Read-only checks of the actual CPU volume and explicit launcher substitutes."""
import hashlib
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
from unittest.mock import Mock
import pytest
from training_workloads.unified_h1 import native_volume as volume
from training_workloads.scale_003 import resources


def cpu_pin():
    path=volume.BASE/'CPU-QUOTA-01/volume.json'
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_actual_cpu_volume_identity_read_only():
    binding=volume.validate(cpu_pin(),allow_cpu_fixture=True,minimum_free_bytes=0)
    assert binding['maximum_bytes']==1610612736
    assert binding['device'].startswith('/dev/loop')


def test_cpu_evidence_volume_never_accepted_for_production():
    with pytest.raises(ValueError,match='PRODUCTION_IDENTITY'):volume.validate(cpu_pin())


def test_wrong_pin_and_insufficient_space_are_rejected():
    pin=cpu_pin()
    with pytest.raises(ValueError,match='MANIFEST_HASH'):volume.validate({**pin,'sha256':'0'*64})
    with pytest.raises(ValueError,match='FREE_SPACE'):
        volume.validate(pin,allow_cpu_fixture=True,minimum_free_bytes=256*1024**2)


@pytest.fixture
def wired(tmp_path,monkeypatch):
    target=tmp_path/'quota';target.mkdir()
    log=target/'console.log';log.touch()
    output=tmp_path/'result';output.mkdir()
    binding=dict(target=str(target));capture=dict(path=str(log))
    # Only mount validation is substituted. Path membership/device checks and
    # actual resources.py launcher argument construction are exercised.
    monkeypatch.setattr(volume,'validate',lambda *a,**k:binding)
    from training_workloads.unified_h1 import extension_inputs
    inputs=tmp_path/'installed_extensions';inputs.mkdir()
    config=tmp_path/'user.config.json';config.write_text('{"persistent":{"sentinel":42}}')
    config_pin=dict(path=str(config),sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    metadata=tmp_path/'extensions.json'
    import json
    metadata.write_text(json.dumps(dict(schema_version='r6-extension-inputs/2',tree=extension_inputs.tree_identity(inputs),user_config=config_pin)))
    input_pin=dict(path=str(metadata),sha256=hashlib.sha256(metadata.read_bytes()).hexdigest())
    monkeypatch.setattr(resources,'_r6_capture_binding',lambda:capture)
    monkeypatch.setattr(resources,'validate_shutdown_runtime_sources',lambda:None)
    calls=[];module=ModuleType('isaaclab.app')
    def launch(**kwargs):calls.append(kwargs);return SimpleNamespace(app='CPU_ONLY')
    module.AppLauncher=launch;monkeypatch.setitem(sys.modules,'isaaclab.app',module)
    gate=Mock(return_value={'status':'PASS'})
    def start():return resources.start_after_coexistence_gate(output,mode='research-r6-evaluation',
        attempt_id='CPU',authorization_sha256='a'*64,gate_runner=gate,native_volume_pin={'fixture':True},extension_inputs_pin=input_pin)
    return SimpleNamespace(target=target,output=output,capture=capture,calls=calls,gate=gate,start=start,inputs=inputs,config=config)


def test_persistent_config_copied_exactly_without_changing_source(wired):
    original=wired.config.read_bytes()
    wired.start()
    dest=wired.target/'user.config.json'
    assert dest.read_bytes()==original==wired.config.read_bytes()
    assert dest.stat().st_mode & 0o777==0o600
    assert '--/app/userConfigPath='+str(dest) in wired.calls[0]['kit_args'].split()
    dest.write_text('{"persistent":{"sentinel":99}}')
    assert wired.config.read_bytes()==original


def test_persistent_config_drift_rejected_before_gate(wired):
    wired.config.write_text('{}')
    with pytest.raises(Exception):wired.start()
    wired.gate.assert_not_called();assert not wired.calls


def test_existing_config_output_is_preserved_and_launcher_not_called(wired):
    dest=wired.target/'user.config.json';dest.write_text('old evidence')
    with pytest.raises(FileExistsError):wired.start()
    assert dest.read_text()=='old evidence' and not wired.calls


def test_extension_drift_rejected_before_gpu_gate(wired):
    (wired.inputs/'unexpected.py').write_text('changed')
    with pytest.raises(ValueError,match='content mismatch'):wired.start()
    wired.gate.assert_not_called();assert not wired.calls


def test_capture_outside_volume_rejected_before_gpu_gate(wired):
    wired.capture['path']=str(wired.output/'external.log')
    with pytest.raises(ValueError,match='CAPTURE_OUTSIDE_VOLUME'):wired.start()
    wired.gate.assert_not_called();assert not wired.calls


@pytest.mark.parametrize('setting',[
    '/crashreporter/dumpDir','/rtx/shaderDb/shaderCachePath',
    '/rtx/shaderDb/driverShaderCachePath','/rtx-transient/resourcemanager/localTextureCachePath',
    '/app/extensions/registryCacheFull','/exts/omni.kit.registry.nucleus/cachePath',
    '/app/extensions/registryCache','/exts/omni.kit.pipapi/envPath','/structuredLog/logDirectory'])
def test_exact_settings_reach_launcher_and_readback_rejects_drift(wired,monkeypatch,setting):
    wired.start()
    assert ' --ext-folder ' in wired.calls[0]['kit_args']
    requested=resources._r6_native_settings(wired.output,{'fixture':True})
    assert '--'+setting+'='+requested[setting] in wired.calls[0]['kit_args'].split()
    assert Path(requested[setting]).is_dir()
    monkeypatch.setattr(resources,'_read_r6_native_settings',lambda keys:dict(requested))
    resources.verify_r6_native_paths(wired.output,{'fixture':True})
    monkeypatch.setattr(resources,'_read_r6_native_settings',lambda keys:{**requested,setting:'/tmp/unbounded'})
    with pytest.raises(RuntimeError,match='settings differ'):
        resources.verify_r6_native_paths(wired.output,{'fixture':True})


@pytest.mark.parametrize('setting',[
    '/telemetry/log/file','/telemetry/testLogFile','/telemetry/cloudLogEndpoint'])
def test_telemetry_files_and_endpoint_are_inside_volume(wired,monkeypatch,setting):
    from urllib.parse import urlparse
    wired.start()
    expected=resources._r6_native_settings(wired.output,{'fixture':True})
    value=expected[setting]
    path=Path(urlparse(value).path if setting.endswith('Endpoint') else value)
    assert path.is_relative_to(wired.target) and path.parent.is_dir()
    assert '--'+setting+'='+value in wired.calls[0]['kit_args'].split()
    monkeypatch.setattr(resources,'_read_r6_native_settings',lambda keys:{**expected,setting:'/tmp/escape'})
    with pytest.raises(RuntimeError,match='settings differ'):
        resources.verify_r6_native_paths(wired.output,{'fixture':True})


@pytest.mark.parametrize('setting',[
    '/app/extensions/registryEnabled',
    '/exts/omni.kit.pipapi/allowOnlineIndex','/exts/omni.kit.pipapi/tryUpgradePipOnFirstUse',
    '/exts/omni.kit.registry.nucleus/cachePrune/enabled',
    '/exts/omni.kit.registry.nucleus/cachePrune/runOnStartup'])
def test_registry_policy_reaches_launcher_and_rejects_enable_or_type_drift(wired,monkeypatch,setting):
    wired.start()
    requested=resources._r6_native_settings(wired.output,{'fixture':True})
    assert '--'+setting+'=false' in wired.calls[0]['kit_args'].split()
    assert '/app/exts/folders' not in requested
    for bad in (True,0,'false',None):
        monkeypatch.setattr(resources,'_read_r6_native_settings',lambda keys:{**requested,setting:bad})
        with pytest.raises(RuntimeError,match='settings differ'):
            resources.verify_r6_native_paths(wired.output,{'fixture':True})
