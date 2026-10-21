"""Real CPU fork/exec and exact native log routing; mount validation substituted."""
import json
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest
from training_workloads.unified_h1 import local_dispatch as dispatch
from training_workloads.unified_h1 import native_volume


@pytest.fixture
def setup(tmp_path,monkeypatch):
    target=tmp_path/'quota';target.mkdir()
    control=tmp_path/'control';control.mkdir()
    coordination=tmp_path/'coordination';coordination.mkdir()
    binding=dict(target=str(target),manifest={'fixture':True})
    monkeypatch.setattr(native_volume,'validate',lambda pin:binding)
    def run(code):
        return dispatch.dispatch_held([sys.executable,'-B','-c',code],cwd=tmp_path,
            identity_path=control/'held.json',coordination_root=coordination,
            native_output_cap_bytes=4096,native_volume_pin={'fixture':True},
            environment={'TMPDIR':'/unverified','CUDA_CACHE_PATH':'/unverified','WARP_CACHE_PATH':'/unverified'},completion_timeout_s=5)
    return target,control,run


def test_actual_child_console_and_environment_are_bound_to_volume(setup):
    target,control,run=setup
    assert run("import os,json;print(json.dumps({k:os.environ[k] for k in ['TMPDIR','TMP','TEMP','XDG_CACHE_HOME','CUDA_CACHE_PATH','WARP_CACHE_PATH','INSTINCTRL_R6_NATIVE_CAPTURE']}))")==0
    record=json.loads((target/'native-output.log').read_text())
    for key in ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','CUDA_CACHE_PATH','WARP_CACHE_PATH'):
        assert Path(record[key]).is_relative_to(target) and Path(record[key]).is_dir()
    capture=json.loads(record['INSTINCTRL_R6_NATIVE_CAPTURE'])
    assert capture['path']==str(target/'native-output.log')
    assert capture['volume']['target']==str(target)
    assert not (control/'native-output.log').exists()
    receipt=json.loads((control/'native-output.json').read_text())
    assert receipt['path']==str(target/'native-output.log') and receipt['status']=='COMPLETE'


def test_exact_volume_log_still_has_pipe_cap_and_failure_receipt(setup):
    target,control,run=setup
    with pytest.raises(dispatch.DispatchError,match='cap exceeded'):
        run("import os\nwhile True:os.write(1,b'x'*8192)")
    assert (target/'native-output.log').read_bytes()==b'x'*4096
    assert json.loads((control/'native-output.json').read_text())['status']=='FAILED_PREFIX_PRESERVED'


def test_rejected_mount_cannot_fork_or_create_log(setup,monkeypatch):
    target,control,run=setup
    def reject(pin):raise ValueError('CPU MOUNT_ABSENT')
    monkeypatch.setattr(native_volume,'validate',reject)
    fork=Mock();monkeypatch.setattr(dispatch.os,'fork',fork)
    with pytest.raises(ValueError,match='MOUNT_ABSENT'):run('pass')
    fork.assert_not_called();assert not list(target.iterdir())


def test_previous_native_directory_is_preserved_and_not_reused(setup):
    target,control,run=setup
    (target/'tmp').mkdir();(target/'tmp'/'original').write_text('preserve')
    with pytest.raises(FileExistsError):run('pass')
    assert (target/'tmp'/'original').read_text()=='preserve'
    assert not (control/'held.json').exists()


def test_store_adapter_uses_validated_authority_not_untrusted_spec_path(monkeypatch):
    from rlflow import h1
    approved={'path':'/approved/volume.json','sha256':'1'*64}
    check=Mock(return_value={'authorization':{'parameters':{'native_volume':approved}}})
    monkeypatch.setattr(h1,'inspect_authority',check)
    spec={'backend':'real','h1':{'mode':'research-r6-evaluation'},'native_volume':{'path':'/untrusted'}}
    assert h1.native_dispatch_options(spec)==dict(native_output_cap_bytes=256*1024**2,native_volume_pin=approved)
    check.assert_called_once_with(spec)
    check.side_effect=RuntimeError('CPU invalid authority')
    with pytest.raises(RuntimeError,match='invalid authority'):h1.native_dispatch_options(spec)
    assert h1.native_dispatch_options({'backend':'runner_cpu'})=={}
