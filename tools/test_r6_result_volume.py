import hashlib
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from training_workloads.unified_h1 import result_volume as result
from tools import provision_r6_result_volume as provision


def test_result_binding_uses_exact_target_and_job_remainder(tmp_path,monkeypatch):
    attempts=tmp_path/'attempts';target=attempts/'R6-CPU/results';target.mkdir(parents=True)
    monkeypatch.setattr(result,'ATTEMPTS',attempts)
    size=64*1024**2;cap=result.MAXIMUM+result.CONTROL_ALLOWANCE+size
    path=tmp_path/'volume.json'
    def pin(target_value=target,size_value=size):
        path.write_text(json.dumps(dict(target=str(target_value),maximum_backing_bytes=size_value)))
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    check=Mock(return_value=dict(target=str(target)));monkeypatch.setattr(result,'validate_volume',check)
    assert result.validate(pin(),output_cap_bytes=cap,expected_target=target)['target']==str(target)
    assert check.call_args.kwargs['maximum']==size
    for wrong_size in (size+4096,size-4096):
        with pytest.raises(ValueError,match='RESULT_AGGREGATE_CAP'):result.validate(pin(size_value=wrong_size),output_cap_bytes=cap)
    with pytest.raises(ValueError,match='RESULT_ATTEMPT_BINDING'):
        result.validate(pin(),output_cap_bytes=cap,expected_target=attempts/'R6-OTHER/results')
    with pytest.raises(ValueError,match='RESULT_TARGET'):result.validate(pin(target_value=tmp_path),output_cap_bytes=cap)


def test_provision_plan_does_not_mount_and_enforces_new_result_scope(tmp_path,monkeypatch):
    monkeypatch.setattr(provision,'ATTEMPTS',tmp_path/'attempts')
    target=provision.ATTEMPTS/'R6-CPU/results'
    spec=provision.plan('CPU-QUOTA-02',target,16*1024**2)
    assert spec['target']==str(target) and spec['commands'][-1][-1]==str(target)
    assert not target.exists()
    with pytest.raises(ValueError):provision.plan('CPU-QUOTA-02',tmp_path/'results',16*1024**2)
    with pytest.raises(ValueError):provision.plan('CPU-QUOTA-02',target,16*1024**2+1)
    monkeypatch.setattr(provision.os,'geteuid',lambda:1000)
    run=Mock();monkeypatch.setattr(provision.subprocess,'run',run)
    with pytest.raises(PermissionError):provision.apply(spec)
    run.assert_not_called()


@pytest.mark.parametrize('case',['existing_identity','nonempty_target','disk_floor'])
def test_result_provision_guards_before_image_or_mount(tmp_path,monkeypatch,case):
    from types import SimpleNamespace
    monkeypatch.setattr(provision,'BASE',tmp_path/'volumes')
    monkeypatch.setattr(provision,'ATTEMPTS',tmp_path/'attempts')
    monkeypatch.setattr(provision.os,'geteuid',lambda:0)
    monkeypatch.setattr(provision,'secure',lambda p:None)
    monkeypatch.setattr(provision.shutil,'disk_usage',lambda p:SimpleNamespace(free=provision.FLOOR if case=='disk_floor' else 100*1024**3))
    target=provision.ATTEMPTS/'R6-CPU/results';target.mkdir(parents=True)
    spec=provision.plan('CPU-QUOTA-02',target,16*1024**2)
    if case=='existing_identity':Path(spec['image']).parent.mkdir(parents=True)
    if case=='nonempty_target':(target/'original').write_text('preserve')
    run=Mock();monkeypatch.setattr(provision.subprocess,'run',run)
    with pytest.raises((ValueError,FileExistsError)):provision.apply(spec)
    assert not Path(spec['image']).exists()
    run.assert_not_called()
    if case=='nonempty_target':assert (target/'original').read_text()=='preserve'
