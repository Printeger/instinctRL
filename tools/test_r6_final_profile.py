"""Frozen process transport fixtures; no runtime, GPU, Store or model call."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools import r6_final_profile as m
from tools.r6_recovery import file_pin,RecoveryError


@pytest.fixture
def case(tmp_path,monkeypatch):
    def pin(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return file_pin(p)
    exe=file_pin(sys.executable)
    config=pin('config.json',dict(execution=dict(python_path=exe['path'],python_sha256=exe['sha256'])))
    identity=dict(config_sha256=config['sha256'])
    receipt=pin('receipt.json',dict(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',
        config=config,snapshot=str(tmp_path),source_identity=identity))
    verify=Mock();run=Mock(return_value=SimpleNamespace(returncode=0,stderr='',
        stdout=json.dumps(dict(source_identity=identity,payload_bytes=4096))))
    monkeypatch.setattr(m,'verify_frozen',verify);monkeypatch.setattr(m.subprocess,'run',run)
    return SimpleNamespace(receipt=receipt,run=run,verify=verify)


def test_isolated_frozen_interpreter_and_exact_profile(case):
    draft=dict(final_selection={'fixture':True})
    assert m.validate(case.receipt,draft,4096)==4096
    case.verify.assert_called_once();args,kw=case.run.call_args
    assert args[0][1:4]==['-I','-B','-c'] and args[0][4]==m.PROGRAM
    assert kw['env']['CUDA_VISIBLE_DEVICES']=='' and kw['timeout']==60
    assert json.loads(kw['input'])==dict(profile=draft,payload_bytes=4096)


@pytest.mark.parametrize('fault',['source','failure','size','boolean'])
def test_frozen_failure_or_invalid_return_cannot_reserve_payload(case,fault):
    result=case.run.return_value
    if fault=='source':result.stdout=json.dumps(dict(source_identity={},payload_bytes=4096))
    if fault=='failure':result.returncode=1;result.stderr='capacity not accepted'
    if fault=='size':result.stdout='x'*65537
    if fault=='boolean':result.stdout=json.dumps(dict(source_identity=json.loads(result.stdout)['source_identity'],payload_bytes=True))
    with pytest.raises(RecoveryError):m.validate(case.receipt,{},4096)


def test_snapshot_failure_precedes_process(case):
    case.verify.side_effect=RecoveryError('inventory drift')
    with pytest.raises(RecoveryError,match='inventory drift'):m.validate(case.receipt,{},4096)
    case.run.assert_not_called()
