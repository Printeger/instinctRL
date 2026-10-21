"""Frozen compatibility transport; actual payload/source validation covered separately."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools import r6_candidate_review_inputs as m
from tools.r6_recovery import RecoveryError,file_pin


@pytest.fixture
def transport(tmp_path,monkeypatch):
    def pin(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return file_pin(path)
    exe=file_pin(sys.executable);cp=pin('config.json',dict(execution=dict(python_path=exe['path'],python_sha256=exe['sha256'])))
    identity=dict(config_sha256=cp['sha256'])
    snapshot=pin('snapshot.json',dict(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',config=cp,
        snapshot=str(tmp_path),source_identity=identity))
    payload=pin('payload.json',{})
    run=Mock(return_value=SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(dict(source_identity=identity,payload={}))))
    monkeypatch.setattr(m,'verify_frozen',Mock());monkeypatch.setattr(m.subprocess,'run',run)
    return SimpleNamespace(snapshot=snapshot,payload=payload,run=run,identity=identity)


@pytest.mark.parametrize('kind',['selection','compatibility'])
def test_transport_never_accepts_or_issues(transport,kind):
    b=transport;request=dict(kind=kind,selection=b.payload)
    if kind=='compatibility':request['origin_config']=b.payload
    r=m.prepare(b.snapshot,request)
    assert r['status']=='REVIEW_PAYLOAD_PREPARED_NOT_ACCEPTED' and not r['dispatch_authorized']
    args,kw=b.run.call_args
    assert args[0][1:4]==['-I','-B','-c'] and kw['env']['CUDA_VISIBLE_DEVICES']==''
    assert json.loads(kw['input'])==request and kw['timeout']==60


@pytest.mark.parametrize('fault',['failure','identity','review','size','fields'])
def test_frozen_failure_rejects_review_material(transport,fault):
    b=transport;request=dict(selection=b.payload,origin_config=b.payload);result=b.run.return_value
    if fault=='failure':result.returncode=1
    if fault=='identity':result.stdout=json.dumps(dict(source_identity={},payload={}))
    if fault=='review':result.stdout=json.dumps(dict(source_identity=b.identity,payload={'independent_review':b.payload}))
    if fault=='size':result.stdout='x'*65537
    if fault=='fields':request['approval']=True
    with pytest.raises(RecoveryError):m.prepare(b.snapshot,request)
    if fault=='fields':b.run.assert_not_called()
