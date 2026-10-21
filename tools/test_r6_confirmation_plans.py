"""Frozen plan transport and PROGRAM tests; no worker, GPU or reservation."""
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools import r6_confirmation_plans as m
from tools import r6_training_review_inputs as reviews
from tools.test_r6_confirmation import frozen,ready
from tools.test_r6_training_plan import case
from tools.r6_recovery import file_pin,RecoveryError


@pytest.mark.parametrize('program',[m.PROGRAM,reviews.PROGRAM])
def test_frozen_program_checks_source_and_returns_exact_plans_or_freeze(frozen,monkeypatch,program):
    from training_workloads.unified_h1 import config
    b=frozen;cfg=dict(schema_version=config.R6_TRAINING_CONFIG_SCHEMA,resolved_identity=b.identity)
    monkeypatch.setattr(config,'load_config',lambda:cfg)
    request=b.arguments if program==m.PROGRAM else dict(kind='freeze',recipe=b.plan['recipe'],recipe_review=b.plan['recipe_review'],selection_evidence=b.write('evidence.json',b.evidence))
    def execute():
        out=io.StringIO();monkeypatch.setattr(sys,'stdout',out);monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(request)))
        monkeypatch.setattr(sys,'argv',['fixture',str(__import__('pathlib').Path.cwd())]);before=list(sys.path)
        try:exec(program,{})
        finally:sys.path[:]=before
        return json.loads(out.getvalue())
    value=execute();assert value['source_identity']==b.identity
    if program==m.PROGRAM:assert len(value['bundle']['plans'])==3 and not value['bundle']['dispatch_authorized']
    else:assert value['payloads']['freeze']['confirmation_seeds']==[21,22,23] and 'independent_review' not in value['payloads']['freeze']
    cfg['resolved_identity']=dict(b.identity,config_sha256='0'*64)
    with pytest.raises(PermissionError,match='execution identity'):execute()


@pytest.fixture
def transport(tmp_path,monkeypatch):
    def pin(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return file_pin(p)
    exe=file_pin(sys.executable);cp=pin('config.json',dict(execution=dict(python_path=exe['path'],python_sha256=exe['sha256'])))
    identity=dict(config_sha256=cp['sha256']);snapshot=pin('snapshot.json',dict(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',config=cp,snapshot=str(tmp_path),source_identity=identity))
    dummy=pin('dummy.json',{})
    request=dict(freeze_pin=dummy,recipe_review=dummy,baseline_bindings=dummy,experiment_ids=['R6-A','R6-B','R6-C'],wall_seconds=60,storage={})
    bundle=dict(schema_version='r6-three-confirmation-plans/1',freeze=dummy,baseline_bindings=dummy,
        plans=[dict(experiment_id=x) for x in request['experiment_ids']],scientific_acceptance=False,dispatch_authorized=False,holdout_released=False)
    run=Mock(return_value=SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(dict(source_identity=identity,bundle=bundle))))
    verify=Mock();monkeypatch.setattr(m,'verify_frozen',verify);monkeypatch.setattr(m.subprocess,'run',run)
    return SimpleNamespace(snapshot=snapshot,request=request,run=run,verify=verify,identity=identity,bundle=bundle)


def test_transport_checks_frozen_snapshot_and_bounded_cpu_process(transport):
    b=transport;value=m.prepare(b.snapshot,b.request)
    assert value['status']=='THREE_CONFIRMATION_PLANS_PREPARED_NOT_RESERVED' and not value['dispatch_authorized']
    args,kw=b.run.call_args;assert args[0][1:4]==['-I','-B','-c']
    assert kw['timeout']==180 and kw['env']['CUDA_VISIBLE_DEVICES']=='' and json.loads(kw['input'])==b.request
    b.verify.assert_called_once()


@pytest.mark.parametrize('fault',['process','source','plans','acceptance','cap','fields'])
def test_transport_rejects_unbound_or_failed_result(transport,fault):
    b=transport
    if fault=='process':b.run.return_value.returncode=1
    if fault=='source':b.identity={}
    if fault=='plans':b.bundle['plans'].pop()
    if fault=='acceptance':b.bundle['scientific_acceptance']=True
    if fault=='fields':b.request['authorization']=True
    b.run.return_value.stdout='x'*65537 if fault=='cap' else json.dumps(dict(source_identity=b.identity,bundle=b.bundle))
    with pytest.raises(RecoveryError):m.prepare(b.snapshot,b.request)
    if fault=='fields':b.run.assert_not_called()
