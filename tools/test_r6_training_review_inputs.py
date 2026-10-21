"""Exact review payload arithmetic/source bytes; frozen process transport is a fixture."""
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools import r6_training_review_inputs as m
from tools.test_r6_training_plan import case
from tools.test_r6_training_contract import compatibility,stage
from tools.r6_recovery import RecoveryError,file_pin
from training_workloads.unified_h1 import robustness_training_plan as p
from training_workloads.unified_h1 import robustness_training_contract as c


def test_recipe_draft_is_not_a_recipe_acceptance(case):
    b=case;assert p.validate_recipe(b.recipe)==b.recipe
    with pytest.raises(PermissionError):p.recipe(b.plan['recipe'],b.ev)
    b.recipe['confirmation_seeds']=[11,22,23]
    with pytest.raises(PermissionError,match='SEED_REUSE'):p.validate_recipe(b.recipe)


def test_capacity_payload_is_exact_reviewed_stage_content(stage):
    b,profile=stage
    expected=c.pinned_json(profile['capacity_review'])
    assert c.capacity_payload(b.plan)=={k:v for k,v in expected.items() if k!='independent_review'}
    profile['capacity_review']=b.pin('unreviewed.json',c.capacity_payload(b.plan))
    with pytest.raises(PermissionError,match='CAPACITY'):c.validate_parameters(c.MODE,profile)


def test_compatibility_payload_is_exact_but_not_accepted(compatibility):
    b,record,config=compatibility
    value=c.compatibility_payload(record['origin_config'],record['execution_config'],b.plan,config)
    assert value=={k:v for k,v in record.items() if k!='independent_review'}
    with pytest.raises(PermissionError,match='COMPATIBILITY_FIELDS'):
        c.validate_compatibility(b.pin('unreviewed.json',value),b.plan,config)
    value['changed_product_sources']={}
    with pytest.raises(PermissionError):c.validate_compatibility_payload(value,b.plan,config)


def test_recipe_frozen_program_checks_exact_source(case,monkeypatch):
    from training_workloads.unified_h1 import config
    b=case;cfg=dict(schema_version=config.R6_TRAINING_CONFIG_SCHEMA,resolved_identity=b.identity)
    monkeypatch.setattr(config,'load_config',lambda:cfg)
    request=dict(kind='recipe',recipe=b.plan['recipe'])
    def run():
        monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(request)));out=io.StringIO()
        monkeypatch.setattr(sys,'stdout',out);monkeypatch.setattr(sys,'argv',['fixture',str(__import__('pathlib').Path.cwd())])
        original=list(sys.path)
        try:exec(m.PROGRAM,{})
        finally:sys.path[:]=original
        return json.loads(out.getvalue())
    assert run()['payloads']==dict(recipe=b.recipe)
    cfg['resolved_identity']=dict(b.identity,config_sha256='0'*64)
    with pytest.raises(PermissionError,match='execution identity'):run()


@pytest.fixture
def transport(tmp_path,monkeypatch):
    def pin(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return file_pin(path)
    exe=file_pin(sys.executable);cp=pin('config.json',dict(execution=dict(python_path=exe['path'],python_sha256=exe['sha256'])))
    identity=dict(config_sha256=cp['sha256'])
    snapshot=pin('snapshot.json',dict(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',config=cp,
        snapshot=str(tmp_path),source_identity=identity))
    payload=pin('payload.json',{})
    run=Mock(return_value=SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(dict(source_identity=identity,payloads={'recipe':{}}))))
    monkeypatch.setattr(m,'verify_frozen',Mock());monkeypatch.setattr(m.subprocess,'run',run)
    return SimpleNamespace(snapshot=snapshot,payload=payload,run=run,identity=identity)


@pytest.mark.parametrize('kind',['recipe','stage'])
def test_transport_returns_only_pending_review_data(transport,kind):
    b=transport;request=dict(kind=kind)
    if kind=='recipe':request['recipe']=b.payload;payloads={'recipe':{}}
    else:request.update(plan=b.payload,origin_config=b.payload);payloads={'capacity':{},'compatibility':{}}
    b.run.return_value.stdout=json.dumps(dict(source_identity=b.identity,payloads=payloads))
    result=m.prepare(b.snapshot,request)
    assert result['status']=='REVIEW_PAYLOADS_PREPARED_NOT_ACCEPTED' and not result['dispatch_authorized']
    args,kw=b.run.call_args
    assert args[0][1:4]==['-I','-B','-c'] and kw['env']['CUDA_VISIBLE_DEVICES']==''
    assert json.loads(kw['input'])==request and kw['timeout']==60


@pytest.mark.parametrize('fault',['failure','identity','review','size','fields'])
def test_frozen_transport_failure_is_not_reviewable(transport,fault):
    b=transport;request=dict(kind='recipe',recipe=b.payload)
    result=b.run.return_value
    if fault=='failure':result.returncode=1
    if fault=='identity':result.stdout=json.dumps(dict(source_identity={},payloads={'recipe':{}}))
    if fault=='review':result.stdout=json.dumps(dict(source_identity=b.identity,payloads={'recipe':{'independent_review':b.payload}}))
    if fault=='size':result.stdout='x'*65537
    if fault=='fields':request['approval']=True
    with pytest.raises(RecoveryError):m.prepare(b.snapshot,request)
    if fault=='fields':b.run.assert_not_called()


def test_stage_frozen_program_builds_actual_capacity_and_source_payloads(compatibility,tmp_path,monkeypatch):
    from tools.test_r6_training_plan import update_recipe
    from training_workloads.unified_h1 import config as config_module
    b,record,config=compatibility
    config['schema_version']=config_module.R6_TRAINING_CONFIG_SCHEMA
    snapshot=tmp_path/'snapshot';path=snapshot/'training_workloads/unified_h1/candidate_config.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({k:v for k,v in config.items() if k!='resolved_identity'}))
    config['resolved_identity']=dict(config['resolved_identity'],config_sha256=file_pin(path)['sha256'])
    b.recipe['source_identity']=config['resolved_identity'];update_recipe(b)
    request=dict(kind='stage',plan=b.pin('ready-plan.json',b.plan),origin_config=record['origin_config'])
    monkeypatch.setattr(config_module,'load_config',lambda:config)
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(request)));out=io.StringIO()
    monkeypatch.setattr(sys,'stdout',out);monkeypatch.setattr(sys,'argv',['fixture',str(snapshot)])
    original=list(sys.path)
    try:exec(m.PROGRAM,{})
    finally:sys.path[:]=original
    value=json.loads(out.getvalue())
    assert value['source_identity']==config['resolved_identity']
    assert value['payloads']['capacity']==c.capacity_payload(b.plan)
    expected=c.compatibility_payload(record['origin_config'],file_pin(path),b.plan,config)
    assert value['payloads']['compatibility']==expected
    assert not any('independent_review' in payload for payload in value['payloads'].values())
