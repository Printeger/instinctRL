"""Real quota accounting and isolated CPU validation; mount/provenance are fixtures."""
import json
import sys
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools import r6_render_preparation as p
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import file_pin, RecoveryError

FREE=500*1024**3

@pytest.fixture
def ready(case,monkeypatch):
    b=case
    monkeypatch.setattr(p.storage.result_tool,'ATTEMPTS',b.store.root.resolve()/'attempts')
    b.description=dict(controller_key='fixture-controller',extraction=b.context,snapshot=b.context,
        config=b.context,source_snapshot=str(b.root),implementation={},master_authority=b.context,
        extension_inputs=b.context,stage_id='CPU',conditions=[],runner={},store_root=str(b.store.root.resolve()))
    monkeypatch.setattr(p,'describe',Mock(return_value=b.description))
    monkeypatch.setattr(p,'validate_frozen',lambda d,v,a:dict(parameters=v,budgets=p.contract.budget(v['wall_seconds']),source_identity={}))
    def inspect(j,key):
        _,_,_,cap=p.storage.validate(j,key)
        return dict(status='VOLUMES_VERIFIED_NOT_ACCEPTED',manifests=dict(native_volume=b.context,result_volume=b.context),output_cap_bytes=cap)
    b.inspect=inspect
    monkeypatch.setattr(p.volumes,'inspect',inspect)
    return b


def run(b):
    with b.j.writer():return p.prepare(b.j,'fixture-controller',b.store,free_disk_bytes=FREE)


def test_missing_volumes_resume_without_recharging_or_store_effects(ready,monkeypatch):
    b=ready;before=b.store.replay()
    monkeypatch.setattr(p.volumes,'inspect',lambda *a:dict(status='WAITING_FOR_ADMIN_PROVISION',missing=['fixture']))
    first=run(b);budget=b.j.read()['budget'];assert run(b)==first
    monkeypatch.setattr(p.volumes,'inspect',b.inspect)
    out=run(b);assert out['status']=='RENDER_INPUTS_READY_NOT_REVIEWED'
    assert b.j.read()['budget']==budget and b.store.replay()==before
    pin=b.j.read()['events'][out['event_key']]['preparation']
    with b.j.writer():assert p.describe_prepared(b.j,pin,b.store,free_disk_bytes=FREE)==out
    assert run(b)==out
    _,n,r,cap=p.storage.validate(b.j,out['storage_key'])
    assert n['maximum_backing_bytes']==1536*1024**2
    assert r['minimum_free_bytes']==1024**3 and cap==n['maximum_backing_bytes']+r['maximum_backing_bytes']+64*1024**2
    assert not Path(n['image']).exists() and not Path(r['image']).exists()


def test_input_drift_and_stop_prevent_reuse(ready):
    b=ready;run(b);b.description['stage_id']='CHANGED'
    with pytest.raises(RecoveryError,match='INPUT_DRIFT'):run(b)
    (b.j.directory/'STOP').touch()
    with pytest.raises(RecoveryError):run(b)


def test_frozen_failure_leaves_charged_plans_without_ready_receipt(ready,monkeypatch):
    b=ready;before=b.store.replay()
    monkeypatch.setattr(p,'validate_frozen',Mock(side_effect=RecoveryError('fixture-invalid-source')))
    with pytest.raises(RecoveryError,match='invalid-source'):run(b)
    entries=[e for e in b.j.read()['events'].values() if e['identity']['role']==p.ROLE]
    assert len(entries)==1 and 'preparation' not in entries[0] and b.store.replay()==before


def test_isolated_frozen_validator_executes_cpu_program(case,monkeypatch):
    b=case;root=b.root/'snapshot';root.mkdir()
    for relative,code in {
        'training_workloads/unified_h1/config.py':"def load_config():return {'resolved_identity':{'fixture':'isolated-cpu'}}\n",
        'tools/r6_render_contract.py':"def parameters(p,**kw):return p\ndef budget(w):return {'fixture_wall':w}\n",
    }.items():
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(code)
        for parent in path.parents:
            if parent==root:break
            (parent/'__init__.py').touch()
    config=b.write('frozen-config.json',dict(execution=dict(python_path=str(Path(sys.executable).resolve()),python_sha256=file_pin(Path(sys.executable).resolve())['sha256'])))
    receipt=b.write('frozen-receipt.json',dict(source_identity={'fixture':'isolated-cpu'}))
    monkeypatch.setattr(p.snapshot,'verify_frozen',Mock())
    monkeypatch.setattr(p.contract,'budget',lambda w:dict(fixture_wall=w))
    desc=dict(snapshot=receipt,config=config,source_snapshot=str(root))
    assert p.validate_frozen(desc,{'wall_seconds':20},b.root/'attempt')['source_identity']=={'fixture':'isolated-cpu'}
