"""Actual Journal/Store/files; volume and frozen validators are explicit fixtures."""
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

import pytest
from rlflow.store import Store
from tools import r6_materialize_job as m
from tools.r6_recovery import Journal,file_pin,RecoveryError


@pytest.fixture
def case(tmp_path,monkeypatch):
    def pin(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return file_pin(path)
    j=Journal(tmp_path/'batch')
    budget=json.loads((Path(__file__).parent/'fixtures/r6_budget.json').read_text())
    with j.writer():j.initialize(budget)
    store=Store(tmp_path/'store');attempt=tmp_path/'attempt';(attempt/'results').mkdir(parents=True)
    config=pin('snapshot/training_workloads/unified_h1/candidate_config.json',{'fixture':True})
    receipt=pin('receipt.json',dict(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',
        snapshot=str(tmp_path/'snapshot'),config=config))
    native=pin('native.json',{'kind':'native'});result=pin('result.json',{'kind':'result'})
    runner={k:pin(k+'.json',{}) for k in ('query_executable','decision_executable','queue_review','sampler_review')}
    runner.update(purpose='development',sandbox='read-only')
    draft=pin('draft.json',dict(experiment_id='R6-MATERIALIZE-CPU',group='R1',wall_seconds=60))
    inputs=dict(snapshot_receipt=receipt,profile_draft=draft,storage_key='storage',
        stage_id='CPU',conditions=[dict(evidence_id='condition',sha256='1'*64)],runner=runner)
    input_pin=pin('inputs.json',inputs)
    allocation=dict(request={'experiment_id':'R6-MATERIALIZE-CPU'},purpose='development')
    validate=Mock(return_value=(allocation,{},dict(target=str(attempt/'results')),123456))
    inspect=Mock(return_value=dict(status='VOLUMES_VERIFIED_NOT_ACCEPTED',
        manifests=dict(native_volume=native,result_volume=result)))
    def prepare(snapshot_receipt,parameters_pin,target):
        params=json.loads(m.checked(parameters_pin).read_text())
        card=json.loads(m.checked(params['experiment_card']).read_text())
        assert card['profiles']['R1']=={k:v for k,v in params.items() if k!='experiment_card'}
        assert params['native_volume']==native and params['result_volume']==result
        assert params['output_cap_bytes']==123456
        return dict(status='PROPOSED_NOT_ISSUED',store_permission_issued=False,job_submitted=False,
                    authorization_proposal=dict(parameters=params,attempt=target))
    prepare_mock=Mock(side_effect=prepare)
    def build(config_pin,auth_pin,target,stage_id,conditions,root):
        return dict(id='R6-MATERIALIZE-CPU',backend='real',pins={'authorization':auth_pin},
                    h1={'attempt':target},stage_id=stage_id,conditions=conditions)
    build_mock=Mock(side_effect=build);preview=Mock();verify=Mock()
    monkeypatch.setattr(m.storage,'validate',validate)
    monkeypatch.setattr(m.volumes,'inspect',inspect)
    monkeypatch.setattr(m.authority,'prepare',prepare_mock)
    monkeypatch.setattr(m.authority,'verify_frozen',verify)
    monkeypatch.setattr(m.builder,'build',build_mock)
    monkeypatch.setattr(m.storage,'for_spec',preview)
    return SimpleNamespace(j=j,store=store,pin=pin,input_pin=input_pin,inputs=inputs,
        inspect=inspect,prepare=prepare_mock,build=build_mock,preview=preview,
        allocation=allocation,verify=verify,attempt=attempt)


def run(b):
    with b.j.writer():return m.materialize(b.j,b.input_pin,b.store,free_disk_bytes=500*1024**3)


def test_concrete_inputs_and_plan_replay_without_issuing_or_rebuilding(case):
    b=case;before=b.store.replay();first=run(b);budget=b.j.read()['budget']
    assert first['status']=='EXACT_INPUTS_MATERIALIZED_NOT_AUTHORIZED'
    assert not first['dispatch_authorized']
    plan=json.loads(m.checked(first['artifacts']['plan']).read_text())
    assert plan==dict(b.inputs['runner'],spec_pin=first['artifacts']['spec'])
    assert run(b)==first and b.j.read()['budget']==budget
    assert b.store.replay()==before
    b.prepare.assert_called_once();b.build.assert_called_once();b.verify.assert_called_once()
    assert all(c.kwargs.get('preview') is True for c in b.preview.call_args_list)


def test_missing_volumes_wait_then_resume_same_reservation(case):
    b=case;ready=b.inspect.return_value
    b.inspect.return_value=dict(status='WAITING_FOR_ADMIN_PROVISION',missing=['/fixture/volume.json'])
    waiting=run(b);budget=b.j.read()['budget']
    assert waiting['status']=='WAITING_FOR_ADMIN_PROVISION'
    assert not (b.j.directory/'materialized').exists()
    b.prepare.assert_not_called()
    b.inspect.return_value=ready
    assert run(b)['event_key']==waiting['event_key']
    assert b.j.read()['budget']==budget


@pytest.mark.parametrize('name',sorted(m.MANAGED))
def test_draft_cannot_override_controller_bound_values(case,name):
    b=case;value=json.loads(m.checked(b.inputs['profile_draft']).read_text());value[name]={}
    b.inputs['profile_draft']=b.pin('draft-invalid.json',value);b.input_pin=b.pin('inputs-invalid.json',b.inputs)
    before=b.j.read()
    with pytest.raises(RecoveryError,match='MANAGED_FIELDS'):run(b)
    assert b.j.read()==before;b.prepare.assert_not_called()


def test_storage_identity_and_purpose_must_match(case):
    b=case;b.allocation['purpose']='repair'
    with pytest.raises(RecoveryError,match='STORAGE_BINDING'):run(b)
    b.prepare.assert_not_called()


def test_frozen_failure_preserves_partial_inputs_and_charge(case):
    b=case;b.prepare.side_effect=RecoveryError('frozen validation failed')
    with pytest.raises(RecoveryError,match='frozen validation failed'):run(b)
    assert len(b.j.read()['budget']['reservations'])==1
    assert list((b.j.directory/'materialized').rglob('parameters.json'))
    b.build.assert_not_called();b.preview.assert_not_called()
    assert not b.store.replay()['jobs']


def test_crash_before_receipt_record_recovers_exact_artifacts(case,monkeypatch):
    b=case;save=b.j.save
    def crash(state):
        if any(e.get('materialization') for e in state['events'].values()):raise RuntimeError('receipt commit crash')
        save(state)
    monkeypatch.setattr(b.j,'save',crash)
    with pytest.raises(RuntimeError,match='receipt commit crash'):run(b)
    budget=b.j.read()['budget'];files={p.name:p.read_bytes() for p in (b.j.directory/'materialized').rglob('*.json')}
    monkeypatch.setattr(b.j,'save',save)
    first=run(b);assert run(b)==first
    assert b.j.read()['budget']==budget
    assert {p.name:p.read_bytes() for p in (b.j.directory/'materialized').rglob('*.json')}==files


def test_stop_after_slow_validation_stops_remaining_writes(case):
    b=case;prepare=b.prepare.side_effect
    def stop(*args):
        value=prepare(*args);(b.j.directory/'STOP').touch();return value
    b.prepare.side_effect=stop
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    b.build.assert_not_called()
    assert not list((b.j.directory/'materialized').rglob('authorization.json'))


def test_output_drift_or_missing_reservation_rejected(case):
    b=case;first=run(b)
    m.checked(first['artifacts']['plan']).write_text('{}')
    with pytest.raises(RecoveryError):run(b)
    with b.j.writer():
        state=b.j.read();state['budget']['reservations']=[];b.j.save(state)
    with pytest.raises(RecoveryError,match='RESERVATION'):run(b)
