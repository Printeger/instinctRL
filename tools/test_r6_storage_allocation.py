import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools import r6_storage_allocation as allocation
from tools.r6_recovery import Journal,file_pin,RecoveryError
from tools.r6_budget import totals


@pytest.fixture
def case(tmp_path,monkeypatch):
    monkeypatch.setattr(allocation.native_tool,'BASE',tmp_path/'native')
    monkeypatch.setattr(allocation.result_tool,'BASE',tmp_path/'results')
    monkeypatch.setattr(allocation.result_tool,'ATTEMPTS',tmp_path/'attempts')
    target=allocation.result_tool.ATTEMPTS/'R6-FIXTURE/results'
    native=allocation.native_tool.plan('J-'+'1'*16)
    result=allocation.result_tool.plan(native['volume_id'],target,1120*1024**2,1024**3,1024)
    def write(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return file_pin(p)
    n=write('native-plan.json',native);r=write('result-plan.json',result)
    j=Journal(tmp_path/'batch')
    with j.writer():j.initialize(json.loads((Path(__file__).parent/'fixtures/r6_budget.json').read_text()))
    return SimpleNamespace(j=j,n=n,r=r,native=native,result=result,write=write,target=target)


def register(b):
    with b.j.writer():return allocation.register(b.j,'R6-FIXTURE',b.n,b.r,purpose='development',free_disk_bytes=500*1024**3)


def spec(b,monkeypatch):
    pins={}
    for field,plan,tool in [('native_volume',b.native,allocation.native_tool),('result_volume',b.result,allocation.result_tool)]:
        path=Path(plan['image']).parent/'volume.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(plan,status='MOUNT_CREATED_NOT_ACCEPTED',implementation_sha256=file_pin(tool.__file__)['sha256'])))
        pins[field]=file_pin(path)
    cap=b.native['maximum_backing_bytes']+b.result['maximum_backing_bytes']+allocation.CONTROL_ALLOWANCE
    auth=b.write('authorization.json',dict(parameters=dict(pins,experiment_id='R6-FIXTURE',output_cap_bytes=cap)))
    value=dict(id='R6-FIXTURE',backend='real',h1=dict(mode='research-r6-evaluation',attempt=str(b.target.parent)),pins=dict(authorization=auth))
    monkeypatch.setattr('training_workloads.unified_h1.native_volume.validate',Mock())
    monkeypatch.setattr('training_workloads.unified_h1.result_volume.validate',Mock())
    return value,b.write('spec.json',value)


def test_reservation_precedes_mount_and_replay_does_not_charge_twice(case):
    b=case;key=register(b);before=b.j.read()
    assert not Path(b.native['image']).parent.exists() and not b.target.exists()
    assert register(b)==key and b.j.read()==before
    cap=b.native['maximum_backing_bytes']+b.result['maximum_backing_bytes']+allocation.CONTROL_ALLOWANCE
    assert totals(before['budget'])[allocation.DISK]==cap
    assert totals(before['budget'])['live_starts']==0


def test_binding_is_unique_and_preserves_storage_charge(case,monkeypatch):
    b=case;key=register(b);value,pin=spec(b,monkeypatch);before=copy.deepcopy(b.j.read()['budget'])
    with b.j.writer():first=allocation.bind(b.j,key,pin)
    (b.j.directory/'STOP').touch()
    with b.j.writer():assert allocation.bind(b.j,key,pin)==first
    value['id']='OTHER-EXACT-SPEC';other=b.write('other-spec.json',value)
    with b.j.writer(),pytest.raises(RecoveryError,match='ALREADY_BOUND'):allocation.bind(b.j,key,other)
    assert b.j.read()['budget']==before


@pytest.mark.parametrize('kind',['plan','charge','reservation'])
def test_drift_cannot_validate_as_prepaid_storage(case,kind):
    b=case;key=register(b)
    if kind=='plan':Path(b.n['path']).write_text('{}')
    else:
        with b.j.writer():
            s=b.j.read()
            if kind=='charge':s['events'][key]['resources'][allocation.DISK]=1
            else:s['budget']['reservations'][0]['resources'][allocation.DISK]=1
            b.j.save(s)
    with pytest.raises(RecoveryError):allocation.validate(b.j,key)


def test_same_volumes_cannot_be_reserved_under_another_experiment(case):
    b=case;register(b);before=b.j.read()
    with b.j.writer(),pytest.raises(RecoveryError,match='ALLOCATION_REUSE'):
        allocation.register(b.j,'R6-OTHER',b.n,b.r,purpose='development',free_disk_bytes=500*1024**3)
    assert b.j.read()==before


def test_stop_and_preexisting_volume_reject_before_charge(case):
    b=case;(b.j.directory/'STOP').touch()
    with pytest.raises(RecoveryError,match='STOP'):register(b)
    (b.j.directory/'STOP').unlink() # Isolated fixture only.
    Path(b.native['image']).parent.mkdir(parents=True)
    with pytest.raises(RecoveryError,match='PREEXISTING_VOLUME'):register(b)
    assert not b.j.read()['budget']['reservations']


def test_unbound_preview_preserves_ledger_and_then_actual_binding_succeeds(case,monkeypatch):
    b=case;key=register(b);value,pin=spec(b,monkeypatch);before=b.j.read()
    with b.j.writer():
        entry=allocation.for_spec(b.j,pin,purpose='development',preview=True)
        assert 'storage_binding' not in entry and b.j.read()==before
        with pytest.raises(RecoveryError,match='UNBOUND_OWNER'):
            allocation.for_spec(b.j,pin,purpose='development')
        bound=allocation.for_spec(b.j,pin,purpose='development',bind_new=True)
    assert bound['storage_binding']['spec']==pin
    assert b.j.read()['budget']==before['budget']


def test_preview_rejects_existing_owner_drift_without_modification(case,monkeypatch):
    b=case;key=register(b);value,pin=spec(b,monkeypatch)
    with b.j.writer():allocation.bind(b.j,key,pin)
    value['id']='OTHER';other=b.write('other.json',value);before=b.j.read()
    with b.j.writer(),pytest.raises(RecoveryError,match='ALREADY_BOUND'):
        allocation.for_spec(b.j,other,purpose='development',preview=True)
    assert b.j.read()==before
