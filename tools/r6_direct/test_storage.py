import copy
import json
from pathlib import Path
import pytest
from tools.r6_evidence import file_pin
from tools.r6_direct.storage import fresh_binding


@pytest.fixture
def binding(tmp_path):
    spec={};plan=dict(storage_reservation_id='fresh',run_id='R6-DIRECT-NEW')
    for kind,key in (('native','native_dir'),('result','attempt_dir')):
        record=dict(volume_id='new',target=str(tmp_path/kind),image=str(tmp_path/(kind+'.ext4')),
                    maximum_backing_bytes=1024,owner_uid=1000,owner_gid=1000)
        spec[kind]=record
        path=tmp_path/(kind+'.json');path.write_text(json.dumps(record))
        plan[kind+'_volume']=file_pin(path);plan[key]=str(tmp_path/kind/plan['run_id'])
    path=tmp_path/'plan.json';path.write_text(json.dumps(spec))
    ledger=dict(new_reservations=[dict(id='fresh',status='UNRELEASED',storage_plan=file_pin(path),
            resources=dict(output_actual_plus_unreleased_reservations_bytes=2048))])
    contract=dict(parameters=dict(output_cap_bytes=2048+64*1024**2))
    return plan,ledger,contract


def test_new_storage_binding_preserves_ledger(binding):
    plan,ledger,contract=binding;before=copy.deepcopy(ledger)
    assert fresh_binding(plan,ledger,contract)=='fresh'
    assert ledger==before


def test_system_result_path_and_capacity(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.r6_direct import storage

    root=tmp_path/'instinctrl-runs'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path,'home',classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(storage,'SYSTEM_ROOT',root)
    attempt=root/'R6-DIRECT-THIRD-FORMAL-PROBE'
    cap=1024**3
    plan=dict(run_id=attempt.name,attempt_dir=str(attempt),output_root=str(attempt/'control'),
              result_payload_cap_bytes=cap)
    contract=dict(parameters=dict(output_cap_bytes=cap))
    floor=storage.SYSTEM_FLOOR
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=floor+cap))
    with pytest.raises(ValueError,match='SYSTEM_DISK_FLOOR'):
        storage.system_result(plan,contract,before_launch=True)
    assert storage.system_result(plan,contract)['payload_cap_bytes']==cap
    attempt.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='SYSTEM_PATH'):
        storage.system_paths(plan)


def test_system_native_in_run_directory_and_counted_once(tmp_path, monkeypatch):
    from tools.r6_direct import storage

    root = tmp_path / 'instinctrl-runs'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(storage, 'SYSTEM_ROOT', root)
    attempt = root / 'R6-DIRECT-THIRD-FORMAL-PROBE'
    native = attempt / 'native'
    plan = dict(run_id=attempt.name, attempt_dir=str(attempt), output_root=str(attempt / 'control'),
                system_results=True, system_native=True, native_dir=str(native),
                storage_reservation_id='ordinary', result_payload_cap_bytes=1024**3)
    assert storage.system_native(plan) == native
    spec = tmp_path / 'ordinary.json'
    spec.write_text(json.dumps(dict(schema_version=storage.SYSTEM_STORAGE_SCHEMA, root=str(root),
                                    new_volume_allocation_bytes=0)))
    ledger = dict(new_reservations=[dict(id='ordinary', status='UNRELEASED', storage_plan=file_pin(spec),
                                         resources=dict(output_actual_plus_unreleased_reservations_bytes=0))])
    contract = dict(parameters=dict(output_cap_bytes=1024**3))
    assert fresh_binding(plan, ledger, contract, third=True) == 'ordinary'
    with pytest.raises(ValueError, match='ORDINARY_SYSTEM_STORAGE'):
        storage.system_native(dict(plan, native_volume={'path': '/old/volume.json'}))
    with pytest.raises(ValueError, match='SYSTEM_NATIVE_PATH'):
        storage.system_native(dict(plan, native_dir=str(root / 'other')))
    attempt.mkdir(mode=0o700)
    native.mkdir(mode=0o700)
    (native / 'settings.json').write_text('{}')
    assert storage.system_native(plan) == native
    assert storage.evidence_bytes(attempt, native) == storage.evidence_bytes(attempt)
    native.chmod(0o755)
    with pytest.raises(ValueError, match='SYSTEM_NATIVE_PERMISSIONS'):
        storage.system_native(plan)


def test_existing_backing_is_bound_without_new_allocation(binding):
    from pathlib import Path

    plan,ledger,contract=binding
    plan['run_id']='R6-DIRECT-THIRD-PROBE'
    plan.update(result_payload_cap_bytes=1024,native_minimum_free_bytes=1024)
    row=ledger['new_reservations'][0]
    path=Path(row['storage_plan']['path'])
    spec=json.loads(path.read_text())
    spec.update(schema_version='instinctrl-third-storage-reuse/1',new_volume_allocation_bytes=0)
    for kind,key in (('native','native_dir'),('result','attempt_dir')):
        spec[kind]['manifest']=plan[kind+'_volume']
        plan[key]=str(Path(spec[kind]['target'])/plan['run_id'])
    path.write_text(json.dumps(spec))
    row['storage_plan']=file_pin(path)
    row['resources']['output_actual_plus_unreleased_reservations_bytes']=0
    assert fresh_binding(plan,ledger,contract,third=True)=='fresh'
    with pytest.raises(ValueError,match='STORAGE_REUSE_SCOPE'):
        fresh_binding(plan,ledger,contract)
    plan.pop('result_payload_cap_bytes')
    with pytest.raises(ValueError,match='STORAGE_REUSE_SCOPE'):
        fresh_binding(plan,ledger,contract,third=True)


@pytest.mark.parametrize('fault',['missing','undercharged','destination','cap','manifest','hash'])
def test_new_binding_rejects_drift(binding,fault):
    plan,ledger,contract=binding
    if fault=='missing':ledger['new_reservations']=[]
    if fault=='undercharged':ledger['new_reservations'][0]['resources']['output_actual_plus_unreleased_reservations_bytes']=0
    if fault=='destination':plan['attempt_dir']+='/wrong'
    if fault=='cap':contract['parameters']['output_cap_bytes']-=1
    if fault in ('manifest','hash'):
        from pathlib import Path
        path=Path(plan['result_volume']['path']);record=json.loads(path.read_text());record['image']='/wrong';path.write_text(json.dumps(record))
        if fault=='manifest':plan['result_volume']=file_pin(path)
    with pytest.raises((ValueError,RuntimeError)):fresh_binding(plan,ledger,contract)
