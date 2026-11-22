import copy
import json
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
