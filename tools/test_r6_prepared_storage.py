"""Pre-spec pair checks use real prepaid ledger fixtures, mocked kernel checks."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.test_r6_storage_allocation import case,register,spec
from tools.r6_check_prepared_storage import inspect
from tools.r6_recovery import RecoveryError


def test_missing_pair_waits_without_ledger_mutation(case):
    key=register(case);before=case.j.path.read_bytes()
    result=inspect(case.j,key)
    assert result['status']=='WAITING_FOR_ADMIN_PROVISION' and len(result['missing'])==2
    assert result['capacity_pass'] is result['dispatch_authorized'] is False
    assert case.j.path.read_bytes()==before


def test_pair_checks_full_payload_before_spec_binding(case,monkeypatch):
    key=register(case);spec(case,monkeypatch);before=case.j.path.read_bytes()
    with patch('training_workloads.unified_h1.native_volume.validate',return_value={'free_bytes':10}) as native, \
         patch('training_workloads.unified_h1.result_volume.validate',return_value={'free_bytes':20}) as result:
        report=inspect(case.j,key)
    native.assert_called_once_with(report['manifests']['native_volume'])
    result.assert_called_once_with(report['manifests']['result_volume'],output_cap_bytes=report['output_cap_bytes'],
        expected_target=case.result['target'],minimum_free_bytes=case.result['minimum_free_bytes'])
    assert report['status']=='VOLUMES_VERIFIED_NOT_ACCEPTED'
    assert not report['capacity_pass'] and not report['dispatch_authorized']
    assert case.j.path.read_bytes()==before and not case.j.read()['events'][key].get('storage_binding')


def test_missing_one_volume_does_not_hide_bad_plan(case,monkeypatch):
    key=register(case);spec(case,monkeypatch)
    Path(case.result['image']).with_name('volume.json').unlink()
    path=Path(case.native['image']).with_name('volume.json');record=json.loads(path.read_text())
    record['implementation_sha256']='0'*64;path.write_text(json.dumps(record))
    with pytest.raises(RecoveryError,match='MANIFEST_PLAN'):inspect(case.j,key)


def test_full_or_invalid_mount_cannot_be_reported_ready(case,monkeypatch):
    key=register(case);spec(case,monkeypatch)
    with patch('training_workloads.unified_h1.result_volume.validate',side_effect=ValueError('RESULT_LAYOUT_FREE_SPACE')):
        with pytest.raises(ValueError,match='FREE_SPACE'):inspect(case.j,key)


def test_dangling_manifest_symlink_is_not_waiting(case):
    key=register(case);path=Path(case.native['image']).with_name('volume.json')
    path.parent.mkdir(parents=True);path.symlink_to(path.parent/'absent')
    with pytest.raises(RecoveryError,match='MANIFEST_PATH'):inspect(case.j,key)
