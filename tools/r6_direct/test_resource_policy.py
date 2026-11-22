import copy
import json
from pathlib import Path
import pytest
from tools.r6_direct.run import reserve
from tools.r6_evidence import file_pin


def test_authorized_unlimited_preserves_accounting_and_duplicate_guard(tmp_path):
    source=Path('docs/experiments/r6-direct-training/continuation-20261101/RESOURCE_POLICY_USER_AMENDMENT.json')
    authority=tmp_path/'authority.json'
    authority.write_bytes(source.read_bytes())
    policy=file_pin(authority)
    ledger=dict(caps={'x':100},historical_charged={'x':90},new_used={'x':8},
                protected_confirmation_delivery_repair={'x':20},new_reservations=[])
    original=copy.deepcopy(ledger)
    with pytest.raises(ValueError,match='PROTECTED_BUDGET'):
        reserve(copy.deepcopy(ledger),'old',{'x':1000})
    reserve(ledger,'new',{'x':1000},resource_policy=policy)
    for key in original.keys()-{'new_reservations'}:
        assert ledger[key]==original[key]
    assert ledger['new_reservations'][0]['resource_policy']==policy
    with pytest.raises(ValueError,match='DUPLICATE'):
        reserve(ledger,'new',{'x':1},resource_policy=policy)
    with pytest.raises(ValueError,match='RESOURCE_AMOUNT'):
        reserve(ledger,'negative',{'x':-1},resource_policy=policy)
    authority.write_text('{}')
    with pytest.raises(RuntimeError):
        reserve(ledger,'drift',{'x':1},resource_policy=policy)
    with pytest.raises(ValueError,match='RESOURCE_POLICY'):
        reserve(ledger,'invalid',{'x':1},resource_policy=file_pin(authority))


def test_third_requires_own_authority_ledger_and_elapsed_window(tmp_path, monkeypatch):
    from tools.r6_direct import import_legacy
    from tools.r6_direct.resource_policy import (
        THIRD_CAPS, THIRD_SCHEMA, check_elapsed, load_run_ledger, third_predecessor,
    )

    run_id = 'R6-DIRECT-THIRD-PROBE'
    old_authority = Path(
        'docs/experiments/r6-direct-training/continuation-20261101/RESOURCE_POLICY_USER_AMENDMENT.json'
    )
    with pytest.raises(ValueError, match='DIRECT_THIRD_RESOURCE_POLICY'):
        load_run_ledger(run_id, file_pin(old_authority))
    with pytest.raises(ValueError, match='AUTHORITY_REQUIRED'):
        load_run_ledger(run_id, None)

    statement = tmp_path / 'user.txt'
    statement.write_text('Fixture for a finite resource allowance.')
    caps = dict.fromkeys(THIRD_CAPS, 100)
    caps['output_actual_plus_unreleased_reservations_bytes'] = 0
    scope = tmp_path / 'scope.json'
    scope.write_text(json.dumps(dict(schema_version=THIRD_SCHEMA,
                                     user_authority=file_pin(statement), caps=caps, elapsed_seconds=100)))
    scope_pin = file_pin(scope)
    monkeypatch.setattr(import_legacy, 'ROOT', tmp_path)
    ledger_file = tmp_path / 'docs/experiments/third-training/RESOURCE_LEDGER.json'
    ledger_file.parent.mkdir(parents=True)
    zero = dict.fromkeys(THIRD_CAPS, 0)
    ledger = dict(resource_policy=scope_pin, caps=caps, historical_charged=zero, new_used=zero,
                  protected_confirmation_delivery_repair=zero, new_reservations=[])
    ledger_file.write_text(json.dumps(ledger))
    path, loaded, imported, policy = load_run_ledger(run_id, scope_pin)
    assert path == ledger_file and imported is None and loaded == ledger
    assert check_elapsed(ledger, policy, 90, now=500) == 500
    ledger['first_live_unix'] = 500
    with pytest.raises(ValueError, match='ELAPSED_LIMIT'):
        check_elapsed(ledger, policy, 90, now=511)
    with pytest.raises(ValueError, match='PROTECTED_BUDGET'):
        reserve(ledger, run_id, {'allocated_gpu_seconds': 101})
    with pytest.raises(ValueError, match='PROTECTED_BUDGET'):
        reserve(ledger, run_id, {'output_actual_plus_unreleased_reservations_bytes': 1})
    reserve(ledger, run_id, {'allocated_gpu_seconds': 90})
    with pytest.raises(ValueError, match='THIRD_PREDECESSOR'):
        third_predecessor(ledger, None)
    ledger['new_reservations'][0]['receipt'] = file_pin(statement)
    third_predecessor(ledger, file_pin(statement))
