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
