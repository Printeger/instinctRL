import copy
import json
from pathlib import Path
import unittest
from training_workloads.unified_h1.direct_contract import validate_runtime_contract

ROOT=Path(__file__).resolve().parents[2]


class BaselineContractTests(unittest.TestCase):
    def setUp(self):
        self.contract=json.loads((ROOT/'docs/experiments/r6-direct-training/BASELINE_CONTRACT.json').read_text())

    def test_exact_frozen_baseline_has_all_samples(self):
        result=validate_runtime_contract(self.contract)
        self.assertEqual(result['budgets']['nontraining_transitions'],193600)
        self.assertEqual(result['budgets']['explicit_resets'],21)
        self.assertEqual(result['execution_mode'],'local-direct')
        self.assertNotIn('authorization_sha256',result)
        self.assertEqual(len(result['parameters']['r6']['block_ids']),16)

    def test_budget_sample_seed_and_pin_changes_rejected(self):
        changes=[('budget',lambda c:c['budgets'].__setitem__('nontraining_transitions',193568)),
                 ('blocks',lambda c:c['parameters']['r6']['block_ids'].pop()),
                 ('stream',lambda c:c['parameters']['r6'].__setitem__('seed',2026101902)),
                 ('holdout',lambda c:c['parameters']['r6'].__setitem__('split','final')),
                 ('model',lambda c:c['parameters']['baseline_bindings'].__setitem__('sha256','0'*64)),
                 ('protocol',lambda c:c['parameters']['r6']['protocol'].__setitem__('sha256','0'*64))]
        for name,edit in changes:
            with self.subTest(name=name):
                value=copy.deepcopy(self.contract); edit(value)
                with self.assertRaises((ValueError,PermissionError)):validate_runtime_contract(value)

    def test_legacy_approval_fields_are_not_accepted(self):
        value=copy.deepcopy(self.contract);value['authorization_sha256']='0'*64
        with self.assertRaises(ValueError):validate_runtime_contract(value)


if __name__=='__main__':unittest.main()
