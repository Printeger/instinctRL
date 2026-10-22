"""Closed-new-run accounting retains reservations and rejects double settlement."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.r6_direct.import_legacy import pin
from tools.r6_direct.settle import settle_time

class SettlementTests(unittest.TestCase):
    def test_exact_closed_time_and_no_double_refund(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'receipt.json'
            p.write_text(json.dumps(dict(status='CLOSED',remaining=[],parent={'pid':123},owned_processes=[],wall_seconds=12.2)))
            resources=dict(live_wall_seconds=100,allocated_gpu_seconds=100,nontraining_transitions=500,live_starts=1)
            ledger=dict(historical_charged={'live_wall_seconds':200},new_used={'live_wall_seconds':0},new_reservations=[dict(id='R6-DIRECT-BASELINE-CPU',status='UNRELEASED',resources=resources.copy(),receipt=pin(p))])
            with patch('tools.r6_direct.settle.process_identity',return_value=None):settle_time(ledger,'R6-DIRECT-BASELINE-CPU')
            row=ledger['new_reservations'][0]
            self.assertEqual(row['resources']['live_wall_seconds'],13)
            self.assertEqual(row['resources']['nontraining_transitions'],500)
            self.assertEqual(row['settlement']['original_resources'],resources)
            self.assertEqual(ledger['historical_charged'],{'live_wall_seconds':200})
            before=copy.deepcopy(ledger)
            with self.assertRaisesRegex(ValueError,'ALREADY_SETTLED'):settle_time(ledger,'R6-DIRECT-BASELINE-CPU')
            self.assertEqual(before,ledger)

    def test_live_or_changed_receipt_not_refunded(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'receipt.json';owner={'pid':123}
            p.write_text(json.dumps(dict(status='CLOSED',remaining=[],parent=owner,owned_processes=[],wall_seconds=1)))
            ledger=dict(new_reservations=[dict(id='R6-DIRECT-BASELINE-CPU',status='UNRELEASED',resources={'live_wall_seconds':100,'allocated_gpu_seconds':100},receipt=pin(p))])
            before=copy.deepcopy(ledger)
            with patch('tools.r6_direct.settle.process_identity',return_value=owner):
                with self.assertRaisesRegex(ValueError,'SETTLEMENT_LIVE'):settle_time(ledger,'R6-DIRECT-BASELINE-CPU')
            self.assertEqual(ledger,before)
            p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'RECEIPT_DRIFT'):settle_time(ledger,'R6-DIRECT-BASELINE-CPU')
            self.assertEqual(ledger,before)

if __name__=='__main__':unittest.main()
