import copy
import json
from pathlib import Path
import unittest

from tools.r6_budget import BudgetError, reserve, totals


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.ledger = json.loads((Path(__file__).resolve().parents[1] /
            'docs/experiments/r6-robustness-auto/goal_budget.json').read_text())
        self.ledger['used'] = dict.fromkeys(self.ledger['caps'], 0)
        self.ledger['reservations'] = []
        self.key = dict(goal_id=self.ledger['goal_id'], event_id='review-01',
                        job_id='CPU-NO-JOB', input_sha256='a'*64, role='review')

    def reserve(self, resources, **kw):
        return reserve(self.ledger, event_key=self.key, resources=resources,
                       purpose=kw.get('purpose', 'development'),
                       free_disk_bytes=kw.get('free_disk_bytes', 500*1024**3))

    def test_exact_replay_is_idempotent_after_serialization(self):
        self.ledger = self.reserve({'automatic_model_sessions': 1})
        self.ledger = json.loads(json.dumps(self.ledger))
        again = self.reserve({'automatic_model_sessions': 1})
        self.assertEqual(totals(again)['automatic_model_sessions'], 1)
        with self.assertRaises(BudgetError): self.reserve({'automatic_model_sessions': 2})

    def test_confirmation_cannot_be_borrowed(self):
        with self.assertRaises(BudgetError): self.reserve({'training_transitions': 1500000})
        result = self.reserve({'training_transitions': 1572864}, purpose='confirmation_delivery')
        self.assertEqual(totals(result)['training_transitions'], 1572864)

    def test_hard_limit_even_for_confirmation(self):
        with self.assertRaises(BudgetError):
            self.reserve({'training_transitions': 3000001}, purpose='confirmation_delivery')

    def test_model_review_reserve_survives_confirmation(self):
        with self.assertRaises(BudgetError):
            self.reserve({'automatic_model_sessions': 25}, purpose='confirmation_delivery')

    def test_invalid_and_unknown_amounts(self):
        for amount in [-1, float('nan'), float('inf'), True, '1']:
            with self.assertRaises(BudgetError): self.reserve({'live_starts': amount})
        with self.assertRaises(BudgetError): self.reserve({'unaccounted': 1})

    def test_repairs_cannot_consume_final_review_slots(self):
        with self.assertRaises(BudgetError):
            self.reserve({'automatic_model_sessions': 29}, purpose='repair')

    def test_disk_floor_includes_reservations(self):
        with self.assertRaises(BudgetError):
            self.reserve({'output_actual_plus_unreleased_reservations_bytes': 1024**3},
                         free_disk_bytes=30*1024**3)

    def test_failed_check_does_not_mutate_ledger(self):
        before = copy.deepcopy(self.ledger)
        with self.assertRaises(BudgetError): self.reserve({'live_starts': 65})
        self.assertEqual(before, self.ledger)

    def test_event_identity_requires_all_dimensions(self):
        del self.key['role']
        with self.assertRaises(BudgetError): self.reserve({'automatic_model_sessions': 1})


if __name__ == '__main__': unittest.main()
