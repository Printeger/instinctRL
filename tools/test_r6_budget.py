import copy
import json
from pathlib import Path
import unittest

from tools.r6_budget import BudgetError, reserve, totals, observe


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.ledger = json.loads((Path(__file__).resolve().parents[1] /
            'tools/fixtures/r6_budget.json').read_text())
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

    def test_unclassified_used_sessions_preserve_combined_reserve(self):
        self.ledger['used']['automatic_model_sessions']=10
        self.ledger=json.loads(json.dumps(self.ledger))
        with self.assertRaises(BudgetError):
            self.reserve({'automatic_model_sessions':15},purpose='confirmation_delivery')

    def test_disk_floor_includes_reservations(self):
        with self.assertRaises(BudgetError):
            self.reserve({'output_actual_plus_unreleased_reservations_bytes': 1024**3},
                         free_disk_bytes=30*1024**3)

    def test_failed_check_does_not_mutate_ledger(self):
        before = copy.deepcopy(self.ledger)
        with self.assertRaises(BudgetError): self.reserve({'live_starts': 65})
        self.assertEqual(before, self.ledger)

    def observed(self, values, digest='b'*64):
        return observe(self.ledger,event_key=self.key,resources=values,
            evidence={'path':'/CPU_ONLY/evidence.json','sha256':digest})

    def test_observation_keeps_reserve_and_charges_only_excess(self):
        self.ledger=self.reserve({'training_transitions':100})
        self.ledger=self.observed({'training_transitions':40})
        self.assertEqual(totals(self.ledger)['training_transitions'],100)
        self.ledger=self.observed({'training_transitions':130},'c'*64)
        self.assertEqual(totals(self.ledger)['training_transitions'],130)
        self.ledger=self.observed({'training_transitions':10},'d'*64)
        self.assertEqual(totals(self.ledger)['training_transitions'],130)
        self.assertEqual(self.ledger['reservations'][0]['resources'],{'training_transitions':100})
        again=self.reserve({'training_transitions':100})
        self.assertEqual(again,self.ledger)

    def test_observation_replay_idempotent_but_drift_rejected(self):
        self.ledger=self.reserve({'live_starts':1})
        self.ledger=self.observed({'live_starts':1})
        self.assertEqual(self.ledger,self.observed({'live_starts':1}))
        with self.assertRaisesRegex(BudgetError,'replay drift'):self.observed({'live_starts':2})

    def test_actual_cap_excess_is_preserved_and_future_reservation_refused(self):
        self.ledger=self.reserve({'training_transitions':100})
        self.ledger=self.observed({'training_transitions':3000001})
        self.assertEqual(self.ledger['observed_cap_excess'],{'training_transitions':1})
        with self.assertRaisesRegex(BudgetError,'Cap exceeded'):totals(self.ledger)
        self.key=dict(self.key,event_id='next')
        with self.assertRaisesRegex(BudgetError,'Cap exceeded'):self.reserve({'automatic_model_sessions':1})

    def test_observed_development_spend_cannot_borrow_confirmation(self):
        self.ledger=self.reserve({'training_transitions':100})
        self.ledger=self.observed({'training_transitions':1427136})
        self.key=dict(self.key,event_id='next')
        with self.assertRaisesRegex(BudgetError,'Protected capacity'):self.reserve({'training_transitions':1})

    def test_observed_sessions_preserve_review_and_repair(self):
        self.ledger=self.reserve({'automatic_model_sessions':1},purpose='confirmation_delivery')
        self.ledger=self.observed({'automatic_model_sessions':24})
        self.key=dict(self.key,event_id='next')
        with self.assertRaisesRegex(BudgetError,'Protected review/repair'):
            self.reserve({'automatic_model_sessions':1},purpose='confirmation_delivery')

    def test_observations_need_exact_reservation_and_valid_values(self):
        with self.assertRaisesRegex(BudgetError,'exact reservation'):self.observed({'live_starts':1})
        self.ledger=self.reserve({'live_starts':1})
        for invalid in (True,float('nan'),-1):
            with self.assertRaises(BudgetError):self.observed({'live_starts':invalid})
        with self.assertRaises(BudgetError):self.observed({'unknown':1})

    def test_event_identity_requires_all_dimensions(self):
        del self.key['role']
        with self.assertRaises(BudgetError): self.reserve({'automatic_model_sessions': 1})


if __name__ == '__main__': unittest.main()
