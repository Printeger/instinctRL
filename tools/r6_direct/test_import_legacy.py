import copy
import unittest
from tools.r6_direct.import_legacy import build


class ImportTests(unittest.TestCase):
    def state(self):
        return {'budget':dict(caps={'output':100},used={'output':7},
            reservations=[dict(event_key='one',purpose='development',resources={'output':10},
                               observations=[{'resources':{'output':12}},{'resources':{'output':3}}])],
            protected_confirmation_delivery_repair={'output':40},protected_model_sessions={})}

    def test_reservations_and_overruns_are_not_refunded_or_double_counted(self):
        state=self.state(); original=copy.deepcopy(state); result=build(state)
        self.assertEqual(result['historical_charged']['output'],19)
        self.assertEqual(result['remaining_before_new_work']['output'],81)
        self.assertEqual(result['reservation_mapping'][0]['original'],state['budget']['reservations'][0])
        self.assertEqual(result['protected_confirmation_delivery_repair'],{'output':40})
        self.assertEqual(state,original)

    def test_duplicate_and_over_cap_rejected(self):
        state=self.state(); state['budget']['reservations']*=2
        with self.assertRaises(ValueError): build(state)
        state=self.state(); state['budget']['used']['output']=99
        with self.assertRaises(ValueError): build(state)


if __name__=='__main__': unittest.main()
