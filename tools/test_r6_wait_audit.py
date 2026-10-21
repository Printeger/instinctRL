"""Actual local CPU child and Journal; no simulation or paid model."""
from unittest.mock import patch
from tools.test_r6_job_handoff import JobTests
from tools import r6_wait_audit as w
from tools.r6_recovery import RecoveryError


class WaitAuditTests(JobTests):
    def test_complete_local_window_has_no_platform_claim(self):
        with patch('tools.r6_job_handoff.time.sleep',side_effect=lambda _:self.finish()):self.handoff()
        value=self.j.read()['events'][self.key]['local_wait_audit']
        self.assertTrue(value['complete']);self.assertFalse(value['platform_inactivity_proven'])
        self.assertFalse(value['unobserved_restart_gap']);self.assertGreaterEqual(value['segments'][0]['checks'],3)
        self.handoff();self.assertEqual(value,self.j.read()['events'][self.key]['local_wait_audit'])

    def test_restart_retains_unobserved_gap(self):
        with self.j.writer(),patch.object(w,'identity',return_value={'fixture':'prior-controller'}):w.record(self.j,self.key)
        with patch('tools.r6_job_handoff.time.sleep',side_effect=lambda _:self.finish()):self.handoff()
        value=self.j.read()['events'][self.key]['local_wait_audit']
        self.assertTrue(value['complete']);self.assertTrue(value['unobserved_restart_gap'])
        self.assertEqual(len(value['segments']),2)

    def test_changed_platform_accounting_blocks_false_zero_call_window(self):
        with self.j.writer():
            w.record(self.j,self.key);state=self.j.read();state['platform_automatic_wake_accounting']={'count':1};self.j.save(state)
        with self.assertRaisesRegex(RecoveryError,'NEW_MODEL_OR_PLATFORM_WAKE'):self.handoff()
        self.submit.assert_not_called()

    def test_historical_closure_does_not_get_synthetic_wait_evidence(self):
        self.finish()
        from tools.r6_job_handoff import close_terminal_job
        with self.j.writer():close_terminal_job(self.j,self.key,self.store)
        self.handoff();self.assertNotIn('local_wait_audit',self.j.read()['events'][self.key])
