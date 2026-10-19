from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

from rlflow.jobs import identity
from rlflow.store import sha
from tools.r6_recovery import Journal, RecoveryError

ROOT = Path(__file__).resolve().parents[1]


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.j = Journal(self.directory)
        budget = json.loads((ROOT/'tools/fixtures/r6_budget.json').read_text())
        self.request = {'prompt': 'cpu fixture', 'schema': 'fixture', 'argv': ['CPU_ONLY']}
        self.event = dict(goal_id=budget['goal_id'], event_id='decision-01', job_id='CPU-01',
                          input_sha256=sha(self.request), role='review')
        with self.j.writer(): self.j.initialize(budget)

    def tearDown(self): self.tmp.cleanup()

    def intent(self):
        return self.j.intent(self.event, kind='model', request=self.request,
                             resources={'automatic_model_sessions': 1}, purpose='development',
                             free_disk_bytes=500*1024**3)

    def receipt(self, key, raw=None):
        p = self.directory/'receipt.json'
        p.write_text(raw if raw is not None else json.dumps(dict(
            event_key=key, request_sha256=sha(self.request), exit_code=0, process=None)))
        return p

    def test_restarts_do_not_repeat_model_intent(self):
        with self.j.writer():
            key, fresh = self.intent()
            self.assertTrue(fresh)
        self.j = Journal(self.directory)
        with self.j.writer():
            self.assertEqual(self.intent(), (key, False))
        self.assertEqual(len(self.j.read()['budget']['reservations']), 1)
        self.assertEqual(self.j.observe(key)['action'], 'RECONCILE')

    def test_partial_receipt_preserved_without_retry(self):
        with self.j.writer(): key, _ = self.intent()
        p = self.receipt(key, '{"event_key":')
        self.assertEqual(self.j.observe(key, receipt_path=p)['action'], 'RECONCILE')
        self.assertEqual(p.read_text(), '{"event_key":')

    def test_receipt_consumption_survives_restart(self):
        with self.j.writer():
            key, _ = self.intent()
            p = self.receipt(key)
            self.assertTrue(self.j.consume(key, p))
        self.j = Journal(self.directory)
        with self.j.writer(): self.assertFalse(self.j.consume(key, p))
        self.assertEqual(self.j.observe(key)['action'], 'ALREADY_CONSUMED')

    def test_receipt_drift_rejected_after_consumption(self):
        with self.j.writer():
            key, _ = self.intent(); p = self.receipt(key); self.j.consume(key, p)
        p.write_text('{}')
        with self.assertRaises(RecoveryError): self.j.observe(key)

    def test_stop_blocks_new_intent_but_allows_receipt(self):
        with self.j.writer():
            key, _ = self.intent()
            (self.directory/'STOP').touch()
            with self.assertRaises(RecoveryError): self.intent()
            self.assertTrue(self.j.consume(key, self.receipt(key)))
        self.assertTrue((self.directory/'STOP').exists())

    def test_live_worker_then_dead_worker_not_restarted(self):
        proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'],
                                start_new_session=True)
        try:
            with self.j.writer():
                key, _ = self.intent(); self.j.bind_process(key, identity(proc.pid))
            self.assertEqual(self.j.observe(key)['action'], 'WAIT_OWNED_WORKER')
            proc.terminate(); proc.wait(timeout=5)
            self.assertEqual(self.j.observe(key)['action'], 'RECONCILE')
            with self.j.writer(): self.assertFalse(self.intent()[1])
        finally:
            if proc.poll() is None: proc.kill(); proc.wait()

    def test_exact_job_recovery_validates_identity(self):
        with self.j.writer(): key, _ = self.intent()
        job = dict(job_id='CPU-01', input_sha256=sha(self.request), terminal=True)
        self.assertEqual(self.j.observe(key, exact_job=job)['action'], 'CLOSE_AND_SETTLE')
        job['job_id'] = 'ANOTHER_JOB'
        with self.assertRaises(RecoveryError): self.j.observe(key, exact_job=job)

    def test_duplicate_controller_refused_by_os_lock(self):
        code = '''import fcntl,sys
with open(sys.argv[1], 'a') as f:
 try: fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: sys.exit(23)
sys.exit(0)
'''
        with self.j.writer():
            result = subprocess.run([sys.executable, '-c', code, str(self.directory/'writer.lock')])
            self.assertEqual(result.returncode, 23)
        result = subprocess.run([sys.executable, '-c', code, str(self.directory/'writer.lock')])
        self.assertEqual(result.returncode, 0)

    def test_mutations_require_writer(self):
        with self.assertRaises(RecoveryError): self.intent()
        with self.assertRaises(RecoveryError): self.j.save({})

    def test_wrong_receipt_cannot_advance(self):
        with self.j.writer(): key, _ = self.intent()
        p = self.receipt('wrong-key')
        with self.j.writer():
            with self.assertRaises(RecoveryError): self.j.consume(key, p)


if __name__ == '__main__': unittest.main()
