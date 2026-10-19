"""Exercise complete local receipts through the real decision verifier."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.r6_batch import verified_decision, ROOT
from tools.r6_job_handoff import no_model_workers
from tools.r6_recovery import Journal, RecoveryError, file_pin
from tools.r6_worker_processes import cleanup_verified


class CleanupValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.j = Journal(self.root)
        self.good = dict(status='VERIFIED_NO_OWNED_DESCENDANTS', observed_identities=[],
                         stable_empty_scans=2, signalled_process_count=0)
        self.prompt = self.pin('prompt.txt', 'CPU test fixture')
        self.schema = self.pin('schema.json', json.dumps({'type': 'object', 'properties': {
            'decision': {'const': 'CPU_FIXTURE'}}, 'required': ['decision'], 'additionalProperties': False}))
        self.result = self.pin('last-message.json', '{"decision":"CPU_FIXTURE"}')
        self.events = self.pin('events.jsonl', '')
        self.stderr = self.pin('stderr.log', '')
        self.request = dict(prompt=self.prompt, schema=self.schema, executable=file_pin(sys.executable),
            cwd=str(ROOT), sandbox='read-only', timeout_seconds=60, output_cap_bytes=65536, stop_paths=[])

    def pin(self, name, text):
        p = self.root / name; p.write_text(text); return file_pin(p)

    def record(self, cleanup):
        receipt = dict(exit_code=0, error=None, schema_valid=True, cleanup=cleanup,
                       events=self.events, stderr=self.stderr, last_message=self.result)
        entry = dict(kind='model', process=None, request=self.request,
                     receipt=self.pin('receipt.json', json.dumps(receipt)))
        self.j.path.write_text(json.dumps({'events': {'modern': entry}, 'current_model_event': None}))

    def test_complete_receipt_passes_real_verifier_and_handoff_guard(self):
        self.record(self.good)
        self.assertEqual(verified_decision(self.j, 'modern'), {'decision': 'CPU_FIXTURE'})
        self.assertEqual(no_model_workers(self.j), 1)

    def test_status_only_and_missing_fields_rejected_by_real_paths(self):
        invalid = [{'status': self.good['status']}, None]
        invalid += [{k: v for k, v in self.good.items() if k != missing} for missing in self.good]
        for value in invalid:
            with self.subTest(value=value):
                self.record(value)
                with self.assertRaisesRegex(RecoveryError, 'MODEL_CLEANUP_NOT_VERIFIED'):
                    verified_decision(self.j, 'modern')
                with self.assertRaisesRegex(RecoveryError, 'MODEL_CLEANUP_UNCONFIRMED'):
                    no_model_workers(self.j)

    def test_counts_and_types_rejected(self):
        for field, values in [('stable_empty_scans', [True, 1, 2.0, '2']),
                              ('signalled_process_count', [False, -1, 1, '0']),
                              ('observed_identities', [None, {}, [None], [{}]])]:
            for value in values:
                bad = copy.deepcopy(self.good); bad[field] = value; self.record(bad)
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(RecoveryError, 'MODEL_CLEANUP_NOT_VERIFIED'):
                        verified_decision(self.j, 'modern')

    def test_owned_identity_rechecked_through_real_verifier(self):
        record = {**self.good, 'observed_identities': [dict(pid=123, pgid=123,
                    boot_id='fixture', starttime='456')], 'signalled_process_count': 1}
        self.record(record)
        with patch('tools.r6_worker_processes.owns', return_value=True):
            with self.assertRaisesRegex(RecoveryError, 'MODEL_CLEANUP_NOT_VERIFIED'):
                verified_decision(self.j, 'modern')
        with patch('tools.r6_worker_processes.owns', return_value=False):
            self.assertTrue(cleanup_verified(record))


if __name__ == '__main__':
    unittest.main()
