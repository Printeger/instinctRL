"""CPU fixtures only; these decisions are substitutes, never real approvals."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rlflow.store import sha
from tools import r6_legacy_model_reconciliation as module
from tools.audit_r6_legacy_model_cleanup import build, LEGACY_EVENTS
from tools.r6_job_handoff import no_model_workers
from tools.r6_recovery import Journal, RecoveryError, file_pin


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.j = Journal(Path(self.tmp.name))
        self.state = {'goal_id': 'CPU', 'events': {}, 'current_model_event': None}
        self.originals = {}
        for key in sorted(LEGACY_EVENTS):
            directory = self.j.directory / 'calls' / key
            directory.mkdir(parents=True)
            events = directory / 'events.jsonl'
            events.write_text(json.dumps({'type': 'turn.completed'}) + '\n')
            other = directory / 'other.json'
            other.write_text('{}')
            receipt = directory / 'receipt.json'
            receipt.write_text(json.dumps(dict(event_key=key, events=file_pin(events),
                stderr=file_pin(other), last_message=file_pin(other), exit_code=0,
                error=None, process=None, child_identity=None)))
            self.originals[key] = file_pin(receipt)
            self.state['events'][key] = dict(kind='model', receipt=file_pin(receipt))
        self.save()
        self.proposal = build(self.j.directory)
        proposal_path = self.j.directory / 'proposal.json'
        proposal_path.write_text(json.dumps(self.proposal))
        event = dict(goal_id='CPU', event_id='fixture', role='independent-review',
                     input_sha256=sha({}), job_id='CPU-NO-LIVE-JOB')
        self.review_key = sha(event)
        receipt_path = self.j.directory / 'calls' / self.review_key / 'receipt.json'
        receipt_path.parent.mkdir(parents=True)
        self.review_receipt = dict(event_key=self.review_key, request_sha256=sha({}),
            cleanup={'status': 'VERIFIED_NO_OWNED_DESCENDANTS', 'observed_identities': []})
        receipt_path.write_text(json.dumps(self.review_receipt))
        self.state['events'][self.review_key] = dict(kind='model', status='RECEIPT_RECORDED',
            identity=event, request={}, receipt=file_pin(receipt_path))
        self.state['legacy_model_reconciliation'] = dict(review_event_key=self.review_key,
            review_receipt=file_pin(receipt_path), proposal=file_pin(proposal_path))
        self.decision = dict(decision='PASS_LEGACY_MODEL_RECONCILIATION', must_fix=[],
            changed_files=[], next_plan=None, historical_descendant_cleanup_proven=False,
            ownership_ambiguity_remaining=False, runtime_dispatch_authorized=False,
            proposal=file_pin(proposal_path), source_pins=module.source_pins(),
            original_receipts=self.originals)
        self.save()
        self.hashes = patch.object(module, 'ORIGINAL_RECEIPTS',
                                   {k: p['sha256'] for k, p in self.originals.items()})
        self.hashes.start(); self.addCleanup(self.hashes.stop)
        # All actual acceptance tests below mock only the external model result.
        # A separate test exercises rejection by the real receipt verifier.
        self.verifier = patch('tools.r6_batch.verified_decision', return_value=self.decision)
        self.mock_verify = self.verifier.start(); self.addCleanup(self.verifier.stop)

    def save(self):
        self.j.path.write_text(json.dumps(self.state))

    def validate(self):
        return module.validate_reconciliation(self.j, sorted(LEGACY_EVENTS)[0])

    def test_exact_review_passes_without_rewriting_originals(self):
        before = {k: Path(p['path']).read_bytes() for k, p in self.originals.items()}
        result = self.validate()
        self.assertFalse(result['historical_descendant_cleanup_proven'])
        self.assertFalse(result['runtime_dispatch_authorized'])
        self.mock_verify.assert_called_once_with(self.j, self.review_key)
        self.assertEqual(before, {k: Path(p['path']).read_bytes() for k, p in self.originals.items()})

    def test_missing_approval_keeps_original_guard_closed(self):
        del self.state['legacy_model_reconciliation']; self.save()
        with self.assertRaisesRegex(RecoveryError, 'MODEL_CLEANUP_UNCONFIRMED'):
            no_model_workers(self.j)

    def test_guard_uses_exact_approved_legacy_pair_only(self):
        self.assertEqual(no_model_workers(self.j), 3)

    def test_detached_review_receipt_rejected(self):
        p = self.j.directory / 'detached.json'
        p.write_text(json.dumps(self.review_receipt))
        self.state['events'][self.review_key]['receipt'] = file_pin(p)
        self.state['legacy_model_reconciliation']['review_receipt'] = file_pin(p); self.save()
        with self.assertRaisesRegex(RecoveryError, 'REVIEW_RECEIPT_PATH'): self.validate()

    def test_missing_or_nonindependent_review_rejected(self):
        for field, value in [('status', 'INTENT'), ('kind', 'cpu'),
                             ('identity', dict(role='builder'))]:
            original = copy.deepcopy(self.state['events'][self.review_key])
            self.state['events'][self.review_key][field] = value; self.save()
            with self.assertRaisesRegex(RecoveryError, 'REVIEW_EVENT'): self.validate()
            self.state['events'][self.review_key] = original

    def test_declined_or_overbroad_decision_rejected(self):
        for field, value in [('decision', 'MUST_FIX'), ('must_fix', ['ownership']),
                ('changed_files', ['code.py']), ('next_plan', {}),
                ('historical_descendant_cleanup_proven', True),
                ('ownership_ambiguity_remaining', True), ('runtime_dispatch_authorized', True)]:
            old = self.decision[field]; self.decision[field] = value
            with self.assertRaisesRegex(RecoveryError, 'REVIEW_NOT_ACCEPTED'): self.validate()
            self.decision[field] = old

    def test_source_proposal_and_original_bindings_rejected(self):
        for field, code in [('source_pins', 'SOURCE_DRIFT'), ('proposal', 'PROPOSAL_BINDING'),
                            ('original_receipts', 'REVIEW_ORIGINALS')]:
            old = self.decision[field]; self.decision[field] = {}
            with self.assertRaisesRegex(RecoveryError, code): self.validate()
            self.decision[field] = old

    def test_real_verifier_rejects_fabricated_approval(self):
        self.verifier.stop()
        with self.assertRaisesRegex(RecoveryError, 'MODEL_FAILED_PRESERVE_RECEIPT'):
            self.validate()

    def test_unknown_missing_cleanup_receipt_cannot_use_exception(self):
        with self.assertRaisesRegex(RecoveryError, 'EVENT_NOT_LISTED'):
            module.validate_reconciliation(self.j, 'unknown')

    def test_bad_modern_cleanup_cannot_use_exception(self):
        self.review_receipt['cleanup']['status'] = 'UNKNOWN'
        p = Path(self.state['events'][self.review_key]['receipt']['path'])
        p.write_text(json.dumps(self.review_receipt))
        self.state['events'][self.review_key]['receipt'] = file_pin(p)
        self.state['legacy_model_reconciliation']['review_receipt'] = file_pin(p); self.save()
        with self.assertRaisesRegex(RecoveryError, 'MODEL_CLEANUP_UNCONFIRMED'):
            no_model_workers(self.j)

    def test_original_hash_cannot_be_refreshed_after_mutation(self):
        key = sorted(LEGACY_EVENTS)[0]
        p = Path(self.originals[key]['path']); p.write_text(p.read_text() + '\n')
        self.state['events'][key]['receipt'] = file_pin(p); self.save()
        with self.assertRaisesRegex(RecoveryError, 'ORIGINAL_RECEIPT'): self.validate()

    def test_review_liveness_rechecked(self):
        with patch.object(module, 'owns', return_value=True):
            with self.assertRaisesRegex(RecoveryError, 'REVIEW_ALIVE'): self.validate()

    def test_original_liveness_rechecked(self):
        with patch('tools.audit_r6_legacy_model_cleanup.owns', return_value=True):
            with self.assertRaisesRegex(RecoveryError, 'AUDIT_DRIFT'): self.validate()

    def test_active_review_rejected(self):
        self.state['current_model_event'] = self.review_key; self.save()
        with self.assertRaisesRegex(RecoveryError, 'REVIEW_ACTIVE'): self.validate()


if __name__ == '__main__':
    unittest.main()
