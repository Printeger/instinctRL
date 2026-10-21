"""Review provenance tests; existing worker receipt validator is a declared boundary."""
import json
from unittest.mock import Mock
import pytest
from tools import r6_holdout as h
from tools.r6_holdout import verify_model_review
from tools.test_r6_holdout_journal import setup
from tools.r6_recovery import RecoveryError
from rlflow.store import sha


def reviewed(setup, monkeypatch):
    j, request, write = setup
    decision = json.loads(open(request['review']['path']).read())
    worker_request = {'fixture_worker': True}
    identity = dict(goal_id=j.read()['goal_id'], event_id='FINAL-FREEZE-01',
        job_id='FINAL-BATCH', input_sha256=sha(worker_request), role='final-freeze-review')
    with j.writer():
        key, _ = j.intent(identity, kind='model', request=worker_request,
            resources={'automatic_model_sessions': 1}, purpose='final_review', free_disk_bytes=500*1024**3)
        receipt = dict(event_key=key, request_sha256=sha(worker_request), process=None,
                       last_message=request['review'])
        pin = write('receipt.json', receipt)
        state = j.read();state['events'][key].update(status='RECEIPT_RECORDED', receipt=pin)
        j.save(state)
    validator = Mock(return_value=decision)
    monkeypatch.setattr(h, 'verified_decision', validator)
    return j, request, write, key, receipt, validator


def test_unregistered_pass_file_is_rejected(setup):
    j, request, _ = setup
    with pytest.raises(RecoveryError, match='EXACT_REGISTERED'):
        verify_model_review(j, request['review'])


def test_exact_registered_output_delegates_schema_cleanup_validation(setup, monkeypatch):
    j, request, _, key, _, validator = reviewed(setup, monkeypatch)
    assert verify_model_review(j, request['review']) == key
    validator.assert_called_once_with(j, key)


@pytest.mark.parametrize('field', ['event_key', 'request_sha256', 'process'])
def test_cross_receipt_identity_cannot_authorize_entropy(setup, monkeypatch, field):
    j, request, write, key, receipt, validator = reviewed(setup, monkeypatch)
    receipt[field] = 'other'
    with j.writer():
        s=j.read();s['events'][key]['receipt']=write('receipt2.json', receipt);j.save(s)
    with pytest.raises(RecoveryError, match='REVIEW_RECEIPT_IDENTITY'):
        verify_model_review(j, request['review'])
    validator.assert_not_called()


def test_failed_worker_validation_propagates(setup, monkeypatch):
    j, request, _, _, _, validator = reviewed(setup, monkeypatch)
    validator.side_effect = RecoveryError('MODEL_CLEANUP_NOT_VERIFIED')
    with pytest.raises(RecoveryError, match='CLEANUP'):
        verify_model_review(j, request['review'])


def test_missing_model_reservation_rejected(setup, monkeypatch):
    j, request, _, _, _, validator = reviewed(setup, monkeypatch)
    with j.writer():
        s=j.read();s['budget']['reservations']=[];j.save(s)
    with pytest.raises(RecoveryError, match='REVIEW_RESERVATION'):
        verify_model_review(j, request['review'])
    validator.assert_not_called()
