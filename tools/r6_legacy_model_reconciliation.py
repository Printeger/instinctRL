"""Narrow, independently reviewed reconciliation of two immutable old receipts.

This is not retrospective descendant-cleanup proof. A terminal transcript and
absent original processes alone never authorize the exception. The independent
decision must explicitly assess the complete tool transcript for ownership
ambiguity; its own modern receipt must prove cleanup. No state is written here.
"""
import json
from pathlib import Path

from rlflow.jobs import owns
from rlflow.store import sha
from tools.audit_r6_legacy_model_cleanup import audit_event, LEGACY_EVENTS
from tools.r6_recovery import RecoveryError, checked, file_pin

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RECEIPTS = {
    '655b94d7271bec2c5e1b6132bebcff0bbb620170f49122caf77595acfd35fe01':
        '60104087056120fbfc6f049d54fd508cb87859b7b2744f3dfe5ed99da22a0f0c',
    'badf0aab50e96ec4ec4a8c2aedc325d7f8e6dc8f09bacefcf79566456584ec75':
        '82cdb6ba2ac8cb559d3055c7def8bc74537f477ab4c16b784b4251d347ae554e',
}
SOURCE_FILES = (
    'tools/r6_legacy_model_reconciliation.py',
    'tools/audit_r6_legacy_model_cleanup.py',
    'tools/r6_job_handoff.py', 'tools/r6_batch.py',
    'tools/r6_model_worker.py', 'tools/r6_recovery.py',
)


def source_pins():
    return {name: file_pin(ROOT / name) for name in SOURCE_FILES}


def read_pin(pin):
    return json.loads(checked(pin).read_text())


def require(condition, code):
    if not condition:
        raise RecoveryError('LEGACY_RECONCILIATION_' + code)


def validate_reconciliation(journal, key):
    """Verify an explicit approval tied to a completed independent model event.

    State may contain ``legacy_model_reconciliation`` with review_event_key,
    review_receipt and proposal pins. Merely creating this field cannot pass:
    the model's pinned, schema-validated output must approve these exact inputs.
    Any drift requires a new review, never an automatic refreshed approval.
    """
    require(key in LEGACY_EVENTS, 'EVENT_NOT_LISTED')
    state = journal.read()
    approval = state.get('legacy_model_reconciliation')
    require(isinstance(approval, dict) and set(approval) == {
        'review_event_key', 'review_receipt', 'proposal'}, 'APPROVAL_REQUIRED')
    review_key = approval['review_event_key']
    review = state['events'].get(review_key, {})
    require(review_key not in LEGACY_EVENTS and review.get('kind') == 'model'
            and review.get('status') == 'RECEIPT_RECORDED'
            and review.get('identity', {}).get('role') == 'independent-review'
            and review.get('identity', {}).get('goal_id') == state['goal_id']
            and sha(review.get('identity')) == review_key, 'REVIEW_EVENT')
    require(review.get('receipt') == approval['review_receipt'], 'REVIEW_RECEIPT')
    require(approval['review_receipt']['path'] == str(
        (journal.directory / 'calls' / review_key / 'receipt.json').resolve()), 'REVIEW_RECEIPT_PATH')
    receipt = read_pin(approval['review_receipt'])
    require(receipt.get('event_key') == review_key
            and receipt.get('request_sha256') == sha(review['request'])
            and review['identity']['input_sha256'] == sha(review['request']), 'REVIEW_BINDING')
    require(state.get('current_model_event') is None, 'REVIEW_ACTIVE')
    identities = [review.get('process'), receipt.get('process'), receipt.get('child_identity'),
                  *receipt.get('cleanup', {}).get('observed_identities', [])]
    for filename in ('worker.json', 'child.json'):
        path = journal.directory / 'calls' / review_key / filename
        if path.exists():
            require(not path.is_symlink(), 'REVIEW_IDENTITY_SYMLINK')
            identities.append(json.loads(path.read_text()).get('identity'))
    require(not any(owns(i) for i in identities), 'REVIEW_ALIVE')
    # Lazy import avoids the batch/handoff dependency cycle. This verifies the
    # real request/schema and output pins, successful exit and modern cleanup.
    from tools.r6_batch import verified_decision
    decision = verified_decision(journal, review_key)
    require(decision.get('decision') == 'PASS_LEGACY_MODEL_RECONCILIATION'
            and decision.get('must_fix') == [] and decision.get('changed_files') == []
            and decision.get('next_plan') is None
            and decision.get('historical_descendant_cleanup_proven') is False
            and decision.get('ownership_ambiguity_remaining') is False
            and decision.get('runtime_dispatch_authorized') is False,
            'REVIEW_NOT_ACCEPTED')
    require(decision.get('proposal') == approval['proposal'], 'PROPOSAL_BINDING')
    require(decision.get('source_pins') == source_pins(), 'SOURCE_DRIFT')
    proposal = read_pin(approval['proposal'])
    require(proposal.get('schema_version') == 'r6-legacy-model-reconciliation-proposal/1'
            and proposal.get('modern_cleanup_schema_waived') is False
            and proposal.get('runtime_dispatch_authorized') is False, 'PROPOSAL_SCOPE')
    require(proposal.get('auditor') == file_pin(ROOT / SOURCE_FILES[1]), 'AUDITOR_DRIFT')
    rows = proposal.get('events', [])
    require(len(rows) == 2 and {r['event_key'] for r in rows} == LEGACY_EVENTS, 'EXACT_TWO_EVENTS')
    expected = {}
    for row in rows:
        event_key = row['event_key']
        entry = state['events'][event_key]
        original = entry['receipt']
        require(original['sha256'] == ORIGINAL_RECEIPTS[event_key]
                and original['path'] == str((journal.directory / 'calls' / event_key / 'receipt.json').resolve()),
                'ORIGINAL_RECEIPT')
        current = audit_event(event_key, entry)
        require(current == row, 'AUDIT_DRIFT')
        require(current['all_reported_tool_items_terminal'] is True
                and current['worker_currently_alive'] is False
                and current['cli_currently_alive'] is False
                and current['historical_descendant_cleanup_proven'] is False, 'AUDIT_INCOMPLETE')
        expected[event_key] = original
    require(decision.get('original_receipts') == expected, 'REVIEW_ORIGINALS')
    # Do not allow the review or original evidence to change during validation.
    checked(approval['review_receipt'])
    checked(approval['proposal'])
    for pin in expected.values():
        checked(pin)
    return {'review_event_key': review_key, 'original_receipts': expected,
            'historical_descendant_cleanup_proven': False,
            'runtime_dispatch_authorized': False}
