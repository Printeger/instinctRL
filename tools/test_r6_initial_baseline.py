"""Real review schema/receipt and Store transactions; no worker or simulator.

Runtime inspection, master delegation and kernel volume checks are explicit
fixtures. Test receipts represent completed workers, not real model reviews.
"""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from rlflow import policy
from rlflow.store import sha, Conflict
from tools.test_r6_submit_real_contract import baseline_bound as bound
from tools.r6_recovery import Journal, file_pin, RecoveryError
from tools import r6_initial_baseline as m, r6_issue_next


@pytest.fixture
def ready(bound, tmp_path, monkeypatch):
    b = bound
    def write(name, value):
        p = tmp_path/name; p.write_text(json.dumps(value)); return file_pin(p)
    b.write = write
    budget = json.loads((Path(__file__).parent/'fixtures/r6_budget.json').read_text())
    budget['capacity_pass'] = True
    b.journal = Journal(tmp_path/'first-batch')
    with b.journal.writer(): b.journal.initialize(budget)
    b.newspec = dict(b.spec, id=m.EXPERIMENT)
    sp = write('first-spec.json', b.newspec)
    exe = tmp_path/'fake-reviewer'; exe.write_text('#!/bin/sh\nexit 1\n'); exe.chmod(0o700)
    b.exe = file_pin(exe)
    b.plan = dict(spec_pin=sp, query_executable=b.exe, decision_executable=b.exe,
                  queue_review=None, sampler_review=None, purpose='development', sandbox='read-only')
    b.plan_pin = write('first-plan.json', b.plan)
    monkeypatch.setattr(m, 'delegation', Mock())
    monkeypatch.setattr(r6_issue_next, 'delegation', Mock())
    monkeypatch.setattr('tools.r6_storage_allocation.for_spec', lambda *a, **kw: None)
    with b.journal.writer():
        b.review = m.register_review(b.journal, b.plan_pin, b.store,
            executable=b.exe, free_disk_bytes=500*1024**3)
    b.result = dict(decision='PASS_R6_INITIAL_BASELINE', plan_sha256=b.plan_pin['sha256'],
        summary='CPU fixture', evidence_pins=[b.plan_pin, sp], changed_files=[], must_fix=[])
    set_receipt(b)
    return b


def set_receipt(b):
    b.review_pin = b.write('first-review.json', b.result)
    with b.journal.writer():
        s = b.journal.read(); e = s['events'][b.review]
        b.receipt = dict(event_key=b.review, request_sha256=sha(e['request']), process=None,
            last_message=b.review_pin, exit_code=0, error=None, schema_valid=True,
            events=b.write('review-events.json', {}), stderr=b.write('review-stderr.json', {}),
            cleanup=dict(status='VERIFIED_NO_OWNED_DESCENDANTS', stable_empty_scans=2,
                         signalled_process_count=0, observed_identities=[]))
        e.update(receipt=b.write('first-receipt.json', b.receipt), status='RECEIPT_RECORDED')
        b.journal.save(s)


def issue(b):
    with b.journal.writer():
        return m.issue(b.journal, b.plan_pin, b.review, b.review_pin, b.plan_pin, b.store,
                       free_disk_bytes=500*1024**3)


def test_first_review_issues_once_without_parent_or_submission(ready):
    b = ready; result = issue(b); before = b.store.replay(); budget = b.journal.read()['budget']
    assert issue(b) == result and b.store.replay() == before
    assert b.journal.read()['budget'] == budget
    assert result['status'] == 'EXACT_INITIAL_BASELINE_AUTHORIZED_NOT_SUBMITTED'
    assert result['job_id'] not in before['jobs']; b.popen.assert_not_called()
    assert b.journal.read()['events'][result['event_key']]['identity']['role'] == 'initial-baseline-authorization'


def test_crash_after_store_approval_recovers_without_reapproval(ready, monkeypatch):
    b = ready; original = policy.decide
    def crash(*args, **kw):
        original(*args, **kw); raise RuntimeError('after approval')
    monkeypatch.setattr(policy, 'decide', crash)
    with pytest.raises(RuntimeError, match='after approval'): issue(b)
    before = b.store.replay(); budget = b.journal.read()['budget']
    monkeypatch.setattr(policy, 'decide', Mock(side_effect=AssertionError('duplicate approval')))
    issue(b)
    assert b.store.replay() == before and b.journal.read()['budget'] == budget


@pytest.mark.parametrize('damage', ['mustfix', 'plan', 'changed_files', 'cleanup', 'receipt', 'reservation'])
def test_invalid_review_cannot_issue(ready, damage):
    b = ready; before = b.store.replay()
    if damage == 'mustfix': b.result.update(decision='MUST_FIX', must_fix=['missing evidence'])
    if damage == 'plan': b.result['plan_sha256'] = '0'*64
    if damage == 'changed_files': b.result['changed_files'] = ['unexpected.py']
    set_receipt(b)
    with b.journal.writer():
        s = b.journal.read(); e = s['events'][b.review]
        if damage == 'cleanup': b.receipt['cleanup']['stable_empty_scans'] = 0
        if damage == 'receipt': b.receipt['request_sha256'] = '0'*64
        if damage in {'cleanup', 'receipt'}: e['receipt'] = b.write('bad-receipt.json', b.receipt)
        if damage == 'reservation':
            next(r for r in s['budget']['reservations'] if r['identity'] == e['identity'])['purpose'] = 'repair'
        b.journal.save(s)
    import jsonschema
    with pytest.raises((RecoveryError, jsonschema.ValidationError)): issue(b)
    assert b.store.replay() == before


@pytest.mark.parametrize('guard', ['stop', 'capacity', 'prior_job', 'store_job'])
def test_first_job_guards_prevent_issuance(ready, guard):
    b = ready
    with b.journal.writer():
        s = b.journal.read()
        if guard == 'stop': (b.journal.directory/'STOP').touch()
        if guard == 'capacity': s['budget']['capacity_pass'] = False
        if guard == 'prior_job': s['events']['fixture-job'] = dict(kind='job', identity={'role':'job'})
        b.journal.save(s)
    if guard == 'store_job':
        b.store.transact('FIXTURE_JOB', 'existing-r6', lambda s:s['jobs'].update(
            existing={'spec': {'id': 'R6-ALREADY-RAN'}}))
    before = b.store.replay()
    with pytest.raises(RecoveryError): issue(b)
    assert b.store.replay() == before


def test_revoked_first_approval_stays_revoked(ready):
    b = ready; r = issue(b)
    b.store.transact('FIXTURE_REVOKE', 'first', lambda s:s['approvals'][r['approval_id']].update(revoked=True))
    before = b.store.replay()
    with pytest.raises(Conflict): issue(b)
    assert b.store.replay() == before


def test_prompt_recovery_before_dispatch_reuses_exact_event(ready):
    b = ready
    # Explicitly reconstruct the durable intent-before-prompt crash window.
    with b.journal.writer():
        s = b.journal.read(); e = s['events'][b.review]; e.pop('receipt'); e['status'] = 'INTENT'
        b.journal.save(s)
    prompt = Path(e['request']['prompt']['path']); content = prompt.read_bytes(); prompt.unlink()
    budget = b.journal.read()['budget']
    with b.journal.writer():
        assert m.register_review(b.journal, b.plan_pin, b.store,
            executable=b.exe, free_disk_bytes=500*1024**3) == b.review
    assert prompt.read_bytes() == content and b.journal.read()['budget'] == budget
