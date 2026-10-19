"""Real temporary Store/approval/submit integration; no process is launched.

Resource observations and clock are synthetic CPU fixtures. The only submit
boundary replaced is release creation/Popen, where an injected failure proves
recovery from the actual append-only Store reservation.
"""
import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rlflow import experiments, policy
from rlflow.store import Store, Conflict, sha
from tools import r6_resource_queue as q
from tools.r6_recovery import Journal, file_pin
from tools.r6_submit import submit_once
from training_workloads.scale_003.resources import GPU_IDENTITY


@pytest.fixture
def bound(tmp_path, monkeypatch):
    store = Store(tmp_path / 'store')
    work = store.root / 'workspaces' / 'cpu'
    work.mkdir(parents=True)
    script = work / 'inert.py'
    script.write_text('raise AssertionError("fixture must never execute")\n')
    pin = file_pin(script)
    executable = file_pin(sys.executable)
    spec = dict(id='CPU-STORE-SUBMIT', stage_id='CPU', backend='runner_cpu',
        argv=[executable['path'], str(script)], cwd=str(work),
        pins=dict(dispatcher=file_pin(experiments.DISPATCHER), executable=executable,
                  product=pin, workload=pin, runtime=pin, authorization=pin),
        method='cpu-only', evaluation='no-execution', runtime='cpu-only', seed=1,
        resources=dict(starts=1, wall_seconds=60, gpu_seconds=0, transitions=0),
        conditions=[dict(evidence_id='cpu-condition', sha256='1'*64)],
        observations=['exit_code'])
    store.transact('FIXTURE_CONDITION', 'condition', lambda s:s['evidence'].update({
        'cpu-condition':dict(id='cpu-condition', sha256='1'*64, accepted=True)}))
    proposal = experiments.propose(store, spec)
    shown = policy.show_proposal(store, proposal)
    policy.decide(store, proposal, shown['sha256'], 'APPROVED', 'temporary CPU fixture only')
    approval = 'A-' + shown['sha256'][:16]
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    journal = Journal(tmp_path / 'batch')
    from pathlib import Path
    budget = json.loads((Path(__file__).parent / 'fixtures/r6_budget.json').read_text())
    budget['capacity_pass'] = True
    clock = dict(boot_id='CPU', monotonic_ns=10**12)
    monkeypatch.setattr(q, 'now', lambda:dict(clock))
    with journal.writer():
        journal.initialize(budget)
        queue_key = q.begin(journal, file_pin(path), ordinal=0, free_disk_bytes=500*1024**3)
        for i in range(13):
            if i:clock['monotonic_ns'] += 5*10**9
            q.append_sample(journal, queue_key, i, dict(monotonic_ns=clock['monotonic_ns'],
                gpu=dict(GPU_IDENTITY, memory_used_mib=560,
                    free_memory_mib=GPU_IDENTITY['memory_total_mib']-560, utilization_percent=3),
                compute_apps=[]))
        q.finish(journal, queue_key)
        request = dict(spec=file_pin(path), wait_deadline_unix=time.time()+60)
        jid = 'J-' + sha(dict(id=spec['id'], spec=spec))[:16]
        event = dict(goal_id=budget['goal_id'], event_id=spec['id'], job_id=jid,
                     input_sha256=sha(request), role='job')
        key, _ = journal.intent(event, kind='job', request=request,
            resources={'live_starts':1}, purpose='development', free_disk_bytes=500*1024**3)
    popen = Mock(side_effect=OSError('injected before process launch'))
    monkeypatch.setattr(experiments.subprocess, 'Popen', popen)
    return SimpleNamespace(store=store, journal=journal, spec=spec, key=key, jid=jid,
        queue_key=queue_key, proposal=proposal, approval=approval, script=script, popen=popen)


def invoke(b):
    with b.journal.writer():
        return submit_once(b.journal, b.key, b.store, queue_key=b.queue_key)


@pytest.mark.parametrize('change,reason', [
    (lambda b,s:s['approvals'][b.approval].update(revoked=True), 'no unique'),
    (lambda b,s:s['proposals'][b.proposal].update(revoked=True), 'no unique'),
    (lambda b,s:s['approvals'][b.approval].update(proposal_version=2), 'no unique'),
    (lambda b,s:s['proposals'][b.proposal]['content']['spec'].update(seed=2), 'no unique'),
    (lambda b,s:s['evidence']['cpu-condition'].update(sha256='2'*64), 'conditional'),
    (lambda b,s:s['evidence']['cpu-condition'].update(accepted=False), 'conditional'),
    (lambda b,s:s.update(paused=True), 'paused or STOP'),
])
def test_real_authority_rejection_before_claim(bound, change, reason):
    b = bound
    b.store.transact('FIXTURE_DRIFT', 'drift', lambda s:change(b,s))
    before = b.journal.read()['budget']
    with pytest.raises(Conflict, match=reason):invoke(b)
    assert 'submission_claim' not in b.journal.read()['events'][b.key]
    assert b.journal.read()['budget'] == before
    assert not b.store.replay()['jobs']
    b.popen.assert_not_called()


def test_real_pin_drift_blocks_before_claim(bound):
    bound.script.write_text('changed fixture input\n')
    with pytest.raises(Conflict, match='input drift'):invoke(bound)
    assert 'submission_claim' not in bound.journal.read()['events'][bound.key]
    bound.popen.assert_not_called()


def test_store_stop_blocks_before_claim(bound):
    (bound.store.root / 'STOP').touch()
    with pytest.raises(Conflict, match='paused or STOP'):invoke(bound)
    assert not bound.store.replay()['jobs']
    bound.popen.assert_not_called()


def test_actual_store_reservation_recovered_without_second_submit(bound, monkeypatch):
    b = bound
    # No release snapshot or child is created. Everything before Popen, including
    # validate, authorized, Store transaction and request persistence, is real.
    monkeypatch.setattr('rlflow.releases.freeze', lambda root:b.store.root)
    budget = b.journal.read()['budget']
    with pytest.raises(OSError, match='before process launch'):invoke(b)
    job = b.store.replay()['jobs'][b.jid]
    assert job['status'] == 'START_INTENT' and job['identity'] is None
    assert job['approval_id'] == b.approval and job['spec'] == b.spec
    assert b.journal.read()['events'][b.key]['submission_claim']['approval_id'] == b.approval
    # Reconstruct Store to demonstrate replay from disk, not a cached object.
    b.store = Store(b.store.root)
    first = invoke(b)
    assert invoke(b) == first
    assert b.popen.call_count == 1
    assert b.store.replay()['jobs'][b.jid] == job
    assert b.journal.read()['budget'] == budget
    assert first['job_observed_in_store'] and not first['scientific_acceptance']
