"""First-start composition with real review verifier, Store and job registration.

The model handoff and final job runner are explicit substitutes. Runtime,
delegation and kernel checks inherit the documented first-issuance fixture.
"""
from unittest.mock import Mock
import subprocess
REAL_POPEN = subprocess.Popen
import pytest
from tools.test_r6_submit_real_contract import baseline_bound as bound
from tools.test_r6_initial_baseline import ready
from tools import r6_start_baseline as m
from tools.r6_recovery import Journal, RecoveryError
from tools.r6_register_job import register


@pytest.fixture
def case(ready, monkeypatch):
    b = ready
    b.handoff = Mock()
    b.runner = Mock(return_value={'status': 'RESOURCE_OBSERVATION_REJECTED'})
    monkeypatch.setattr(m, 'model_handoff', b.handoff)
    monkeypatch.setattr(m, 'run_job', b.runner)
    return b


def run(b):
    with b.journal.writer():
        return m.start(b.journal, b.plan_pin, b.plan_pin, b.exe, b.store, free_disk_bytes=500*1024**3)


def register_job(b, claimed=False):
    with b.journal.writer():
        key = register(b.journal, b.plan['spec_pin'], b.store, free_disk_bytes=500*1024**3)
        if claimed:
            s = b.journal.read(); s['events'][key]['submission_claim'] = {'fixture': 'existing claim'}
            b.journal.save(s)
    return key


def test_repeated_admission_does_not_repeat_model_or_approval(case):
    b = case; first = run(b); store = b.store.replay(); budget = b.journal.read()['budget']
    b.journal = Journal(b.journal.directory)
    assert run(b) == first
    b.handoff.assert_called_once(); assert b.runner.call_count == 2
    assert b.store.replay() == store and b.journal.read()['budget'] == budget


def test_unclaimed_registered_job_resumes_without_first_job_rejection(case):
    b = case; run(b); register_job(b); before = b.store.replay()
    run(b); assert b.store.replay() == before
    b.handoff.assert_called_once(); assert b.runner.call_count == 2


def test_stop_blocks_unclaimed_but_claim_reaches_runner_cleanup(case):
    b = case; run(b); key = register_job(b)
    stop = b.journal.directory/'STOP'; stop.touch()
    with pytest.raises(RecoveryError, match='STOP'): run(b)
    assert b.runner.call_count == 1
    with b.journal.writer():
        s = b.journal.read(); s['events'][key]['submission_claim'] = {'fixture': 'original claim'}
        b.journal.save(s)
    run(b); assert b.runner.call_count == 2; b.handoff.assert_called_once()


def test_completion_is_cached_against_same_canonical_job(case):
    b = case; run(b); key = register_job(b, claimed=True)
    b.runner.return_value = {'status': 'DECISION_PROPOSAL_RETURNED', 'event_key': key}
    result = run(b); (b.journal.directory/'STOP').touch()
    assert run(b) == result and b.runner.call_count == 2
    b.handoff.assert_called_once()


def test_changed_plan_rejected_before_another_review(case):
    b = case; run(b); b.plan_pin = b.write('different-plan.json', dict(b.plan, sandbox='danger-full-access'))
    with pytest.raises(RecoveryError, match='ISSUANCE_DRIFT'): run(b)
    b.handoff.assert_called_once(); b.runner.assert_called_once()


def test_changed_review_evidence_blocks_unclaimed_job(case):
    b = case; run(b)
    from pathlib import Path
    Path(b.review_pin['path']).write_text('{}')
    with pytest.raises(RecoveryError, match='PIN_DRIFT'): run(b)
    b.runner.assert_called_once()


def test_failed_handoff_does_not_issue_or_run(case):
    b = case; b.handoff.side_effect = RecoveryError('failed model')
    before = b.store.replay()
    with pytest.raises(RecoveryError, match='failed model'): run(b)
    assert b.store.replay() == before; b.runner.assert_not_called()


def test_crash_after_issuance_before_runner_resumes_without_model(case):
    b = case; b.runner.side_effect = RuntimeError('before run')
    with pytest.raises(RuntimeError, match='before run'): run(b)
    before = b.store.replay(); budget = b.journal.read()['budget']
    b.journal = Journal(b.journal.directory); b.runner.side_effect = None
    run(b)
    b.handoff.assert_called_once()
    assert b.store.replay() == before and b.journal.read()['budget'] == budget


@pytest.mark.parametrize('decision', ['PASS_R6_INITIAL_BASELINE', 'MUST_FIX'])
@pytest.mark.parametrize('resume', [False, True, 'missing_prompt'])
def test_real_local_worker_first_baseline_chain(ready, monkeypatch, tmp_path, decision, resume):
    import functools
    import json
    import sys
    from pathlib import Path
    from tools.r6_recovery import file_pin
    from tools.r6_batch import model_handoff
    from tools.r6_job_handoff import no_model_workers
    b = ready
    budget = json.loads((Path(__file__).parent/'fixtures/r6_budget.json').read_text())
    budget['capacity_pass'] = True
    b.journal = Journal(tmp_path/'real-worker-batch')
    with b.journal.writer(): b.journal.initialize(budget)
    def worker_only(argv, *args, **kw):
        assert argv[1:4] == ['-B', '-m', 'tools.r6_model_worker']
        return REAL_POPEN(argv, *args, **kw)
    monkeypatch.setattr(subprocess, 'Popen', worker_only)
    result = dict(b.result, decision=decision,
                  must_fix=[] if decision == 'PASS_R6_INITIAL_BASELINE' else ['deliberate CPU rejection'])
    launch = tmp_path/'fake-launches.txt'; exe = tmp_path/'actual-fake-cli'
    event = {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': json.dumps(result)}}
    exe.write_text('#!'+sys.executable+'\nimport sys,time\nfrom pathlib import Path\n'
        +'sys.stdin.read()\nwith Path('+repr(str(launch))+').open("a") as f:f.write("launch\\n")\n'
        +'time.sleep(.2)\nprint('+repr(json.dumps(event))+')\n')
    exe.chmod(0o700); b.exe = file_pin(exe)
    monkeypatch.setattr(m, 'model_handoff', functools.partial(model_handoff, poll_seconds=.05))
    runner = Mock(return_value={'status': 'RESOURCE_OBSERVATION_REJECTED'})
    monkeypatch.setattr(m, 'run_job', runner)
    before = len(b.store.replay()['approvals'])
    if resume:
        with b.journal.writer():
            key = m.initial.register_review(b.journal, b.plan_pin, b.store,
                executable=b.exe, free_disk_bytes=500*1024**3)
            if resume == 'missing_prompt':
                Path(b.journal.read()['events'][key]['request']['prompt']['path']).unlink()
            else:
                model_handoff(b.journal, key, poll_seconds=.05)
        b.journal = Journal(b.journal.directory)
    for iteration in range(2):
        if decision == 'MUST_FIX':
            with pytest.raises(RecoveryError, match='REVIEW_BINDING'): run(b)
        else:
            assert run(b)['status'] == 'RESOURCE_OBSERVATION_REJECTED'
        if iteration == 0: charged = b.journal.read()['budget']
        else: assert b.journal.read()['budget'] == charged
    assert launch.read_text() == 'launch\n'
    assert len(b.store.replay()['approvals']) == before+(decision == 'PASS_R6_INITIAL_BASELINE')
    assert runner.call_count == (2 if decision == 'PASS_R6_INITIAL_BASELINE' else 0)
    assert not b.store.replay()['jobs']; b.popen.assert_not_called()
    with b.journal.writer(): no_model_workers(b.journal)
    assert b.journal.read().get('current_model_event') is None
