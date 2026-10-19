"""CPU composition tests: actual Journal/Store/close; synthetic terminal receipts.

No job or GPU process is created. Existing handoff tests separately cover live
CPU handle waiting, cancellation and cleanup; this file covers phase composition.
"""
import json
from unittest.mock import Mock

import pytest
from rlflow import experiments
from rlflow.store import atomic, canonical, sha
from tools.r6_recovery import Journal, RecoveryError
from tools.r6_submit import submit_and_close, main
from tools.test_r6_submit_store import bound, cli_args


@pytest.fixture
def cycle(bound, monkeypatch):
    b=bound
    with b.journal.writer():
        state=b.journal.read();event=state['events'][b.key]
        resources=dict(live_starts=1,live_wall_seconds=240,allocated_gpu_seconds=240)
        event['resources']=resources
        next(r for r in state['budget']['reservations'] if r['identity']==event['identity'])['resources']=resources
        b.journal.save(state)
    def finish(store,spec):
        with pytest.raises(BlockingIOError):
            with Journal(b.journal.directory).writer():pass
        assert b.journal.read()['events'][b.key].get('submission_claim')
        owner=dict(pid=99999999,pgid=99999999,starttime='0',boot_id='CPU-FIXTURE-NOT-LIVE')
        receipt=dict(job_id=b.jid,spec_sha256=sha(spec),status='FAILED',exit_code=-9,
                     elapsed_seconds=2,allocated_seconds=2)
        job=dict(id=b.jid,spec=spec,approval_id=b.approval,status='FAILED',identity=owner,receipt=receipt)
        store.transact('FIXTURE_TERMINAL','terminal',lambda s:s['jobs'].update({b.jid:job}))
        clean=dict(quiescent=True,timed_out=False,survivors=[],stable_scans=2,required_stable_scans=2,discovered=[])
        records={'receipt.json':receipt,'descendant-cleanup.json':clean,
            'lifeguard-ack.json':dict(owner=owner,no_survivors=True,descendant_cleanup=clean),
            'lifeguard-ready.json':dict(identity=owner),'process-identity.json':owner,
            'descendant-ledger.json':dict(owner=owner,descendants=[])}
        for name,value in records.items():atomic(store.root/'jobs'/b.jid/name,canonical(value))
        return job
    b.submit=Mock(side_effect=finish)
    monkeypatch.setattr(experiments,'submit',b.submit)
    return b


def invoke(b):
    with b.journal.writer():return submit_and_close(b.journal,b.key,b.store,queue_key=b.queue_key)


def test_cli_combines_submit_and_close_with_native_failure_preserved(cycle,capsys):
    b=cycle
    main([*cli_args(b),'--wait-close'])
    result=json.loads(capsys.readouterr().out)
    assert result['closure']['native_status']=='FAILED'
    assert result['closure']['native_exit_code']==-9
    assert not result['closure']['scientific_acceptance']
    assert b.journal.read()['current_job'] is None
    assert b.journal.read()['status']=='CPU_ANALYZE'
    assert b.submit.call_count==1
    b.popen.assert_not_called()


def test_closed_replay_does_not_overwrite_downstream_state(cycle):
    b=cycle;first=invoke(b)
    with b.journal.writer():
        state=b.journal.read();state['status']='MODEL_DECISION_RECORDED';b.journal.save(state)
    before=b.journal.read()
    assert invoke(b)==first
    assert b.journal.read()==before
    assert b.submit.call_count==1


def test_closed_replay_revalidates_raw_evidence(cycle):
    b=cycle;invoke(b)
    path=b.store.root/'jobs'/b.jid/'descendant-cleanup.json'
    value=json.loads(path.read_text());value['quiescent']=False
    atomic(path,canonical(value))
    with pytest.raises(RecoveryError,match='NORMAL_ACK_REQUIRED'):invoke(b)
    assert b.submit.call_count==1


def test_cleanup_reservation_checked_before_any_submit(bound):
    with pytest.raises(RecoveryError,match='CLOSE_RESERVATION_REQUIRED'):invoke(bound)
    assert not bound.store.replay()['jobs']
    assert 'submission_claim' not in bound.journal.read()['events'][bound.key]
    bound.popen.assert_not_called()


def test_crash_after_submit_before_close_reconnects_without_submit(cycle,monkeypatch):
    from tools import r6_job_handoff
    b=cycle;original=r6_job_handoff.job_handoff
    monkeypatch.setattr(r6_job_handoff,'job_handoff',Mock(side_effect=RuntimeError('CPU handoff crash')))
    with pytest.raises(RuntimeError,match='handoff crash'):invoke(b)
    assert b.journal.read()['events'][b.key].get('submission')
    monkeypatch.setattr(r6_job_handoff,'job_handoff',original)
    result=invoke(b)
    assert result['closure']['native_exit_code']==-9 and b.submit.call_count==1


def test_stop_allows_existing_closed_evidence_recovery(cycle):
    first=invoke(cycle)
    (cycle.journal.directory/'STOP').touch()
    before=cycle.journal.read()
    assert invoke(cycle)==first
    assert cycle.journal.read()==before and cycle.submit.call_count==1
