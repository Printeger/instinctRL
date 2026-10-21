"""Replacement reviews use immutable predecessor evidence and real Store issuance.

Review receipts, runtime and kernel checks are CPU fixtures, not scientific proof.
"""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from tools.test_r6_submit_real_contract import baseline_bound as bound
from tools.test_r6_initial_baseline import ready, set_receipt, issue
from tools import r6_initial_baseline as m, r6_start_baseline as start
from tools.r6_recovery import RecoveryError


def failed(b):
    b.result.update(decision='MUST_FIX', must_fix=['missing evidence'])
    set_receipt(b)


def repair(b, name='repair.json'):
    old=b.journal.read()['events'][b.review]
    return b.write(name, dict(plan=b.plan_pin, predecessor=b.review,
        predecessor_receipt=old['receipt'], reason='Provide missing exact input evidence',
        evidence_pins=[b.plan['spec_pin']]))


def register(b, pin):
    with b.journal.writer():
        return m.register_review(b.journal,b.plan_pin,b.store,executable=b.exe,
            free_disk_bytes=500*1024**3,repair_pin=pin)


def accept(b, key):
    write=b.write
    b.write=lambda name,value:write('replacement-'+name,value)
    b.review=key
    b.result.update(decision='PASS_R6_INITIAL_BASELINE',must_fix=[])
    set_receipt(b)
    b.write=write


def test_replacement_issues_using_new_review_once(ready,monkeypatch):
    b=ready; failed(b); oldkey=b.review
    old=copy.deepcopy(b.journal.read()['events'][oldkey])
    original=Path(old['receipt']['path']).read_bytes()
    pin=repair(b); key=register(b,pin); budget=b.journal.read()['budget']
    assert key!=oldkey and register(b,pin)==key
    assert b.journal.read()['budget']==budget
    accept(b,key)
    handoff=Mock(); runner=Mock(return_value={'status':'RESOURCE_OBSERVATION_REJECTED'})
    monkeypatch.setattr(start,'model_handoff',handoff);monkeypatch.setattr(start,'run_job',runner)
    for _ in range(2):
        with b.journal.writer():
            start.start(b.journal,b.plan_pin,b.plan_pin,b.exe,b.store,free_disk_bytes=500*1024**3)
    assert handoff.call_count==1 and handoff.call_args.args[1]==key
    assert runner.call_count==2
    assert b.journal.read()['events'][oldkey]==old
    assert Path(old['receipt']['path']).read_bytes()==original
    assert m.active_review(b.journal,b.plan_pin)==key
    with pytest.raises(RecoveryError,match='REPAIR_AFTER_ISSUANCE'):register(b,pin)


def test_new_attempt_cannot_fork_or_issue_old_review(ready):
    b=ready; failed(b); pin=repair(b); register(b,pin)
    with pytest.raises(RecoveryError,match='REVIEW_NOT_ACTIVE'):issue(b)
    doc=json.loads(Path(pin['path']).read_text());doc['reason']='Another repair'
    with pytest.raises(RecoveryError,match='REPAIR_NOT_ACTIVE_TIP'):
        register(b,b.write('fork.json',doc))


def test_prompt_crash_recovery_preserves_reservation(ready,monkeypatch):
    from tools import r6_decision_packet as packet
    b=ready;failed(b);pin=repair(b);persist=packet.persist_exact
    monkeypatch.setattr(packet,'persist_exact',Mock(side_effect=RuntimeError('write crash')))
    with pytest.raises(RuntimeError,match='write crash'):register(b,pin)
    key=m.active_review(b.journal,b.plan_pin);budget=b.journal.read()['budget']
    monkeypatch.setattr(packet,'persist_exact',persist)
    entry=b.journal.read()['events'][key]
    assert register(b,m.repair_for_entry(b.journal,entry,b.plan_pin))==key
    assert b.journal.read()['budget']==budget
    assert Path(entry['request']['prompt']['path']).exists()


@pytest.mark.parametrize('damage',['accepted','cleanup','missing_receipt','missing_exit','live','identity'])
def test_unproven_or_accepted_predecessor_cannot_be_replaced(ready,damage):
    from rlflow.jobs import identity
    import os
    b=ready
    if damage!='accepted':failed(b)
    with b.journal.writer():
        state=b.journal.read();entry=state['events'][b.review]
        if damage=='cleanup':b.receipt['cleanup']['stable_empty_scans']=0
        if damage=='missing_exit':b.receipt.pop('exit_code')
        if damage=='live':b.receipt['child_identity']=identity(os.getpid())
        if damage=='identity':b.receipt['child_identity']={'pid':999999999}
        entry['receipt']=b.write('damaged-receipt.json',b.receipt)
        if damage=='missing_receipt':entry['receipt']=None
        b.journal.save(state)
    pin=repair(b);before=b.journal.read();store=b.store.replay()
    with pytest.raises(RecoveryError):register(b,pin)
    assert b.journal.read()==before and b.store.replay()==store


def test_failed_worker_with_verified_cleanup_can_be_replaced(ready):
    b=ready
    with b.journal.writer():
        state=b.journal.read();entry=state['events'][b.review]
        b.receipt.update(exit_code=1,error='sandbox failure',schema_valid=False,last_message=None)
        entry['receipt']=b.write('failed-worker.json',b.receipt);b.journal.save(state)
    key=register(b,repair(b))
    assert b.journal.read()['events'][key]['purpose']=='repair'
