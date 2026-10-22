"""Narrow stale resource-review recovery; science/reviewer are explicit fixtures."""
import copy
from unittest.mock import Mock
import pytest
from tools.test_r6_baseline_failure import case,enter,FREE
from tools.test_r6_initial_review_repair import ready,bound
from tools import r6_baseline_input_repair as m
from tools.r6_recovery import RecoveryError

@pytest.fixture
def repair_case(case,monkeypatch):
    b=case
    oldcontext={'runner':{k:b.plan[k] for k in b.plan if k!='spec_pin'},'scientific_input':'unchanged'}
    with b.journal.writer():
        s=b.journal.read();owner=s['events'][b.controller]
        owner['cursor']['context']=b.write('old-context.json',oldcontext)
        owner['controller_process']=None
        b.journal.save(s)
    enter(b)
    b.newplan=copy.deepcopy(b.plan)
    b.newplan['queue_review']=b.write('new-queue.json',{'decision':'fixture'})
    b.newpin=b.write('new-plan.json',b.newplan)
    b.newargs=dict(b.args,plan_pin=b.newpin)
    b.newrequest=b.write('new-request.json',b.newargs)
    context=copy.deepcopy(oldcontext);context['runner']['queue_review']=b.newplan['queue_review']
    b.context=b.write('new-context.json',context)
    b.verify=Mock();monkeypatch.setattr(m.initial,'verify_review',b.verify)
    return b

def adopt(b):
    with b.journal.writer():
        return m.adopt_reviewed_inputs(b.journal,b.controller,b.newrequest,b.context,b.review,b.store)

def test_adopts_only_new_resource_pins_preserves_originals(repair_case):
    b=repair_case;before=b.journal.read();store=b.store.replay()
    assert adopt(b)==b.newargs
    after=b.journal.read();old=before['events'][b.controller];new=after['events'][b.controller]
    assert new['request']==old['request'] and new['identity']==old['identity']
    assert after['events'][b.oldkey]==before['events'][b.oldkey]
    assert after['budget']==before['budget'] and b.store.replay()==store
    assert new['cursor']['phase']=='FIRST_JOB' and new['cursor']['context']==b.context
    assert m.effective_initial(b.journal,b.controller,b.args)==b.newargs

@pytest.mark.parametrize('damage',['master','executable','spec','purpose','context','review','issued','live','stop'])
def test_rejects_unsafe_rebinding_without_state_mutation(repair_case,damage):
    b=repair_case
    if damage in ('master','executable'):
        field='master_pin' if damage=='master' else 'review_executable'
        b.newargs[field]=b.newpin;b.newrequest=b.write('bad-request.json',b.newargs)
    if damage in ('spec','purpose'):
        b.newplan['spec_pin' if damage=='spec' else 'purpose']=b.newpin if damage=='spec' else 'repair'
        # spec damage must remain valid JSON/spec to test scope, not parser failure.
        if damage=='spec':b.newplan['spec_pin']=b.write('other-spec.json',__import__('json').loads(__import__('pathlib').Path(b.plan['spec_pin']['path']).read_text()))
        b.newargs['plan_pin']=b.write('bad-plan.json',b.newplan);b.newrequest=b.write('bad-request.json',b.newargs)
    if damage=='context':b.context=b.write('bad-context.json',{})
    if damage=='review':b.verify.side_effect=RecoveryError('REJECTED_REVIEW')
    if damage=='stop':(b.journal.directory/'STOP').write_text('user stop')
    if damage in ('issued','live'):
        with b.journal.writer():
            s=b.journal.read()
            if damage=='issued':s['events']['issued']={'kind':'cpu','identity':{'role':'initial-baseline-authorization'}}
            else:s['events'][b.controller]['controller_process']=__import__('rlflow.jobs',fromlist=['identity']).identity(__import__('os').getpid())
            b.journal.save(s)
    before=b.journal.read()
    with pytest.raises((RecoveryError,KeyError,ValueError)):adopt(b)
    assert b.journal.read()==before

def test_actual_preserved_failure_packet_still_reads_original_implementation():
    # Production read-only evidence: the real failed cursor must not PIN_DRIFT.
    import json
    from pathlib import Path
    from tools.r6_recovery import checked
    root=Path(__file__).resolve().parents[1]
    packet=root/'docs/experiments/r6-robustness-auto/batch/baseline-failure/b57920e81ad596938c1e556b955330fba9af61591ea1a91feee2afccb2e5e1a2/packet.json'
    value=json.loads(packet.read_text())
    assert checked(value['implementation'])==root/'tools/r6_baseline_failure.py'
