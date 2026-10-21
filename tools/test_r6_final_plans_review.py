"""Real full-batch receipts and local fake CLI; frozen spec validation is substituted."""
import sys
from unittest.mock import Mock
import pytest
from tools import r6_final_plans_review as r
from tools import r6_issue_final as issuer
from tools.test_r6_final_plans import batch, prepared, ready, case, FREE
from tools.test_r6_final_freeze_review import SCRIPT as FREEZE_SCRIPT
from tools.r6_recovery import file_pin, RecoveryError
from tools.r6_batch import model_handoff

SCRIPT=FREEZE_SCRIPT.replace('FINAL_FREEZE','FINAL_PLANS').replace('final-freeze','final-plans').replace('freeze_sha256','materializations_sha256').replace("v['freeze']","v['materializations']")

@pytest.fixture
def review_ready(batch,monkeypatch):
    b=batch
    # Separate CLI so existing compatibility receipts keep their exact executable pin.
    b.reviewer=b.builder.with_name('plans-reviewer')
    b.reviewer.write_text('#!'+sys.executable+'\n'+SCRIPT);b.reviewer.chmod(0o755)
    b.reviewer.with_suffix('.settings').write_text('{}')
    b.aggregate_pin=b.write('aggregate.json',b.aggregate)
    monkeypatch.setattr(r,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
    return b


def run(b):
    with b.j.writer():return r.prepare(b.j,b.aggregate_pin,b.store,executable=file_pin(b.reviewer),free_disk_bytes=FREE)


def test_batch_review_replays_once_and_issuer_uses_exact_index(review_ready,monkeypatch):
    b=review_ready;out=run(b);budget=b.j.read()['budget']
    assert run(b)==out and b.j.read()['budget']==budget
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'
    assert r.verify(b.j,out['event_key'],out['review'],b.aggregate_pin,b.store)['decision']=='PASS_R6_FINAL_PLANS'
    issue=Mock(return_value={'status':'fixture-issued'});monkeypatch.setattr(issuer.issuer,'_issue_reviewed',issue)
    with b.j.writer():
        assert issuer.issue(b.j,b.aggregate_pin,1,out['event_key'],out['review'],{},b.store,free_disk_bytes=FREE)=={'status':'fixture-issued'}
    assert issue.call_args.args[2]==b.aggregate['assignments'][1]['materialization']['artifacts']['plan']
    assert issue.call_args.kwargs['role']=='final-batch-authorization'


def test_rejected_review_remains_durable(review_ready):
    b=review_ready;b.reviewer.with_suffix('.settings').write_text('{"reject":true}')
    for _ in range(2):
        with pytest.raises(RecoveryError,match='REJECTED'):run(b)
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'


def test_missing_prompt_recovers_without_second_charge(review_ready,monkeypatch):
    b=review_ready;original=r.model_handoff
    monkeypatch.setattr(r,'model_handoff',Mock(side_effect=RuntimeError('interruption')))
    with pytest.raises(RuntimeError):run(b)
    event=next(e for e in b.j.read()['events'].values() if e['identity']['role']=='final-plans-review')
    budget=b.j.read()['budget'];r.checked(event['request']['prompt']).unlink()
    monkeypatch.setattr(r,'model_handoff',original);run(b)
    assert b.j.read()['budget']==budget


@pytest.mark.parametrize('fault',['index','receipt','wrong_batch','stop'])
def test_issuer_rejects_before_store_mutation(review_ready,monkeypatch,fault):
    b=review_ready;out=run(b);index=0;pin=b.aggregate_pin
    if fault=='index':index=True
    elif fault=='receipt':out['review']=b.write('fake-review.json',{})
    elif fault=='wrong_batch':pin=b.write('different-batch.json',{})
    else:(b.j.directory/'STOP').touch()
    issue=Mock();monkeypatch.setattr(issuer.issuer,'_issue_reviewed',issue)
    with b.j.writer(),pytest.raises(RecoveryError):
        issuer.issue(b.j,pin,index,out['event_key'],out['review'],{},b.store,free_disk_bytes=FREE)
    issue.assert_not_called()
