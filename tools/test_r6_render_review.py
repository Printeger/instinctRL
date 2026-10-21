"""Real local fake CLI/receipts; no independent model, mount or live job."""
import json
import sys
from unittest.mock import Mock
import pytest
from tools import r6_render_review as r, r6_render_materialization as m
from tools.test_r6_render_preparation import ready, case, run as prepare, FREE
from tools.test_r6_final_freeze_review import SCRIPT as BASE_SCRIPT
from tools.r6_recovery import file_pin, RecoveryError
from tools.r6_batch import model_handoff

SCRIPT=BASE_SCRIPT.replace('FINAL_FREEZE','RENDER_PLAN').replace('final-freeze','render-plan').replace('freeze_sha256','preparation_sha256').replace("v['freeze']","v['preparation']")
SCRIPT=SCRIPT.replace("if settings.get('reject'):","import hashlib\np=json.loads(Path(json.loads(Path(v['preparation']['path']).read_text())['parameters']['path']).read_text())\nd['payload_sha256']=hashlib.sha256(json.dumps({k:x for k,x in p.items() if k!='independent_review'},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()\nif settings.get('reject'):")

@pytest.fixture
def review_ready(ready,monkeypatch):
    b=ready;out=prepare(b);b.preparation=b.j.read()['events'][out['event_key']]['preparation']
    b.reviewer=b.root/'render-reviewer';b.reviewer.write_text('#!'+sys.executable+'\n'+SCRIPT);b.reviewer.chmod(0o700)
    b.reviewer.with_suffix('.settings').write_text('{}')
    monkeypatch.setattr(r,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
    return b


def run(b):
    with b.j.writer():return r.prepare(b.j,b.preparation,b.store,executable=file_pin(b.reviewer),free_disk_bytes=FREE)


def test_review_receipt_replays_without_new_call(review_ready):
    b=review_ready;before=b.store.replay();out=run(b);budget=b.j.read()['budget']
    assert run(b)==out and b.j.read()['budget']==budget and b.store.replay()==before
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'
    assert r.verify(b.j,out['event_key'],out['review'],b.preparation,b.store)['decision']=='PASS_R6_RENDER_PLAN'


def test_rejected_review_is_durable(review_ready):
    b=review_ready;b.reviewer.with_suffix('.settings').write_text('{"reject":true}')
    for _ in range(2):
        with pytest.raises(RecoveryError,match='REJECTED'):run(b)
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'


def test_missing_prompt_recovery_uses_original_reservation(review_ready,monkeypatch):
    b=review_ready;original=r.model_handoff
    monkeypatch.setattr(r,'model_handoff',Mock(side_effect=RuntimeError('fixture interruption')))
    with pytest.raises(RuntimeError):run(b)
    event=next(e for e in b.j.read()['events'].values() if e['identity']['role']=='render-plan-review')
    budget=b.j.read()['budget'];r.checked(event['request']['prompt']).unlink()
    monkeypatch.setattr(r,'model_handoff',original);run(b)
    assert b.j.read()['budget']==budget


def test_materialization_replays_and_issuer_gets_exact_plan(review_ready,monkeypatch):
    b=review_ready;out=run(b);before=b.store.replay()
    monkeypatch.setattr(m.builder,'build',lambda config,auth,attempt,stage,conditions,root:dict(id='CPU-RENDER',pins=dict(authorization=auth)))
    monkeypatch.setattr(m.storage,'for_spec',Mock())
    def materialize():
        with b.j.writer():return m.materialize(b.j,b.preparation,out['event_key'],out['review'],b.store,free_disk_bytes=FREE)
    result=materialize();budget=b.j.read()['budget']
    assert materialize()==result and b.j.read()['budget']==budget and b.store.replay()==before
    auth=m.read(result['authorization'])
    assert auth['parameters']['independent_review']==out['review'] and auth['mode']==m.contract.MODE
    issue=Mock(return_value={'status':'CPU-ISSUER-FIXTURE'});monkeypatch.setattr(m.issuer,'_issue_reviewed',issue)
    pin=b.j.read()['events'][result['event_key']]['materialization']
    with b.j.writer():assert m.issue(b.j,pin,b.store,free_disk_bytes=FREE)['status']=='CPU-ISSUER-FIXTURE'
    assert issue.call_args.args[2]==result['plan'] and issue.call_args.kwargs['role']=='render-authorization'
    (b.j.directory/'STOP').touch();issue.reset_mock()
    with b.j.writer(),pytest.raises(RecoveryError):m.issue(b.j,pin,b.store,free_disk_bytes=FREE)
    issue.assert_not_called()


def test_replaced_receipt_rejected_before_materialization(review_ready,monkeypatch):
    b=review_ready;out=run(b);builder=Mock();monkeypatch.setattr(m.builder,'build',builder)
    with b.j.writer(),pytest.raises(RecoveryError,match='REVIEW_PIN'):
        m.materialize(b.j,b.preparation,out['event_key'],b.write('fake.json',{}),b.store,free_disk_bytes=FREE)
    builder.assert_not_called()
