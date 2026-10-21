"""Real local fake review worker, receipts and ledger; frozen preparation is substituted."""
import json
import sys
from unittest.mock import Mock
import pytest
from tools import r6_candidate_review as r
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import file_pin,RecoveryError
from tools.r6_batch import model_handoff

SCRIPT='''import json,sys
from pathlib import Path
v=json.loads(sys.stdin.read().splitlines()[-1]);settings=json.loads(Path(__file__).with_suffix('.settings').read_text())
with Path(__file__).with_suffix('.count').open('a') as f:f.write('call\\n')
items=[dict(kind=k,decision=v['decisions'][k],reviewer='r6-independent-candidate-review',
 payload_sha256=h,summary='CPU fixture',must_fix=[]) for k,h in v['payload_hashes'].items()]
result=dict(decision='PASS_R6_CANDIDATE_REVIEW',summary='CPU fixture',changed_files=[],must_fix=[],next_plan='CPU fixture',
 evidence_pins=[*v['payload_pins'].values(),v['parent'],v['snapshot'],v['inputs']],reviews=items,
 remaining_budget={'automatic_sessions_remaining':v['caps']['automatic_model_sessions']-v['budget']['automatic_model_sessions']-1})
if settings.get('reject'):result.update(decision='MUST_FIX',must_fix=['deliberate CPU rejection'])
if settings.get('hash'):items[0]['payload_sha256']='0'*64
if settings.get('item'):items[0].update(decision='MUST_FIX',must_fix=['one component rejected'])
if settings.get('duplicate'):result['reviews']=[items[0],items[0]]
if settings.get('evidence'):result['evidence_pins']=[]
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(result)}}))
'''


@pytest.fixture(params=['selection','compatibility'])
def ready(case,monkeypatch,request):
    b=case;b.kind=request.param
    b.reviewer=b.root/'fake_candidate_review';b.reviewer.write_text('#!'+sys.executable+'\n'+SCRIPT);b.reviewer.chmod(0o700)
    b.reviewer.with_suffix('.settings').write_text('{}')
    b.snapshot=json.loads(r.checked(b.context).read_text())['snapshot_receipt']
    q=dict(kind=b.kind,selection=b.context)
    if b.kind=='compatibility':q['origin_config']=b.context
    b.input=b.write('review-input.json',q)
    b.prepared=dict(status='REVIEW_PAYLOAD_PREPARED_NOT_ACCEPTED',snapshot=b.snapshot,request=q,
        source_identity={},payload={'fixture':True},dispatch_authorized=False)
    b.prepare=Mock(return_value=b.prepared);monkeypatch.setattr(r.inputs,'prepare',b.prepare)
    monkeypatch.setattr(r,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
    return b


def run(b):
    with b.j.writer():return r.prepare(b.j,b.parent,b.snapshot,b.input,b.store,
        executable=file_pin(b.reviewer),free_disk_bytes=500*1024**3)


def test_exact_component_receipts_and_one_model_call_replay(ready):
    b=ready;before=b.store.replay();first=run(b);budget=b.j.read()['budget']
    assert run(b)==first and b.j.read()['budget']==budget and b.store.replay()==before
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'
    assert not first['dispatch_authorized']
    assert set(first['reviews'])=={b.kind}
    for kind,pin in first['reviews'].items():
        value=json.loads(r.checked(pin).read_text())
        receipt=json.loads(r.checked(value['worker_receipt']).read_text())
        original=json.loads(r.checked(value['parent_review']).read_text())
        assert receipt['last_message']==value['parent_review']
        assert {k:v for k,v in value.items() if k not in ('worker_receipt','parent_review')} in original['reviews']
        assert json.loads(r.checked(first['accepted'][kind]).read_text())['independent_review']==pin


@pytest.mark.parametrize('fault',['reject','hash','item','duplicate','evidence'])
def test_rejected_components_cannot_produce_accepted_artifacts_or_repeat_call(ready,fault):
    b=ready;b.reviewer.with_suffix('.settings').write_text(json.dumps({fault:True}))
    for _ in range(2):
        with pytest.raises(RecoveryError):run(b)
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'
    assert not list((b.j.directory/'candidate-reviews').rglob('*-review.json'))
    assert not list((b.j.directory/'candidate-reviews').rglob('*-accepted.json'))


def test_frozen_recheck_failure_preserves_review_without_acceptance(ready):
    b=ready;b.prepare.side_effect=[b.prepared,RecoveryError('frozen drift')]
    with pytest.raises(RecoveryError,match='frozen drift'):run(b)
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'
    assert not list((b.j.directory/'candidate-reviews').rglob('*-review.json'))


def test_prompt_crash_recovers_same_reservations(ready,monkeypatch):
    b=ready;original=r.persist_exact
    def crash(path,data):
        if path.name=='prompt.txt':raise RuntimeError('CPU prompt crash')
        return original(path,data)
    monkeypatch.setattr(r,'persist_exact',crash)
    with pytest.raises(RuntimeError,match='CPU prompt crash'):run(b)
    budget=b.j.read()['budget'];monkeypatch.setattr(r,'persist_exact',original)
    run(b);assert b.j.read()['budget']==budget
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'


def test_stop_before_model_dispatch_keeps_intent_without_worker(ready,monkeypatch):
    b=ready;original=r.model_handoff
    monkeypatch.setattr(r,'model_handoff',Mock(side_effect=RuntimeError('CPU before dispatch')))
    with pytest.raises(RuntimeError):run(b)
    (b.j.directory/'STOP').touch();monkeypatch.setattr(r,'model_handoff',original)
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert not b.reviewer.with_suffix('.count').exists()


def test_stop_during_frozen_recheck_prevents_acceptance(ready):
    b=ready;calls=[]
    def prepare(*args):
        calls.append(1)
        if len(calls)>1:(b.j.directory/'STOP').touch()
        return b.prepared
    b.prepare.side_effect=prepare
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert not list((b.j.directory/'candidate-reviews').rglob('*-review.json'))


@pytest.mark.parametrize('kind',['cpu','model'])
def test_missing_reservation_after_intent_prevents_worker(ready,monkeypatch,kind):
    b=ready;handoff=r.model_handoff
    monkeypatch.setattr(r,'model_handoff',Mock(side_effect=RuntimeError('before worker')))
    with pytest.raises(RuntimeError):run(b)
    role='candidate-review-input' if kind=='cpu' else r.ROLE
    with b.j.writer():
        state=b.j.read();entry=next(e for e in state['events'].values() if e['identity']['role']==role)
        state['budget']['reservations']=[x for x in state['budget']['reservations'] if x['identity']!=entry['identity']]
        b.j.save(state)
    monkeypatch.setattr(r,'model_handoff',handoff)
    with pytest.raises(RecoveryError,match='RESERVATION'):run(b)
    assert not b.reviewer.with_suffix('.count').exists()


def test_acceptance_write_crash_resumes_from_receipt_without_second_review(ready,monkeypatch):
    b=ready;persist=r.persist_exact
    def crash(path,data):
        if path.name.endswith('-review.json'):raise RuntimeError('acceptance write crash')
        return persist(path,data)
    monkeypatch.setattr(r,'persist_exact',crash)
    with pytest.raises(RuntimeError,match='acceptance write crash'):run(b)
    budget=b.j.read()['budget'];monkeypatch.setattr(r,'persist_exact',persist)
    first=run(b);assert run(b)==first and b.j.read()['budget']==budget
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'


def test_changed_store_before_worker_is_rejected(ready,monkeypatch):
    b=ready;handoff=r.model_handoff
    monkeypatch.setattr(r,'model_handoff',Mock(side_effect=RuntimeError('before worker')))
    with pytest.raises(RuntimeError):run(b)
    changed=dict(b.store.replay(),paused=True)
    monkeypatch.setattr(b.store,'replay',lambda:changed)
    monkeypatch.setattr(r,'model_handoff',handoff)
    with pytest.raises(RecoveryError):run(b)
    assert not b.reviewer.with_suffix('.count').exists()
