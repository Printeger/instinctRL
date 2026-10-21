"""Local fake CLI through real review receipts; final provenance fixture is explicit."""
import json
import sys
from unittest.mock import Mock
import pytest
from tools import r6_final_freeze_review as r
from tools import r6_final_models as f
from tools.r6_holdout import verify_model_review
from tools.test_r6_final_models import ready
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import file_pin,RecoveryError
from tools.r6_batch import model_handoff

SCRIPT='''import json,sys
from pathlib import Path
v=json.loads(sys.stdin.read().splitlines()[-1]);settings=json.loads(Path(__file__).with_suffix('.settings').read_text())
with Path(__file__).with_suffix('.count').open('a') as s:s.write('call\\n')
d=dict(decision='PASS_R6_FINAL_FREEZE',freeze_sha256=v['freeze']['sha256'],reviewer='r6-independent-final-freeze',
 summary='CPU fixture',must_fix=[],changed_files=[],evidence_pins=[v['freeze']],
 remaining_budget={'automatic_sessions_remaining':v['caps']['automatic_model_sessions']-v['budget']['automatic_model_sessions']-1})
if settings.get('reject'):d.update(decision='MUST_FIX',must_fix=['fixture missing semantics'])
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(d)}}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))
'''


@pytest.fixture
def review_ready(ready,monkeypatch):
    b=ready
    with b.j.writer():b.freeze=f.prepare(b.j,b.controller,b.store,free_disk_bytes=500*1024**3)['freeze']
    b.builder.write_text('#!'+sys.executable+'\n'+SCRIPT)
    monkeypatch.setattr(r,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
    return b


def run(b):
    with b.j.writer():return r.prepare(b.j,b.freeze,b.store,executable=file_pin(b.builder),free_disk_bytes=500*1024**3)


def test_consumed_review_receipt_is_accepted_by_holdout_verifier_once(review_ready):
    b=review_ready;before=b.store.replay();out=run(b);budget=b.j.read()['budget']
    assert run(b)==out and b.j.read()['budget']==budget and b.store.replay()==before
    assert verify_model_review(b.j,out['review'])==out['event_key']
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    assert b.j.read()['events'][out['event_key']]['purpose']=='final_review'
    assert not out['holdout_released'] and not out['scientific_acceptance']


def test_must_fix_is_durable_and_never_reinvokes_or_releases(review_ready):
    b=review_ready;b.builder.with_suffix('.settings').write_text('{"reject":true}')
    for _ in range(2):
        with pytest.raises(RecoveryError,match='REJECTED'):run(b)
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    assert not list((b.j.directory/'final-freeze-review').rglob('result.json'))


def test_missing_prompt_after_intent_replays_same_review(review_ready,monkeypatch):
    b=review_ready;original=r.model_handoff
    monkeypatch.setattr(r,'model_handoff',Mock(side_effect=RuntimeError('in flight')))
    with pytest.raises(RuntimeError):run(b)
    budget=b.j.read()['budget'];event=next(e for e in b.j.read()['events'].values() if e['identity']['role']=='final-freeze-review')
    r.checked(event['request']['prompt']).unlink();monkeypatch.setattr(r,'model_handoff',original)
    run(b);assert b.j.read()['budget']==budget and b.builder.with_suffix('.count').read_text()=='call\n'


def test_changed_model_provenance_after_review_is_rejected(review_ready):
    b=review_ready;run(b)
    with b.j.writer():
        state=b.j.read();state['events'][b.controller]['cursor']['confirmation']['results'].pop();b.j.save(state)
    with pytest.raises(RecoveryError,match='THREE_COMPLETED_STREAMS'):run(b)
    assert b.builder.with_suffix('.count').read_text()=='call\n'
