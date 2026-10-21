"""Real local fake CLI/Journal/receipt; parent scientific decision is a fixture."""
import json
import sys
from unittest.mock import Mock
import pytest
from tools import r6_confirmation_inputs as t
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import RecoveryError,file_pin
from tools.r6_batch import model_handoff

SCRIPT='''import json,sys
from pathlib import Path
v=json.loads(sys.stdin.read().splitlines()[-1]);s=json.loads(Path(__file__).with_suffix('.settings').read_text())
with Path(__file__).with_suffix('.count').open('a') as f:f.write('call\\n')
q=dict(snapshot=v['context'],recipe=v['context'],recipe_review=v['context'],origin_config=v['context'],
 selection_evidence={'fixture':True},experiment_ids=['R6-A','R6-B','R6-C'],wall_seconds=100,storage={'fixture':True})
if s.get('bad_fields'):q['context']=v['context']
r=dict(decision='INPUTS_READY',packet_sha256=v['packet_pin']['sha256'],summary='CPU fixture',
 evidence_pins=[v['packet_pin'],v['parent'],v['context']],changed_files=[],must_fix=[],inputs_json=json.dumps(q),
 remaining_budget={'automatic_sessions_remaining':v['caps']['automatic_model_sessions']-v['budget']['automatic_model_sessions']-1})
if s.get('reject'):r.update(decision='MUST_FIX',must_fix=['missing evidence'],inputs_json='')
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(r)}}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))
'''


@pytest.fixture
def ready(case,monkeypatch):
    b=case;b.builder.write_text('#!'+sys.executable+'\n'+SCRIPT)
    monkeypatch.setattr(t.profile,'parent_decision',Mock(return_value={'decision':{'decision':'PROPOSE_CONFIRMATION'}}))
    monkeypatch.setattr(t,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
    return b


def run(b):
    with b.j.writer():return t.prepare(b.j,b.parent,b.context,b.store,executable=file_pin(b.builder),free_disk_bytes=500*1024**3)


def test_real_receipt_replay_one_call_no_store_change(ready):
    b=ready;before=b.store.replay();result=run(b);budget=b.j.read()['budget']
    assert run(b)==result and b.j.read()['budget']==budget and b.store.replay()==before
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    q=json.loads(t.checked(result['inputs']).read_text());assert set(q)==t.preparation.FIELDS
    context=json.loads(t.checked(q['context']).read_text())
    assert context['runner']['purpose']=='confirmation_delivery'
    assert json.loads(t.checked(q['selection_evidence']).read_text())=={'fixture':True}
    assert not result['dispatch_authorized']


@pytest.mark.parametrize('fault,reason',[('bad_fields','FIELDS'),('reject','NOT_READY')])
def test_bad_model_output_never_publishes_inputs(ready,fault,reason):
    b=ready;b.builder.with_suffix('.settings').write_text(json.dumps({fault:True}))
    with pytest.raises(RecoveryError,match=reason):run(b)
    assert not list((b.j.directory/'confirmation-inputs').rglob('inputs.json'))


def test_model_intent_crash_replays_prompt_without_second_charge(ready,monkeypatch):
    b=ready;original=t.model_handoff
    monkeypatch.setattr(t,'model_handoff',Mock(side_effect=RuntimeError('before dispatch')))
    with pytest.raises(RuntimeError):run(b)
    budget=b.j.read()['budget']
    model=next(e for e in b.j.read()['events'].values() if e['identity']['role']=='confirmation-input-model')
    t.checked(model['request']['prompt']).unlink()
    monkeypatch.setattr(t,'model_handoff',original);run(b)
    assert b.j.read()['budget']==budget and b.builder.with_suffix('.count').read_text()=='call\n'


def test_stop_and_context_drift_reject_without_new_model(ready):
    b=ready;run(b);b.context=b.write('changed-context.json',{})
    with pytest.raises(RecoveryError,match='INPUT_DRIFT'):run(b)
    (b.j.directory/'STOP').touch()
    assert b.builder.with_suffix('.count').read_text()=='call\n'


def test_stop_before_intent_preserves_budget(ready):
    b=ready;before=b.j.read();(b.j.directory/'STOP').touch()
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert b.j.read()==before
