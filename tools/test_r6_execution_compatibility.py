"""Real fake-review process; snapshot/payload/frozen-validator boundaries explicit."""
import json
import sys
from unittest.mock import Mock
import pytest
from tools import r6_execution_compatibility as c
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import file_pin,RecoveryError
from tools.r6_batch import model_handoff

SCRIPT='''import json,sys
from pathlib import Path
v=json.loads(sys.stdin.read().splitlines()[-1])
with Path(__file__).with_suffix('.count').open('a') as f:f.write('call\\n')
settings=json.loads(Path(__file__).with_suffix('.settings').read_text())
r=dict(decision='PASS_R6_EXECUTION_COMPATIBILITY',summary='CPU fixture',
 evidence_pins=[v['payload_pin'],v['parent'],v['context'],v['snapshot']],changed_files=[],
 next_plan='CPU fixture',remaining_budget={'automatic_sessions_remaining':v['model_cap']-v['budget']['automatic_model_sessions']-1},
 must_fix=[],reviewer='r6-independent-source-compatibility',payload_sha256=v['payload_sha256'])
if settings.get('reject'):r.update(decision='MUST_FIX',must_fix=['CPU deliberate rejection'])
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(r)}}))
'''


@pytest.fixture
def ready(case,monkeypatch):
    b=case;b.reviewer=b.root/'fake_compat';b.reviewer.write_text('#!'+sys.executable+'\n'+SCRIPT);b.reviewer.chmod(0o700)
    b.reviewer.with_suffix('.settings').write_text('{}')
    b.snapshot=json.loads(c.checked(b.context).read_text())['snapshot_receipt']
    b.payload={'schema_version':'r6-baseline-compatibility/1','origin_source_identity':{},'execution_source_identity':{},
        'origin_config':b.context,'execution_config':b.context,'changed_product_sources':{}}
    monkeypatch.setattr(c,'payload',Mock(return_value=b.payload))
    b.frozen=Mock();monkeypatch.setattr(c,'frozen_validate',b.frozen)
    monkeypatch.setattr(c,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
    return b


def run(b):
    with b.j.writer():return c.prepare(b.j,b.parent,b.snapshot,b.context,b.store,
        executable=file_pin(b.reviewer),free_disk_bytes=500*1024**3)


def test_review_context_acceptance_replays_one_call_without_store_mutation(ready):
    b=ready;before=b.store.replay();result=run(b);budget=b.j.read()['budget']
    assert result['status']=='EXACT_COMPATIBILITY_ACCEPTED_CONTEXT_READY'
    assert not result['dispatch_authorized']
    context=json.loads(c.checked(result['context']).read_text())
    reference=json.loads(c.checked(context['baseline_profile']).read_text())
    assert reference['checkpoint_compatibility']==result['accepted']
    assert run(b)==result and b.j.read()['budget']==budget and b.store.replay()==before
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'
    assert b.frozen.call_count==2


def test_rejection_preserves_receipt_no_context_or_second_worker(ready):
    b=ready;b.reviewer.with_suffix('.settings').write_text('{"reject":true}')
    for _ in range(2):
        with pytest.raises(RecoveryError,match='REVIEW_REJECTED'):run(b)
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'
    assert not list((b.j.directory/'source-compatibility').rglob('context.json'))
    b.frozen.assert_not_called()


def test_frozen_validator_rejects_nominal_review_pass(ready):
    b=ready;b.frozen.side_effect=RecoveryError('exact frozen validator failure')
    with pytest.raises(RecoveryError,match='frozen validator failure'):run(b)
    assert not list((b.j.directory/'source-compatibility').rglob('context.json'))
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'


def test_crash_before_prompt_write_recovers_original_intents(ready,monkeypatch):
    b=ready;original=c.persist_exact
    def crash(path,data):
        if path.name=='prompt.txt':raise RuntimeError('CPU prompt crash')
        return original(path,data)
    monkeypatch.setattr(c,'persist_exact',crash)
    with pytest.raises(RuntimeError):run(b)
    budget=b.j.read()['budget'];monkeypatch.setattr(c,'persist_exact',original)
    run(b);assert b.j.read()['budget']==budget
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'


def test_stop_after_intent_prevents_new_worker(ready,monkeypatch):
    b=ready;original=c.model_handoff
    monkeypatch.setattr(c,'model_handoff',Mock(side_effect=RuntimeError('CPU before worker')))
    with pytest.raises(RuntimeError):run(b)
    (b.j.directory/'STOP').touch();monkeypatch.setattr(c,'model_handoff',original)
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert not b.reviewer.with_suffix('.count').exists()


def test_payload_drift_after_review_is_rejected(ready):
    b=ready;run(b);b.payload['changed_product_sources']['unexpected']={}
    with pytest.raises(RecoveryError,match='PAYLOAD_DRIFT'):run(b)
    assert b.reviewer.with_suffix('.count').read_text()=='call\n'


@pytest.mark.parametrize('change',['none','actor','method'])
def test_payload_builder_enforces_config_invariants(tmp_path,monkeypatch,change):
    import copy
    def pin(name,v):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v));return file_pin(path)
    metric='training_workloads/unified_h1/evaluation.py'
    metricpin=pin(metric,{})
    config=dict(versions={},actor={},training={},execution={},method=dict(version='v1',proposal_sha256='a',hashes={}),
        product=dict(path=str(tmp_path),source_sha256={}),workload=dict(path=str(tmp_path),source_sha256={metric:metricpin['sha256']}))
    old=pin('origin.json',config);new=copy.deepcopy(config)
    if change=='actor':new['actor']={'privileged':True}
    if change=='method':new['method']['version']='v2'
    newpin=pin('training_workloads/unified_h1/candidate_config.json',new)
    snapshot=pin('snapshot.json',dict(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',config=newpin,
        snapshot=str(tmp_path),source_identity={'config_sha256':newpin['sha256']}))
    previous=pin('previous.json',dict(origin_config=old,origin_source_identity={'config_sha256':old['sha256']}))
    reference=pin('reference.json',dict(checkpoint_compatibility=previous))
    monkeypatch.setattr(c,'verify_frozen',Mock())
    if change!='none':
        with pytest.raises(RecoveryError,match='PHASE_CONTRACT_REQUIRED'):c.payload(snapshot,reference)
    else:
        value=c.payload(snapshot,reference)
        assert value['changed_product_sources']=={} and value['execution_config']==newpin
