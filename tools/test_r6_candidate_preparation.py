"""Real Journal/template files; source receipt, independent reviews and frozen validator are fixtures."""
import json
from unittest.mock import Mock
import pytest
from rlflow.store import sha
from tools import r6_candidate_preparation as t
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import RecoveryError


@pytest.fixture
def ready(case,monkeypatch):
    b=case;context=json.loads(t.checked(b.context).read_text())
    identity=dict(workload_commit='a'*40,config_sha256='b'*64)
    b.snapshot=b.write('candidate-snapshot.json',dict(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',source_identity=identity))
    from training_workloads.unified_h1.robustness_evaluation_selection import FIELDS
    selected={k:b.context for k in FIELDS-{'independent_review'}}
    selected.update(role='trained_candidate',training_plan={'training_seed':11})
    design=dict(schema_version='r6-candidate-design/1',selection_template=selected,
        evaluation_template=dict(experiment_id='R6-CANDIDATE-CPU',wall_seconds=60,protocol_review=b.context,r6={}),
        origin_config=b.context,result_payload_bytes=4096)
    b.design=b.write('candidate-design.json',design)
    receipt=b.j.read()['events'][b.parent]['receipt']
    b.result=dict(candidate_design=b.design,source_commit='a'*40,worker_receipt=receipt)
    result_pin=b.write('source-result.json',b.result)
    with b.j.writer():
        request={'fixture':'source'};event=dict(goal_id=b.j.read()['goal_id'],event_id='SOURCE',job_id='SOURCE',input_sha256=sha(request),role=t.source.ROLE)
        key,_=b.j.intent(event,kind='model',request=request,resources={'automatic_model_sessions':1},purpose='development',free_disk_bytes=500*1024**3)
        state=b.j.read();state['events'][key].update(source_input={'parent_key':b.parent},source_result=result_pin,receipt=receipt)
        b.j.save(state)
    b.calls=[]
    def review(j,parent,snapshot,input_pin,store,**kwargs):
        q=json.loads(t.checked(input_pin).read_text());b.calls.append(q)
        if q['kind']=='selection':
            value=json.loads(t.checked(q['selection']).read_text())
            accepted=b.write('accepted-selection.json',dict(value,independent_review=b.context))
            return dict(status='EXACT_CANDIDATE_REVIEWS_RECORDED_NOT_AUTHORIZED',event_key='selection-fixture',
                payloads={'selection':q['selection']},reviews={'selection':b.context},accepted={'selection':accepted},dispatch_authorized=False)
        return dict(status='EXACT_CANDIDATE_REVIEWS_RECORDED_NOT_AUTHORIZED',event_key='compatibility-fixture',accepted={'compatibility':b.context},dispatch_authorized=False)
    b.review=Mock(side_effect=review);b.validate=Mock(return_value=4096)
    monkeypatch.setattr(t.review,'prepare',b.review);monkeypatch.setattr(t.review,'verified_decision',Mock(return_value={}))
    monkeypatch.setattr(t.validator,'validate',b.validate)
    return b


def run(b):
    with b.j.writer():return t.prepare(b.j,b.parent,b.snapshot,b.context,b.result,b.store,
        executable=b.context,free_disk_bytes=500*1024**3)


def test_exact_selection_review_flows_to_compatibility_and_profile(ready):
    b=ready;first=run(b);budget=b.j.read()['budget'];assert run(b)==first and b.j.read()['budget']==budget
    assert [q['kind'] for q in b.calls]==['selection','compatibility','compatibility']
    selected=json.loads(t.checked(b.calls[1]['selection']).read_text())
    assert selected['independent_review']==b.context
    context=json.loads(t.checked(first['context']).read_text());profile=json.loads(t.checked(context['baseline_profile']).read_text())
    assert profile['evaluation_selection']==selected and profile['training_seed']==11
    assert not first['dispatch_authorized']


def test_compatibility_interruption_reconnects_without_reentering_selection_review(ready):
    b=ready;original=b.review.side_effect;failed=[]
    def review(*args,**kwargs):
        q=json.loads(t.checked(args[3]).read_text())
        if q['kind']=='compatibility' and not failed:failed.append(True);raise RuntimeError('compatibility in flight')
        if failed and q['kind']=='selection':raise AssertionError('must resume compatibility directly')
        if failed and q['kind']=='compatibility':
            state=b.j.read();assert state['current_model_event']=='compatibility-fixture-running'
            state['current_model_event']=None;b.j.save(state)
        return original(*args,**kwargs)
    b.review.side_effect=review
    with pytest.raises(RuntimeError,match='compatibility in flight'):run(b)
    with b.j.writer():
        state=b.j.read();state['current_model_event']='compatibility-fixture-running';b.j.save(state)
    assert run(b)['status']=='CANDIDATE_REVIEWS_ACCEPTED_CONTEXT_READY'


def test_frozen_profile_failure_or_stop_does_not_publish_context(ready):
    b=ready;b.validate.side_effect=RecoveryError('frozen failure')
    with pytest.raises(RecoveryError,match='frozen failure'):run(b)
    assert not list((b.j.directory/'candidate-preparation').rglob('context.json'))
    def stop(*args): (b.j.directory/'STOP').touch()
    b.validate.side_effect=stop
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert not list((b.j.directory/'candidate-preparation').rglob('context.json'))


def test_cached_selection_result_drift_rejected(ready):
    b=ready;run(b)
    entry=next(e for e in b.j.read()['events'].values() if e['identity']['role']==t.ROLE)
    t.checked(entry['selection_review_result']).write_text('{}')
    with pytest.raises(RecoveryError,match='PIN_DRIFT'):run(b)


@pytest.mark.parametrize('field',['master_authority','evaluation_selection','checkpoint_compatibility'])
def test_template_cannot_override_controller_bindings(ready,field):
    b=ready;design=json.loads(t.checked(b.design).read_text());design['evaluation_template'][field]={}
    bad=b.write('bad-design.json',design)
    b.result['candidate_design']=bad
    # Rebind only the isolated source fixture so the design validator is reached.
    with b.j.writer():
        state=b.j.read();entry=next(e for e in state['events'].values() if e['identity']['role']==t.source.ROLE)
        entry['source_result']=b.write('bad-source-result.json',b.result);b.j.save(state)
    with pytest.raises(RecoveryError,match='EVALUATION_TEMPLATE'):run(b)
    b.review.assert_not_called()
