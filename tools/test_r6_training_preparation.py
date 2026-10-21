"""Real Journal/template files; source receipt, independent reviews and frozen validator are fixtures."""
import json
from unittest.mock import Mock
import pytest
from rlflow.store import sha
from tools import r6_training_preparation as t
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import RecoveryError


@pytest.fixture
def ready(case,monkeypatch):
    b=case;context=json.loads(t.checked(b.context).read_text())
    identity=dict(workload_commit='a'*40,config_sha256='b'*64)
    b.snapshot=b.write('training-snapshot.json',dict(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',source_identity=identity))
    design=dict(schema_version='r6-training-design/1',recipe_template={'hypothesis':'fixture'},
        plan_template=dict(experiment_id='R6-TRAINING-CPU',phase='development',training_seed=11),origin_config=b.context)
    b.design=b.write('training-design.json',design)
    receipt=b.j.read()['events'][b.parent]['receipt']
    b.result=dict(training_design=b.design,source_commit='a'*40,worker_receipt=receipt)
    result_pin=b.write('source-result.json',b.result)
    with b.j.writer():
        request={'fixture':'source'};event=dict(goal_id=b.j.read()['goal_id'],event_id='SOURCE',job_id='SOURCE',input_sha256=sha(request),role=t.source.ROLE)
        key,_=b.j.intent(event,kind='model',request=request,resources={'automatic_model_sessions':1},purpose='development',free_disk_bytes=500*1024**3)
        state=b.j.read();state['events'][key].update(source_input={'parent_key':b.parent},source_result=result_pin,receipt=receipt)
        b.j.save(state)
    b.calls=[]
    def review(j,parent,snapshot,input_pin,store,**kwargs):
        q=json.loads(t.checked(input_pin).read_text());b.calls.append(q)
        if q['kind']=='recipe':return dict(status='EXACT_TRAINING_REVIEWS_RECORDED_NOT_AUTHORIZED',event_key='recipe-fixture',payloads={'recipe':q['recipe']},reviews={'recipe':b.context},dispatch_authorized=False)
        return dict(status='EXACT_TRAINING_REVIEWS_RECORDED_NOT_AUTHORIZED',event_key='stage-fixture',accepted={'capacity':b.context,'compatibility':b.context},dispatch_authorized=False)
    b.review=Mock(side_effect=review);b.validate=Mock(return_value=4096)
    monkeypatch.setattr(t.review,'prepare',b.review);monkeypatch.setattr(t.review,'verified_decision',Mock(return_value={}))
    monkeypatch.setattr(t.validator,'validate',b.validate)
    return b


def run(b):
    with b.j.writer():return t.prepare(b.j,b.parent,b.snapshot,b.context,b.result,b.store,
        executable=b.context,free_disk_bytes=500*1024**3)


def test_exact_frozen_identity_and_recipe_receipt_flow_to_stage_and_profile(ready):
    b=ready;first=run(b);budget=b.j.read()['budget'];assert run(b)==first and b.j.read()['budget']==budget
    assert [q['kind'] for q in b.calls]==['recipe','stage','stage']
    recipe=json.loads(t.checked(b.calls[0]['recipe']).read_text())
    assert recipe['source_identity']==json.loads(t.checked(b.snapshot).read_text())['source_identity']
    plan=json.loads(t.checked(b.calls[1]['plan']).read_text());assert plan['recipe_review']==b.context
    context=json.loads(t.checked(first['context']).read_text());profile=json.loads(t.checked(context['baseline_profile']).read_text())
    assert profile['r6_training_plan']==plan and profile['training_seed']==11
    assert not first['dispatch_authorized']


def test_stage_interruption_reconnects_without_reentering_recipe_review(ready):
    b=ready;original=b.review.side_effect;failed=[]
    def review(*args,**kwargs):
        q=json.loads(t.checked(args[3]).read_text())
        if q['kind']=='stage' and not failed:failed.append(True);raise RuntimeError('stage in flight')
        if failed and q['kind']=='recipe':raise AssertionError('must resume stage directly')
        if failed and q['kind']=='stage':
            state=b.j.read();assert state['current_model_event']=='stage-fixture-running'
            state['current_model_event']=None;b.j.save(state)
        return original(*args,**kwargs)
    b.review.side_effect=review
    with pytest.raises(RuntimeError,match='stage in flight'):run(b)
    with b.j.writer():
        state=b.j.read();state['current_model_event']='stage-fixture-running';b.j.save(state)
    assert run(b)['status']=='TRAINING_REVIEWS_ACCEPTED_CONTEXT_READY'


def test_frozen_profile_failure_or_stop_does_not_publish_context(ready):
    b=ready;b.validate.side_effect=RecoveryError('frozen failure')
    with pytest.raises(RecoveryError,match='frozen failure'):run(b)
    assert not list((b.j.directory/'training-preparation').rglob('context.json'))
    def stop(*args): (b.j.directory/'STOP').touch()
    b.validate.side_effect=stop
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert not list((b.j.directory/'training-preparation').rglob('context.json'))


def test_cached_recipe_result_drift_rejected(ready):
    b=ready;run(b)
    entry=next(e for e in b.j.read()['events'].values() if e['identity']['role']==t.ROLE)
    t.checked(entry['recipe_review_result']).write_text('{}')
    with pytest.raises(RecoveryError,match='PIN_DRIFT'):run(b)
