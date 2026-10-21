"""Real Journal/files; independent reviewer and frozen-source validators are fixtures."""
import json
from unittest.mock import Mock
import pytest
from tools import r6_confirmation_preparation as t
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import RecoveryError


@pytest.fixture
def ready(case,monkeypatch):
    b=case;context=json.loads(t.checked(b.context).read_text())
    context['runner']['purpose']='confirmation_delivery'
    context_pin=b.write('confirmation-context.json',context)
    identity={'workload_commit':'a'*40}
    snapshot=b.write('confirmation-snapshot.json',dict(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',source_identity=identity))
    b.q=dict(snapshot=snapshot,context=context_pin,recipe=b.write('recipe.json',{'source_identity':identity}),
        recipe_review=b.context,selection_evidence=b.context,origin_config=b.context,
        experiment_ids=['R6-C1','R6-C2','R6-C3'],wall_seconds=100,storage={'fixture':True})
    b.input=b.write('confirmation-input.json',b.q)
    monkeypatch.setattr(t.profile,'parent_decision',Mock(return_value={'decision':{'decision':'PROPOSE_CONFIRMATION'}}))
    b.calls=[]
    def reviewed(j,parent,snapshot,pin,store,**kwargs):
        q=json.loads(t.checked(pin).read_text());b.calls.append(q)
        assert kwargs['purpose']=='confirmation_delivery' and snapshot==b.q['snapshot']
        keys=['freeze'] if q['kind']=='freeze' else ['capacity','compatibility']
        return dict(status='EXACT_TRAINING_REVIEWS_RECORDED_NOT_AUTHORIZED',event_key='review-fixture',
            payloads={k:b.context for k in keys},reviews={k:b.context for k in keys},
            accepted={k:b.context for k in keys},dispatch_authorized=False)
    b.review=Mock(side_effect=reviewed)
    monkeypatch.setattr(t.review,'prepare',b.review)
    monkeypatch.setattr(t.review,'verified_decision',Mock(return_value={}))
    b.built=dict(status='THREE_CONFIRMATION_PLANS_PREPARED_NOT_RESERVED',dispatch_authorized=False,
        bundle={'plans':[dict(experiment_id=e,training_seed=i,phase='confirmation') for i,e in enumerate(b.q['experiment_ids'],21)]})
    monkeypatch.setattr(t.plans,'prepare',Mock(return_value=b.built))
    b.validate=Mock(return_value=4096);monkeypatch.setattr(t.validator,'validate',b.validate)
    return b


def run(b):
    with b.j.writer():return t.prepare(b.j,b.parent,b.input,b.store,executable=b.context,free_disk_bytes=500*1024**3)


def test_three_contexts_retain_frozen_source_and_order_without_job_reservations(ready):
    b=ready;before=b.store.replay();result=run(b);budget=b.j.read()['budget']
    assert run(b)==result and b.j.read()['budget']==budget and b.store.replay()==before
    assert [q['kind'] for q in b.calls]==['freeze','stage','stage','stage']
    assert len(result['contexts'])==3 and not result['training_jobs_reserved'] and not result['dispatch_authorized']
    for i,pin in enumerate(result['contexts']):
        context=json.loads(t.checked(pin).read_text());draft=json.loads(t.checked(context['baseline_profile']).read_text())
        assert context['snapshot_receipt']==b.q['snapshot'] and draft['training_seed']==21+i
        assert draft['r6_training_plan']==b.built['bundle']['plans'][i]
    events=[e for e in b.j.read()['events'].values() if e['identity']['role']==t.ROLE]
    assert len(events)==1 and events[0]['purpose']=='confirmation_delivery'


def test_interrupted_second_review_resumes_without_repeating_prior_reviews(ready):
    b=ready;original=b.review.side_effect;failed=[]
    def review(*args,**kwargs):
        q=json.loads(t.checked(args[3]).read_text())
        if q['kind']=='stage':
            p=json.loads(t.checked(q['plan']).read_text())
            if p['training_seed']==22 and not failed:
                failed.append(True)
                state=b.j.read();state['current_model_event']='second-stage-fixture';b.j.save(state)
                raise RuntimeError('second stage in flight')
            if p['training_seed']==22 and failed:
                state=b.j.read();assert state['current_model_event']=='second-stage-fixture'
                state['current_model_event']=None;b.j.save(state)
        return original(*args,**kwargs)
    b.review.side_effect=review
    with pytest.raises(RuntimeError,match='in flight'):run(b)
    assert not list((b.j.directory/'confirmation-preparation').rglob('result.json'))
    assert run(b)['status']=='THREE_CONFIRMATION_CONTEXTS_READY_NOT_RESERVED'
    assert [q['kind'] for q in b.calls]==['freeze','stage','stage','stage']


@pytest.mark.parametrize('change,reason',[('purpose','PURPOSE'),('source','FROZEN_RECIPE_SOURCE'),('decision','PARENT_DECISION')])
def test_wrong_authority_or_source_rejects_before_reservation(ready,change,reason):
    b=ready
    if change=='purpose':
        c=json.loads(t.checked(b.q['context']).read_text());c['runner']['purpose']='development'
        b.q['context']=b.write('wrong-context.json',c)
    elif change=='source':b.q['recipe']=b.write('wrong-recipe.json',{'source_identity':{'workload_commit':'b'*40}})
    else:t.profile.parent_decision.return_value={'decision':{'decision':'PROPOSE_CANDIDATE'}}
    b.input=b.write('wrong-input.json',b.q);budget=b.j.read()['budget']
    with pytest.raises(RecoveryError,match=reason):run(b)
    assert b.j.read()['budget']==budget and not b.calls


def test_same_parent_cannot_replace_confirmation_input(ready):
    b=ready;run(b);b.q['wall_seconds']=101;b.input=b.write('changed-input.json',b.q)
    with pytest.raises(RecoveryError,match='INPUT_DRIFT'):run(b)


def test_stop_during_validation_never_publishes_complete_result(ready):
    b=ready;b.validate.side_effect=lambda *a:(b.j.directory/'STOP').touch()
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert not list((b.j.directory/'confirmation-preparation').rglob('result.json'))


def test_cached_review_pin_drift_rejected(ready):
    b=ready;run(b)
    event=next(e for e in b.j.read()['events'].values() if e['identity']['role']==t.ROLE)
    t.checked(event['stage-0']).write_text('{}')
    with pytest.raises(RecoveryError,match='PIN_DRIFT'):run(b)


@pytest.mark.parametrize('fault',[None,'plan','issue','seed','gate','rng','checkpoint','binding'])
def test_collection_requires_exact_plan_and_all_training_audits(ready,monkeypatch,fault):
    from tools import r6_analysis_binding
    b=ready;plan=dict(experiment_id='R6-C1',phase='confirmation',training_seed=21,start_update=640,
        target_update=896,checkpoint_updates=[896])
    c=b.write('collect-context.json',{'baseline_profile':b.write('collect-profile.json',{'r6_training_plan':plan})})
    summary=dict(plan,issue_count=0,probability_gate_recomputed=True,rng_initialization_audit={'status':'PASS'},
        checkpoints=[dict(update=896,metadata_verified=True,state_audit={'actor_critic_state_loaded':True,'adam_state_loaded':True})])
    if fault=='seed':summary['training_seed']=22
    if fault=='gate':summary['probability_gate_recomputed']=False
    if fault=='rng':summary['rng_initialization_audit']['status']='FAIL'
    if fault=='checkpoint':summary['checkpoints'][0]['state_audit']['adam_state_loaded']=False
    report=b.write('collect-report.json',dict(summary=summary,issues=[]))
    bound=dict(schema_version='r6-bound-training-analysis/1',training_plan_sha256=t.sha(plan),issues=[],report=report)
    if fault=='plan':bound['training_plan_sha256']='0'*64
    if fault=='issue':bound['issues']=['INCOMPLETE']
    job={'analysis':b.write('collect-bound.json',bound)}
    monkeypatch.setattr(b.j,'canonical_job',lambda key:job)
    monkeypatch.setattr(r6_analysis_binding,'bind_analysis',lambda *a:{} if fault=='binding' else bound)
    with b.j.writer():
        if fault:
            with pytest.raises(RecoveryError):t.collect(b.j,'job',c,b.store)
        else:
            result=t.collect(b.j,'job',c,b.store)
            assert result['training_seed']==21 and result['analysis']==job['analysis'] and not result['scientific_acceptance']
