"""Real Journal/artifacts, synthetic full-denominator reports; closed-chain recheck is substituted."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from rlflow.store import sha
from tools import r6_delivery_evidence as m
from tools.test_r6_final_collection import batch,collect
from tools.r6_recovery import Journal,file_pin,RecoveryError

FREE=500*1024**3

@pytest.fixture
def delivery(batch,tmp_path,monkeypatch):
    b=batch;final=collect(b)
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return file_pin(p)
    budget=json.loads(Path('tools/fixtures/r6_budget.json').read_text())
    j=Journal(tmp_path/'journal');store=SimpleNamespace(root=tmp_path/'store');keys=[]
    freeze=write('delivery-freeze.json',dict(models={r['assignment']['model_role']:r['assignment']['checkpoint'] for r in final['results']},
        branch='BASELINE_SUFFICIENT' if len(b.keys)==3 else 'THREE_CONTINUATIONS'))
    batch_value=m.read(b.pin);batch_value['freeze']=freeze
    bp=write('delivery-batch.json',batch_value);final['batch']=bp
    with j.writer():
        j.initialize(budget)
        for i,row in enumerate(final['results']):
            auth=write(f'delivery-auth-{i}.json',b.auths[i])
            spec=write(f'delivery-spec-{i}.json',dict(pins={'authorization':auth},argv=['fixture'],cwd=str(tmp_path),id=str(i)))
            request=dict(spec=spec,wait_deadline_unix=1)
            event=dict(goal_id=budget['goal_id'],event_id='DELIVERY-FIXTURE-'+str(i),job_id=row['job_id'],input_sha256=sha(request),role='job')
            key,_=j.intent(event,kind='job',request=request,resources={'live_starts':1},purpose='confirmation_delivery',free_disk_bytes=FREE)
            state=j.read();state['events'][key].update(status='JOB_CLOSED',analysis=b.entries[str(i)]['analysis'],
                closure=write(f'closure-{i}.json',dict(native_status='FAILED',native_exit_code=-9)),
                inventory=write(f'inventory-{i}.json',dict(files=[])),postrun=write(f'postrun-{i}.json',dict(receipts=[])))
            j.save(state);row['job_event_key']=key;keys.append(key)
        request=dict(store_root=str(store.root.resolve()));event=dict(goal_id=budget['goal_id'],event_id='CONTROLLER',job_id='CONTROLLER',input_sha256=sha(request),role='r6-controller')
        controller,_=j.intent(event,kind='cpu',request=request,resources={'output_actual_plus_unreleased_reservations_bytes':256*1024},purpose='development',free_disk_bytes=FREE)
        state=j.read();state['events'][controller]['cursor']=dict(final_collection=dict(status='FINAL_BATCH_COLLECTED_REQUIRES_INDEPENDENT_AUDIT',collection=final,completed_jobs=keys));j.save(state)
    monkeypatch.setattr(m.collection,'collect',Mock(return_value=final))
    return SimpleNamespace(j=j,controller=controller,store=store,final=final,keys=keys,write=write)


def run(b):
    with b.j.writer():return m.prepare(b.j,b.controller,b.store,free_disk_bytes=FREE)


def test_all_jobs_failed_native_and_all_blocks_preserved_without_acceptance(delivery):
    b=delivery;result=run(b);budget=b.j.read()['budget'];assert run(b)==result and b.j.read()['budget']==budget
    assert not result['goal_complete'] and not result['scientific_acceptance']
    assert 'LOCAL_WAIT_EVIDENCE_GAPS' in result['pending_requirements']
    manifest=m.read(result['artifacts']['TRAINING_AND_EVALUATION_MANIFEST.json'])
    assert manifest['actual_job_count']==len(b.keys) and all(r['native_status']=='FAILED' for r in manifest['jobs'])
    report=m.checked(result['artifacts']['FINAL_R6_RESULTS.md']).read_text()
    assert report.count('/32 |')==16*len(b.keys)
    audit=m.read(result['artifacts']['MODEL_USAGE_AUDIT.json'])
    assert audit['pure_wait_model_calls']=='UNKNOWN' and not audit['platform_inactivity_proven']
    resources=m.read(result['artifacts']['RESOURCE_AUDIT.json'])
    assert resources['charged']['output_actual_plus_unreleased_reservations_bytes']>=m.CAP


def test_missing_prior_attempt_evidence_remains_visible(delivery):
    b=delivery
    with b.j.writer():
        state=b.j.read();state['events'][b.keys[0]].pop('inventory');b.j.save(state)
    out=run(b);assert 'ATTEMPT_EVIDENCE_GAPS' in out['pending_requirements']
    assert 'inventory' in m.read(out['artifacts']['TRAINING_AND_EVALUATION_MANIFEST.json'])['jobs'][0]['missing_evidence'] or any(
        'inventory' in j['missing_evidence'] for j in m.read(out['artifacts']['TRAINING_AND_EVALUATION_MANIFEST.json'])['jobs'])


@pytest.mark.parametrize('fault',['stop','collection','artifact'])
def test_drift_or_stop_rejects(delivery,fault):
    b=delivery
    if fault=='stop':(b.j.directory/'STOP').touch()
    elif fault=='collection':m.collection.collect.return_value={}
    else:
        result=run(b);m.checked(result['artifacts']['FINAL_R6_RESULTS.md']).write_text('false success')
    with pytest.raises(RecoveryError):run(b)


def test_no_jobs_does_not_prove_zero_model_wait(delivery):
    state=delivery.j.read();state['events']={}
    audit=m.model_audit(state)
    assert not audit['all_registered_local_waits_verified'] and audit['pure_wait_model_calls']=='UNKNOWN'


def test_unknown_model_usage_not_counted_as_zero(delivery):
    state=delivery.j.read();state['events']['unknown']={'kind':'model','identity':{'role':'decision'},'purpose':'development','status':'INTENT'}
    value=m.model_audit(state)
    assert value['unknown_usage_events']==['unknown'] and value['reported_token_subtotals']=={}


def test_reported_model_tokens_are_subtotals_with_receipt_binding(delivery):
    b=delivery;state=b.j.read();request={'fixture':'model-request'}
    receipt=b.write('model-receipt.json',dict(event_key='reported',request_sha256=sha(request),
        usage={'availability':'REPORTED','totals':{'input_tokens':17,'output_tokens':3}}))
    state['events']['reported']=dict(kind='model',identity={'role':'review'},request=request,receipt=receipt,purpose='final_review',status='RECEIPT_RECORDED')
    audit=m.model_audit(state)
    assert audit['reported_token_subtotals']=={'input_tokens':17,'output_tokens':3}
    assert audit['platform_token_usage']=='UNKNOWN'
    state['events']['reported']['request']={}
    with pytest.raises(RecoveryError,match='MODEL_RECEIPT'):m.model_audit(state)


def test_wait_record_cannot_be_transplanted_to_another_job(delivery):
    b=delivery;state=b.j.read();key=b.keys[0];entry=state['events'][key]
    entry['local_wait_audit']=dict(schema_version='r6-local-wait-audit/1',event_key=key,job_id=entry['identity']['job_id'],
        scope='REGISTERED_LOCAL_MODEL_WORKERS_ONLY',complete=True,unobserved_restart_gap=False,
        segments=[dict(checks=3,registered_model_workers_present=False,interrupted=False,started_at=1,last_checked_at=3)])
    audit=m.model_audit(state)
    assert next(w for w in audit['waits'] if w['event_key']==key)['registered_local_wait_verified']
    entry['local_wait_audit']['job_id']='wrong-job'
    audit=m.model_audit(state)
    assert not next(w for w in audit['waits'] if w['event_key']==key)['registered_local_wait_verified']
