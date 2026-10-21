"""Real Journal/storage plans/fake CLI receipts; frozen publication and process validation are fixtures."""
import json,sys
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools import r6_final_storage as s
from tools import r6_final_preparation as prep
from tools import r6_final_compatibility_review as review
from tools.test_r6_final_preparation import ready
from tools.test_r6_profile_preparation import case
from tools.test_r6_final_compatibility_review import SCRIPT
from tools.r6_recovery import file_pin,RecoveryError
from tools.r6_batch import model_handoff
REAL_REVIEW=review.prepare

@pytest.fixture
def prepared(ready,monkeypatch):
    b=ready;b.builder.write_text('#!'+sys.executable+'\n'+SCRIPT);b.builder.with_suffix('.settings').write_text('{}')
    monkeypatch.setattr(review,'prepare',REAL_REVIEW)
    monkeypatch.setattr(review,'validate_batch',lambda *a:b.value)
    def inputs(snapshot,q):
        return dict(status='REVIEW_PAYLOAD_PREPARED_NOT_ACCEPTED',snapshot=snapshot,request=q,source_identity={},
            payloads={str(i):{'fixture':i} for i in range(len(b.value['assignments']))},dispatch_authorized=False)
    monkeypatch.setattr(review.inputs,'prepare',inputs)
    monkeypatch.setattr(review,'model_handoff',lambda j,k:model_handoff(j,k,poll_seconds=.01))
    with b.j.writer():b.result=prep.prepare(b.j,b.snapshot,b.input,b.store,executable=file_pin(b.builder),free_disk_bytes=500*1024**3)
    b.pin=b.j.read()['events'][b.result['event_key']]['result']
    return b


def run(b,index=0,**kw):
    with b.j.writer():return s.prepare(b.j,b.pin,index,free_disk_bytes=500*1024**3,**kw)


def test_exact_profile_is_prepaid_once_without_model_or_mount(prepared):
    b=prepared;before=b.store.replay();first=run(b);budget=b.j.read()['budget']
    assert run(b)==first and b.j.read()['budget']==budget and b.store.replay()==before
    assert b.builder.with_suffix('.count').read_text()=='call\n'
    q=json.loads(s.checked(first['inputs']).read_text());context=json.loads(s.checked(b.result['contexts'][0]).read_text())
    assert q['profile_draft']==context['baseline_profile'] and q['snapshot_receipt']==b.snapshot
    owner,native,result,cap=s.storage.validate(b.j,first['storage_key'])
    assert owner['purpose']=='confirmation_delivery' and result['minimum_free_bytes']==b.inputs['payload_bytes']
    assert cap==native['maximum_backing_bytes']+result['maximum_backing_bytes']+s.storage.CONTROL_ALLOWANCE
    assert not Path(native['image']).exists() and not Path(result['image']).exists()
    assert not first['dispatch_authorized']


def test_each_assignment_gets_distinct_storage_and_attempt(prepared):
    b=prepared;outputs=[run(b,i) for i in range(len(b.result['contexts']))]
    assert len({v['storage_key'] for v in outputs})==len(outputs)
    plans=[s.storage.validate(b.j,v['storage_key']) for v in outputs]
    assert len({p[2]['target'] for p in plans})==len(outputs)
    assert b.builder.with_suffix('.count').read_text()=='call\n'


@pytest.mark.parametrize('fault',['index','bool','inodes','stop','frozen','receipt'])
def test_invalid_source_or_receipt_prevents_storage_charge(prepared,fault):
    b=prepared;index=0;kw={}
    if fault=='index':index=len(b.result['contexts'])
    elif fault=='bool':index=False
    elif fault=='inodes':kw['minimum_free_inodes']=True
    elif fault=='stop':(b.j.directory/'STOP').touch()
    elif fault=='frozen':b.validate.side_effect=RecoveryError('frozen profile drift')
    elif fault=='receipt':
        path=Path(b.result['compatibility_review']['reviews']['0']['path']);path.write_text('{}')
    before=b.j.read()['budget']
    with pytest.raises(RecoveryError):run(b,index,**kw)
    assert b.j.read()['budget']==before
    assert not any(e['identity']['role']==s.storage.ROLE for e in b.j.read()['events'].values())


def test_same_assignment_cannot_change_inodes_on_replay(prepared):
    b=prepared;run(b)
    with pytest.raises(RecoveryError,match='INPUT_DRIFT'):run(b,minimum_free_inodes=768)


def test_full_batch_materialization_reports_all_missing_volumes_and_reuses_storage(prepared,monkeypatch):
    from tools import r6_materialize_job as materializer
    b=prepared;calls=[];mounted=set()
    def materialize(journal,inputs,store,**kw):
        q=json.loads(s.checked(inputs).read_text());p=json.loads(s.checked(q['profile_draft']).read_text());i=p['final_selection']['assignment_index'];calls.append(i)
        if i not in mounted:return dict(status='WAITING_FOR_ADMIN_PROVISION',missing=['native','result'],dispatch_authorized=False)
        return dict(status='EXACT_INPUTS_MATERIALIZED_NOT_AUTHORIZED',artifacts={'plan':inputs},dispatch_authorized=False)
    monkeypatch.setattr(materializer,'materialize',materialize)
    def all_plans():
        with b.j.writer():return s.materialize_all(b.j,b.pin,b.store,free_disk_bytes=500*1024**3)
    first=all_plans();count=len(b.result['contexts']);budget=b.j.read()['budget']
    assert calls==list(range(count)) and len(first['missing'])==count
    mounted.add(0);second=all_plans();assert len(second['missing'])==count-1
    mounted.update(range(count));last=all_plans()
    assert last['status']=='FINAL_BATCH_PLANS_READY_NOT_AUTHORIZED' and not last['dispatch_authorized']
    assert len(last['assignments'])==count and b.j.read()['budget']==budget
    assert [r['storage'] for r in first['assignments']]==[r['storage'] for r in last['assignments']]
    assert b.builder.with_suffix('.count').read_text()=='call\n'
