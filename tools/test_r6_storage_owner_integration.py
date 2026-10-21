"""Storage-owner wiring; temporary ledger/Store, mocked authority and mounts."""
import json
from unittest.mock import Mock
import pytest
from rlflow import experiments
from rlflow.store import Store,sha
from tools import r6_storage_allocation as allocation
from tools.r6_register_job import register as register_job
from tools.r6_submit import submit_once
from tools.r6_budget import totals
from tools.r6_recovery import file_pin,RecoveryError
from tools.test_r6_storage_allocation import case,spec,register
from tools.test_r6_native_volume_inventory import native_job


@pytest.fixture
def ready(case,monkeypatch,tmp_path):
    b=case;b.storage_key=register(b);value,pin=spec(b,monkeypatch)
    value['resources']=dict(wall_seconds=70,transitions=512)
    b.spec_value=value;b.spec_pin=b.write('spec.json',value);b.store=Store(tmp_path/'store')
    with b.j.writer():
        state=b.j.read();state['budget']['capacity_pass']=True;b.j.save(state)
    monkeypatch.setattr(experiments,'validate',Mock())
    monkeypatch.setattr(experiments,'authorized',Mock(return_value={'approval_id':'CPU-FIXTURE'}))
    monkeypatch.setattr('tools.r6_submit.queue_ready',Mock(return_value=b.n))
    def submit(store,value):
        jid='J-'+sha({'id':value['id'],'spec':value})[:16]
        store.transact('CPU_FIXTURE','submit',lambda s:s['jobs'].update({jid:dict(id=jid,spec=value,approval_id='CPU-FIXTURE',status='FAILED')}))
    b.submit=Mock(side_effect=submit);monkeypatch.setattr(experiments,'submit',b.submit)
    return b


def register_ready(b):
    with b.j.writer():return register_job(b.j,b.spec_pin,b.store,free_disk_bytes=500*1024**3)


def test_registration_submit_and_replay_use_one_storage_charge(ready):
    b=ready;before=totals(b.j.read()['budget'])[allocation.DISK];key=register_ready(b)
    state=b.j.read();assert state['events'][key]['resources'][allocation.DISK]==0
    assert totals(state['budget'])[allocation.DISK]==before
    assert register_ready(b)==key
    with b.j.writer():first=submit_once(b.j,key,b.store,queue_key='CPU')
    with b.j.writer():assert submit_once(b.j,key,b.store,queue_key='CPU')==first
    assert totals(b.j.read()['budget'])[allocation.DISK]==before
    assert b.submit.call_count==1


@pytest.mark.parametrize('change',['missing','charge','purpose','binding'])
def test_prepaid_owner_drift_blocks_submission(ready,change):
    b=ready;key=register_ready(b)
    with b.j.writer():
        s=b.j.read();owner=s['events'][b.storage_key]
        if change=='missing':del s['events'][b.storage_key]
        elif change=='charge':owner['resources'][allocation.DISK]-=1
        elif change=='purpose':owner['purpose']='repair'
        else:owner['storage_binding']['job_id']='OTHER'
        b.j.save(s)
    with b.j.writer(),pytest.raises(RecoveryError):submit_once(b.j,key,b.store,queue_key='CPU')
    b.submit.assert_not_called()
    assert not b.j.read()['events'][key].get('submission_claim')


def test_crash_after_binding_before_job_intent_recovers_without_storage_refund(ready,monkeypatch):
    b=ready;original=b.j.intent
    monkeypatch.setattr(b.j,'intent',Mock(side_effect=RuntimeError('CPU crash before job intent')))
    with pytest.raises(RuntimeError):register_ready(b)
    s=b.j.read();assert s['events'][b.storage_key]['storage_binding']
    assert len(s['budget']['reservations'])==1
    monkeypatch.setattr(b.j,'intent',original);register_ready(b)
    assert len(b.j.read()['budget']['reservations'])==2
    assert totals(b.j.read()['budget'])[allocation.DISK]==totals(s['budget'])[allocation.DISK]


@pytest.mark.parametrize('native_job',[True],indirect=True)
@pytest.mark.parametrize('cap',[1024**3,2*1024**3])
def test_failed_inventory_observes_prepaid_owner_and_preserves_overrun(native_job,monkeypatch,cap):
    b=native_job
    with b.j.writer():
        s=b.j.read();job=s['events'][b.key];job['resources'][allocation.DISK]=0
        next(r for r in s['budget']['reservations'] if r['identity']==job['identity'])['resources'][allocation.DISK]=0
        b.j.save(s)
        request={'fixture':'inventory owner boundary'}
        event=dict(goal_id=s['goal_id'],event_id='CPU-STORAGE',job_id='CPU-STORAGE',input_sha256=sha(request),role='cpu-fixture')
        owner_key,_=b.j.intent(event,kind='cpu',request=request,resources={allocation.DISK:cap},purpose='development',free_disk_bytes=500*1024**3)
    # Owner validation is covered above; exercise actual inventory/observation here.
    monkeypatch.setattr(allocation,'owner',lambda journal,job:journal.read()['events'][owner_key])
    result=b.record();assert result['storage_allocation_event_key']==owner_key
    assert result['native_status']=='FAILED' and result['native_exit_code']==-9
    charged=result['total_charged_output_bytes'];state=b.j.read()
    assert totals(state['budget'])[allocation.DISK]==max(cap,charged)
    assert bool(state['events'][b.key]['inventory_violations'])==(charged>cap)
    assert b.record()==result and b.j.read()==state
