"""Real Journal/storage/review receipts; frozen spec inspection is a CPU substitute."""
from unittest.mock import Mock
import pytest
from rlflow.store import sha
from tools import r6_final_plans as m
from tools.test_r6_final_storage import prepared
from tools.test_r6_final_preparation import ready
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import RecoveryError

FREE=500*1024**3

@pytest.fixture
def batch(prepared, monkeypatch):
    b=prepared;rows=[]
    monkeypatch.setattr(m.experiments,'validate',Mock())
    monkeypatch.setattr(m.storage.storage,'for_spec',Mock(return_value={'prepaid':True}))
    with b.j.writer():
        for i in range(len(b.result['contexts'])):
            planned=m.storage.prepare(b.j,b.pin,i,free_disk_bytes=FREE)
            inputs=m.read(planned['inputs']);context=m.read(b.result['contexts'][i]);draft=m.read(context['baseline_profile'])
            # The fixture snapshot has no physical tree; validation remains substituted.
            snapshot=m.read(b.snapshot);snapshot['snapshot']=str(b.j.directory/'snapshot')
            # Preserve the pinned input; make this test-only field available through read.
            params=dict(draft,native_volume=b.write(f'native{i}.json',{}),result_volume=b.write(f'result{i}.json',{}),output_cap_bytes=1000)
            card=b.write(f'card{i}.json',dict(experiment_id=draft['experiment_id'],profiles={'R1':params}))
            params=dict(params,experiment_card=card)
            auth=dict(mode=m.charges.MODES[3],parameters=params,budgets=dict(nontraining_transitions=193600,training_transitions=0,optimizer_updates=0,checkpoint_writes=0))
            ap=b.write(f'auth{i}.json',auth)
            spec=dict(id=draft['experiment_id'],backend='real',stage_id=context['stage_id'],conditions=context['conditions'],
                pins=dict(authorization=ap),cwd=snapshot['snapshot'],h1=dict(mode=m.charges.MODES[3],attempt=str(b.j.directory/f'attempt{i}')),
                resources=dict(wall_seconds=70,transitions=193600))
            sp=b.write(f'spec{i}.json',spec)
            artifacts=dict(card=card,parameters=b.write(f'params{i}.json',params),authorization=ap,
                authorization_proposal=b.write(f'proposal{i}.json',dict(authorization_proposal=auth)),spec=sp,
                plan=b.write(f'plan{i}.json',dict(context['runner'],spec_pin=sp)))
            request=dict(inputs=planned['inputs'],implementation=m.materializer.implementation(),store_root=str(b.store.root.resolve()))
            event=dict(goal_id=b.j.read()['goal_id'],event_id=f'MATERIALIZE:{i}',job_id=f'MATERIALIZE:{i}',input_sha256=sha(request),role=m.materializer.ROLE)
            key,_=b.j.intent(event,kind='cpu',request=request,resources={m.charges.DISK:m.materializer.CAP},purpose='confirmation_delivery',free_disk_bytes=FREE)
            value=dict(status='EXACT_INPUTS_MATERIALIZED_NOT_AUTHORIZED',event_key=key,inputs=planned['inputs'],storage_key=planned['storage_key'],
                job_id='J-'+sha(dict(id=spec['id'],spec=spec))[:16],artifacts=artifacts,scientific_acceptance=False,dispatch_authorized=False)
            state=b.j.read();state['events'][key]['materialization']=b.write(f'material{i}.json',value);b.j.save(state)
            rows.append(dict(assignment_index=i,storage=planned,materialization=value))
    original=m.read
    monkeypatch.setattr(m,'read',lambda pin: snapshot if pin==b.snapshot else original(pin))
    b.aggregate=dict(status='FINAL_BATCH_PLANS_READY_NOT_AUTHORIZED',preparation=b.pin,batch=b.result['batch'],assignments=rows,
        missing=[],scientific_acceptance=False,dispatch_authorized=False)
    return b


def run(b):
    pin=b.write('aggregate.json',b.aggregate)
    with b.j.writer():return m.describe(b.j,pin,b.store,free_disk_bytes=FREE)


def test_complete_batch_budget_and_replay_are_read_only(batch):
    b=batch;before=b.j.read();store=b.store.replay();result=run(b)
    assert run(b)==result and b.j.read()==before and b.store.replay()==store
    assert result['planned_job_resources']['nontraining_transitions']==193600*len(b.aggregate['assignments'])
    assert result['planned_job_resources'][m.charges.DISK]==0
    assert not result['dispatch_authorized'] and not result['scientific_acceptance']


@pytest.mark.parametrize('fault',['missing','count','order','storage_receipt','material_receipt','parameters','plan','spec','budget','stop'])
def test_incomplete_or_unbound_batches_rejected_without_side_effects(batch,fault):
    b=batch;row=b.aggregate['assignments'][0]
    if fault=='missing':b.aggregate['missing']=[0]
    elif fault=='count':b.aggregate['assignments'].pop()
    elif fault=='order':row['assignment_index']=1
    elif fault=='storage_receipt':row['storage']['storage_key']='wrong'
    elif fault=='material_receipt':row['materialization']['job_id']='wrong'
    elif fault in ('parameters','plan','spec'):
        artifacts=row['materialization']['artifacts'];value=m.read(artifacts[fault])
        if fault=='parameters':value['training_seed']=999
        elif fault=='plan':value['purpose']='development'
        else:value['stage_id']='wrong'
        artifacts[fault]=b.write('wrong.json',value)
        # Keep the material receipt exact so the deeper binding is exercised.
        with b.j.writer():
            state=b.j.read();state['events'][row['materialization']['event_key']]['materialization']=b.write('changed-material.json',row['materialization']);b.j.save(state)
    elif fault=='budget':
        with b.j.writer():
            state=b.j.read();state['budget']['caps']['nontraining_transitions']=1;b.j.save(state)
    elif fault=='stop':(b.j.directory/'STOP').touch()
    before=b.j.read();store=b.store.replay()
    with pytest.raises((RecoveryError,ValueError)):run(b)
    assert b.j.read()==before and b.store.replay()==store
