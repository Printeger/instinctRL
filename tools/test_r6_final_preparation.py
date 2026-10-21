"""Deterministic profiles; published evidence, review and frozen process are explicit substitutes."""
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools.test_r6_profile_preparation import case
from tools import r6_final_preparation as m
from tools.r6_recovery import file_pin,RecoveryError

@pytest.fixture(params=['baseline','confirmation'])
def ready(case,monkeypatch,request):
    b=case;context=json.loads(m.checked(b.context).read_text());ref=json.loads(m.checked(context['baseline_profile']).read_text())
    protocol_value=json.loads(Path('docs/experiments/r6-robustness-auto/freeze-candidate-01/protocol.json').read_text())
    protocol=b.write('final-protocol.json',protocol_value)
    ref.update(evaluation_seed=20260909);ref['r6'].update(protocol=protocol,block_ids=[x['id'] for x in protocol_value['blocks']])
    context['baseline_profile']=b.write('final-reference.json',ref);context['runner']['purpose']='confirmation_delivery'
    b.context=b.write('final-context.json',context)
    freeze=b.write('final-freeze.json',dict(evidence=dict(baseline_bindings=ref['baseline_bindings'],streams=[dict(training_seed=i) for i in (21,22,23)])))
    pairs=[('baseline',17+i) for i in range(3)] if request.param=='baseline' else [(role,17) for role in ('baseline','seed1','seed2','seed3')]
    value=dict(protocol=protocol,freeze=freeze,assignments=[dict(model_role=role,evaluation_seed=seed) for role,seed in pairs])
    bp=b.write('final-batch.json',value)
    monkeypatch.setattr(m.publication,'validate_batch',Mock(return_value=value))
    config=b.write('final-config.json',dict(schema_version='unified-h1-r6-final-config.v1'))
    b.snapshot=b.write('final-snapshot.json',dict(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',config=config))
    monkeypatch.setattr(m.validator,'verify_frozen',Mock());b.validate=Mock();monkeypatch.setattr(m.validator,'validate',b.validate)
    b.review=Mock(return_value=dict(status='EXACT_FINAL_COMPATIBILITY_REVIEWS_RECORDED_NOT_AUTHORIZED',dispatch_authorized=False,
        accepted={str(i):b.write(str(i)+'-compat.json',{}) for i in range(len(pairs))}))
    monkeypatch.setattr(m.review,'prepare',b.review)
    b.inputs=dict(batch=bp,context=b.context,origin_configs=[config]*len(pairs),experiment_ids=['R6-FINAL-'+str(i) for i in range(len(pairs))],
        wall_seconds=60,payload_bytes=128*1024**2)
    b.input=b.write('final-inputs.json',b.inputs);b.value=value
    return b


def run(b):
    with b.j.writer():return m.prepare(b.j,b.snapshot,b.input,b.store,executable=file_pin(b.builder),free_disk_bytes=500*1024**3)


def test_complete_profiles_replay_without_job_reservation(ready):
    b=ready;before=b.store.replay();result=run(b);budget=b.j.read()['budget'];assert run(b)==result and b.j.read()['budget']==budget
    assert b.store.replay()==before and result['status']=='FINAL_BATCH_CONTEXTS_READY_NOT_RESERVED'
    assert not result['dispatch_authorized'] and not result['scientific_acceptance']
    for i,pin in enumerate(result['contexts']):
        context=json.loads(m.checked(pin).read_text());p=json.loads(m.checked(context['baseline_profile']).read_text())
        assert p['final_selection']['assignment_index']==i and p['final_selection']['batch']==b.inputs['batch']
        assert p['r6']['seed']==b.value['assignments'][i]['evaluation_seed'] and p['r6']['split']=='final'
        assert p['r6']['run_id']==p['experiment_id']==b.inputs['experiment_ids'][i]
        assert p['training_seed']==(20260908 if b.value['assignments'][i]['model_role']=='baseline' else 20+i)
        assert context['snapshot_receipt']==b.snapshot and context['runner']['purpose']=='confirmation_delivery'
        assert p['checkpoint_compatibility']==b.review.return_value['accepted'][str(i)]
    assert not any(e['kind']=='job' and e['identity'].get('event_id','').startswith('FINAL') for e in b.j.read()['events'].values())


@pytest.mark.parametrize('fault',['duplicate','missing','purpose','payload','review','frozen','stop'])
def test_invalid_batch_context_or_validation_prevents_ready_result(ready,fault):
    b=ready
    if fault=='duplicate':b.inputs['experiment_ids'][-1]=b.inputs['experiment_ids'][0]
    elif fault=='missing':b.inputs['origin_configs'].pop()
    elif fault=='purpose':
        context=json.loads(m.checked(b.context).read_text());context['runner']['purpose']='development';b.inputs['context']=b.write('wrong-context.json',context)
    elif fault=='payload':b.inputs['payload_bytes']=True
    elif fault=='review':b.review.return_value['accepted'].pop('0')
    elif fault=='frozen':b.validate.side_effect=RecoveryError('frozen source drift')
    elif fault=='stop':(b.j.directory/'STOP').touch()
    b.input=b.write('changed-inputs.json',b.inputs)
    with pytest.raises(RecoveryError):run(b)
    assert not list((b.j.directory/'final-preparation').rglob('result.json'))


def test_same_batch_changed_inputs_cannot_create_second_preparation(ready):
    b=ready;run(b);b.inputs['wall_seconds']+=1;b.input=b.write('second-inputs.json',b.inputs)
    with pytest.raises(RecoveryError,match='INPUT_DRIFT'):run(b)


def test_full_final_batch_capacity_cannot_be_replaced_by_one_job(ready):
    from tools.r6_budget import BudgetError
    b=ready
    with b.j.writer():
        state=b.j.read();state['budget']['used']['nontraining_transitions']=state['budget']['caps']['nontraining_transitions']-193600
        b.j.save(state)
    with pytest.raises(BudgetError):run(b)
    b.review.assert_not_called()
    assert not list((b.j.directory/'final-preparation').rglob('result.json'))
