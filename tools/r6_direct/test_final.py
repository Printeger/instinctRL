"""Local final routing with explicit fixture checkpoint/freeze verification boundaries."""
import copy
import json
from pathlib import Path
import pytest
from tools.test_r6_training_plan import case
from tools.test_r6_evaluation_selection import selection
from tools.test_r6_final_selection import ready
from training_workloads.unified_h1 import robustness_final_selection as final
from training_workloads.unified_h1 import direct_selection, direct_contract


def local_batch(b, monkeypatch):
    models={}
    for index in range(1,4):
        value=dict(b.value,assignment_index=index)
        models[b.plans[index-1]['training_seed']]=final.validate(value)['model']
    for stream,plan in zip(b.freeze['evidence']['streams'],b.plans):
        stream.pop('context')
        local=copy.deepcopy(plan);local['schema_version']='r6-local-training-plan/1';local.pop('recipe_review')
        local['experiment_id']='R6-DIRECT-CONFIRM-'+str(plan['training_seed'])
        stream.update(plan=local,baseline_bindings=b.ev,plan_sha256=final.base.object_sha(local))
    b.value=b.final_save()
    b.batch['schema_version']='r6-local-final-batch-inputs/1';b.batch.pop('holdout_event_key')
    value=dict(b.value,schema_version='r6-local-final-selection/1',batch=b.pin('local-batch.json',b.batch))
    # Tensor/evidence auditing is covered by test_continuation; here only the
    # final selector is exercised. Publication bytes have a separate real Git test.
    monkeypatch.setattr(direct_selection,'training_model',lambda plan,*_:models[plan['training_seed']])
    monkeypatch.setattr(direct_selection,'verify_publication',lambda *pins:None)
    return value


def test_local_final_model_set_and_common_holdout_stream(ready,monkeypatch):
    b=ready;value=local_batch(b,monkeypatch)
    selected=[final.validate(dict(value,assignment_index=i)) for i in range(4)]
    assert [s['assignment']['model_role'] for s in selected]==['baseline','seed1','seed2','seed3']
    assert len({s['comparison']['seed'] for s in selected})==1
    assert all(s['comparison']['split']=='final' for s in selected)
    bad=dict(value,assignment_index=4)
    with pytest.raises(PermissionError):final.validate(bad)
    b.batch['assignments'][1]['evaluation_seed']+=1
    with pytest.raises(PermissionError,match='ASSIGNMENTS'):
        final.validate(dict(value,batch=b.pin('wrong-stream.json',b.batch)))


def test_final_contract_keeps_all_registered_conditions(ready,monkeypatch):
    b=ready
    baseline=json.loads(Path('docs/experiments/r6-direct-training/BASELINE_CONTRACT.json').read_text())
    b.freeze['protocol']=baseline['parameters']['r6']['protocol'];b.value=b.final_save()
    value=local_batch(b,monkeypatch)
    monkeypatch.setattr(direct_contract,'baseline',final.base.baseline)
    p=copy.deepcopy(baseline['parameters']);p.update(experiment_id='R6-DIRECT-FINAL-CPU',training_seed=21,
        baseline_bindings=b.ev,final_selection=value)
    p['r6'].update(run_id=p['experiment_id'],seed=b.batch['assignments'][1]['evaluation_seed'],split='final')
    contract=direct_contract.evaluation_runtime_contract(p,final=True)
    assert contract['operation']=='final' and contract['budgets']['nontraining_transitions']==193600
    assert contract['budgets']['training_transitions']==0
    assert direct_contract.validate_runtime_contract(contract)==contract
    wrong=copy.deepcopy(p);wrong['r6']['block_ids'].pop()
    with pytest.raises(ValueError):direct_contract.evaluation_runtime_contract(wrong,final=True)
