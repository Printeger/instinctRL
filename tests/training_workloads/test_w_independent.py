"""CPU regression against accepted raw data plus fail-closed protocol tests."""
import json
import hashlib
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import pytest
from training_workloads.unified_h1 import w_independent as w
from training_workloads.unified_h1 import w_closure as primary
from training_workloads.unified_h1 import formal_segment
from training_workloads.unified_h1.evaluation import CASES, replay_raw_case, evaluate_checkpoint
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/experiments/r5-w-closure-auto'

def pin(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,obj):p.write_text(json.dumps(obj));return pin(p)
def profile(tmp_path):
    proto=BASE/'experiments/R5-W-L1-INDEPENDENT-01/INDEPENDENT_PROTOCOL.json'
    r=json.loads(proto.read_text());a=json.loads(Path(r['primary_authorization']['path']).read_text())
    p=deepcopy(a['parameters']);p.update(experiment_id=w.EXPERIMENT,evaluation_seed=w.SEED,evaluation_updates=[640],checkpoints=[r['checkpoint']],independent_validation_protocol=pin(proto),checkpoint_compatibility=write(tmp_path/'compat.json',{}))
    p.pop('experiment_card');p['experiment_card']=write(tmp_path/'card.json',dict(experiment_id=w.EXPERIMENT,profiles={'R1':{k:v for k,v in p.items() if k!='checkpoints'}}))
    return p

def test_exact_protocol_and_budget(tmp_path):
    p=profile(tmp_path)
    assert formal_segment.validate_parameters(primary.EVAL,p)==p
    b=formal_segment.required_budget(primary.EVAL,p)
    assert (b['nontraining_transitions'],b['explicit_resets'],b['training_transitions'],b['optimizer_updates'],b['checkpoint_writes'])==(32960,5,0,0,0)
    r,original,c=w.protocol(p)
    assert formal_segment.method_hashes({},p)==primary.method_hashes(c,original)
    with pytest.raises(PermissionError):primary.validate_parameters(primary.EVAL,p)
    changed_primary=dict(original,evaluation_seed=w.SEED)
    with pytest.raises(PermissionError):primary.validate_parameters(primary.EVAL,changed_primary)
    with pytest.raises(PermissionError):formal_segment.validate_parameters(primary.TRAIN,p)

@pytest.mark.parametrize('key,value',[('evaluation_seed',20260909),('evaluation_seed',1),('ics_clearance_margin_m',.16),('actor_lr',3e-4),('evaluation_updates',[768]),('experiment_id','R5-W-OTHER')])
def test_method_seed_node_mutation_rejected(tmp_path,key,value):
    p=profile(tmp_path);p[key]=value
    with pytest.raises(PermissionError):formal_segment.validate_parameters(primary.EVAL,p)

@pytest.mark.parametrize('field',['product_commit','product_tree','workload_tree','config_sha256'])
def test_candidate_source_mutation_rejected(tmp_path,field):
    p=profile(tmp_path);r=json.loads(Path(p['independent_validation_protocol']['path']).read_text());r['candidate_source_identity'][field]='a'*len(r['candidate_source_identity'][field]);p['independent_validation_protocol']=write(tmp_path/'bad.json',r)
    with pytest.raises(PermissionError):w.protocol(p)

@pytest.fixture(scope='module')
def traces():
    spec=json.loads((BASE/'experiments/R5-W-L1-EVAL-01/inputs/R1-research-formal-evaluation.authorized.json').read_text())
    raw=Path(spec['h1']['attempt'])/'results/research-formal-evaluation/evaluation_raw/w_R1_u640'
    return [t for c in CASES for t in replay_raw_case(str(raw/(c+'.jsonl')),case=c)]

def test_equivalent_scoring_preserves_physics_and_raw(traces):
    independent=[replace(t,seed=w.SEED) for t in traces]
    actual=w.evaluate_independent(independent,update=640)
    assert actual['original_metric_result']==evaluate_checkpoint(traces,update=640)
    assert not evaluate_checkpoint(independent,update=640)['all_gates_pass']
    assert all(t.seed==w.SEED for t in independent)
    for old,new in zip(traces,independent):
        assert all(getattr(old,k) is getattr(new,k) for k in vars(old) if k!='seed')
    for bad in (independent[:-1], independent[:-1]+[independent[0]],traces):
        with pytest.raises(ValueError):w.evaluate_independent(bad,update=640)
    altered=deepcopy(independent);altered[0].command_b[0,0]+=1
    assert not w.evaluate_independent(altered,update=640)['original_metric_result']['all_gates_pass']

def test_backend_loads_original_candidate_and_all_cases(tmp_path,monkeypatch,traces):
    from types import SimpleNamespace
    p=profile(tmp_path);r,original,c=w.protocol(p);calls=[]
    raw=tmp_path/'evaluation_raw/w_independent_R1_u640';raw.mkdir(parents=True)
    (raw/'W.jsonl').write_text(json.dumps({'kind':'T0'})+'\n'+'\n'.join(json.dumps({'root_state_w':[[0.]*7+[.2,.2,0.] for _ in range(32)]}) for _ in range(11)))
    actor=SimpleNamespace(eval=lambda:None)
    backend=SimpleNamespace(research=p,authorization={'mode':primary.EVAL},output=tmp_path,trainer=SimpleNamespace(actor=actor,critic=actor,update_counter=640),_evaluation_rows={})
    def load(path,**kwargs):
        assert path==Path(r['checkpoint']['path'])
        assert kwargs['source_identity']==r['candidate_source_identity']
        assert kwargs['expected_method_hashes']==primary.method_hashes(c,original)
        return {'training_seed':20260908,'curriculum':{'version':'curriculum-h1-v1'}}
    def evaluate(case,**kwargs):
        calls.append(case);backend._evaluation_rows[case]=[replace(t,seed=w.SEED) for t in traces if t.case==case]
    backend._load_evaluation_checkpoint=load;backend._evaluate_case=evaluate;backend.resource_check=lambda *args:None
    monkeypatch.setattr(primary,'apply_profile',lambda b:None)
    w.evaluate_segment(backend)
    assert calls==list(CASES)
    assert json.loads((tmp_path/'w_independent_R1_u640.json').read_text())['original_metric_result']['all_gates_pass']


def test_exact_recovery_identity_only(tmp_path):
    p=profile(tmp_path)
    p['experiment_id']=w.RECOVERY_EXPERIMENT
    p['experiment_card']=write(tmp_path/'recovery-card.json',dict(
        experiment_id=w.RECOVERY_EXPERIMENT,
        profiles={'R1':{k:v for k,v in p.items() if k not in ('checkpoints','experiment_card')}}))
    assert w.validate_parameters(primary.EVAL,p)==p
    assert w.required_budget(primary.EVAL,p)['nontraining_transitions']==32960
    for key,value in [('experiment_id','R5-W-L1-INDEPENDENT-RECOVERY2'),('evaluation_seed',20261021),('evaluation_updates',[768]),('ics_clearance_margin_m',.16)]:
        with pytest.raises(PermissionError):w.validate_parameters(primary.EVAL,dict(p,**{key:value}))
