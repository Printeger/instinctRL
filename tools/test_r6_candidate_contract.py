"""Exact stage fixtures; selection and simulator boundaries are explicit substitutes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools.test_r6_contract import ContractTests
from tools.test_r6_training_plan import case
from tools.test_r6_training_contract import compatibility
from training_workloads.unified_h1 import robustness_candidate_contract as c


@pytest.fixture
def stage(monkeypatch,request):
    b=ContractTests();b.setUp();request.addfinalizer(b.tearDown)
    b.p['training_seed']=11;b.p['evaluation_selection']={'fixture':'independently selected'}
    b.selected=dict(selection=dict(b.p['evaluation_selection'],baseline_bindings=b.p['baseline_bindings']),
        comparison={k:b.p['r6'][k] for k in ('protocol','block_ids','seed','split')},model=dict(training_seed=11))
    monkeypatch.setattr(c.selection,'validate',lambda value:b.selected)
    b.card();return b


def test_stage_uses_all_frozen_denominators_and_no_training_budget(stage):
    from training_workloads.unified_h1 import research_contract
    p=stage.p;assert c.validate_parameters(c.MODE,p)==p
    budget=research_contract.required_budget(c.MODE,p)
    assert budget['nontraining_transitions']==193600 and budget['explicit_resets']==21
    assert budget['training_transitions']==budget['optimizer_updates']==budget['checkpoint_writes']==0
    from tools.r6_job_resources import resources
    charge=resources(dict(backend='real',h1=dict(mode=c.MODE),
        resources=dict(wall_seconds=budget['wall_seconds'],transitions=193600)),
        dict(mode=c.MODE,parameters=p,budgets=budget))
    assert charge['nontraining_transitions']==193600 and 'training_transitions' not in charge
    with pytest.raises(PermissionError):c.base.validate_parameters(c.base.MODE,p)


@pytest.mark.parametrize('fault',['mode','seed','original_seed','protocol','stream','split','blocks','card','selection_missing'])
def test_candidate_cannot_reuse_baseline_or_drift_paired_request(stage,fault):
    b=stage;mode=c.MODE;b.p=copy.deepcopy(b.p)
    if fault=='mode':mode=c.base.MODE
    if fault=='seed':b.p['training_seed']=20260908
    if fault=='original_seed':b.p['evaluation_seed']=17
    if fault=='protocol':b.p['r6']['protocol']=b.write('other-protocol.json',{})
    if fault=='stream':b.p['r6']['seed']+=1
    if fault=='split':b.p['r6']['split']='final'
    if fault=='blocks':b.p['r6']['block_ids'].pop()
    if fault=='selection_missing':del b.p['evaluation_selection']
    if fault!='card':b.card()
    else:b.p['experiment_id']='R6-DIFFERENT'
    with pytest.raises(PermissionError):c.validate_parameters(mode,b.p)


def test_formal_authorization_schema_and_budget(stage,monkeypatch,tmp_path):
    from training_workloads.unified_h1 import authorization as a,config as cm
    b=stage;p=b.p;identity={k:'a'*40 for k in ('product_commit','product_tree','workload_commit','workload_tree')}
    config=dict(schema_version=cm.R6_CANDIDATE_CONFIG_SCHEMA,future_entry={'modes':[c.MODE]},resolved_identity=identity,
        method=dict(version='unified-h1-formal-readiness-v1',proposal_sha256='c'*64))
    monkeypatch.setattr(a,'load_config',lambda:config);monkeypatch.setattr(a,'sha256_file',lambda p:'d'*64)
    seen=[];monkeypatch.setattr(c,'validate_compatibility',lambda *args:seen.append(args))
    attempt=tmp_path/'attempt';attempt.mkdir();path=tmp_path/'authorization.json'
    auth=dict(schema_version='unified-h1-r3-authorization.v2',status='AUTHORIZED',order_id=p['experiment_id']+'-R1-'+c.MODE.upper(),
        method_sha256='c'*64,config_sha256='d'*64,**identity,attempt=str(attempt),mode=c.MODE,budgets=c.required_budget(c.MODE,p),parameters=p)
    path.write_text(json.dumps(auth));assert a.load_r3_authorization(path,attempt,c.MODE)['mode']==c.MODE
    assert seen[0][1]==p['evaluation_selection']
    auth['budgets']['nontraining_transitions']-=1;path.write_text(json.dumps(auth))
    with pytest.raises(PermissionError,match='budget'):a.load_r3_authorization(path,attempt,c.MODE)
    config['schema_version']=cm.R6_CONFIG_SCHEMA
    with pytest.raises(PermissionError,match='dedicated'):a.load_r3_authorization(path,attempt,c.MODE)


def test_candidate_compatibility_needs_selection_specific_review(compatibility,monkeypatch):
    b,r,new=compatibility;value={'fixture':'selection'};model=dict(source_identity=r['origin_source_identity'],checkpoint=r['checkpoint'])
    value['method']=r['method'];monkeypatch.setattr(c.selection,'validate',lambda v:dict(model=model))
    r={k:v for k,v in r.items() if k not in ('independent_review','recipe')};r.update(schema_version='r6-candidate-compatibility/1',selection_sha256=c.base.object_sha(value))
    assert c.validate_compatibility_payload(r,value,new)==model['source_identity']
    r['independent_review']=b.pin('candidate-review.json',dict(decision='PASS_R6_CANDIDATE_COMPATIBILITY',must_fix=[],reviewer='fixture-independent',payload_sha256=c.base.object_sha(r)))
    assert c.validate_compatibility(b.pin('candidate-compat.json',r),value,new)==model['source_identity']
    r['independent_review']=b.pin('wrong-review.json',dict(decision='PASS_R6_EXECUTION_COMPATIBILITY',must_fix=[],reviewer='fixture-independent'))
    with pytest.raises(PermissionError):c.validate_compatibility(b.pin('candidate-compat.json',r),value,new)


def test_candidate_execution_retains_identity_and_entire_original_suite(stage,monkeypatch,tmp_path):
    from training_workloads.unified_h1 import robustness_candidate_restore as restore_module,evaluation
    b=stage;b.p['evaluation_selection'].update(comparison={'fixture':'comparison'},method={'fixture':'method'});b.card()
    model=dict(checkpoint={'fixture':'checkpoint'},source_identity={'fixture':'source'},method_hashes={'fixture':'method'},update=896)
    selected=dict(model=model,method={'ics_clearance_margin_m':.2})
    monkeypatch.setattr(restore_module,'restore',lambda *a:dict(selection=selected,metadata={'fixture':True},method_binding={'fixture':True}))
    rows={};calls=[]
    def evaluate_case(case,**kwargs):
        calls.append((case,kwargs));rows[case]=[{'fixture':case}]*32
        if case=='W':
            path=tmp_path/'evaluation_raw'/kwargs['evaluation_label']/'W.jsonl';path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'kind':'T0'})+'\n'+json.dumps({'root_state_w':[[0]*13 for _ in range(32)]})+'\n')
    seen=[]
    monkeypatch.setattr(evaluation,'evaluate_checkpoint',lambda traces,update:seen.append((len(traces),update)) or {'fixture':'scores'})
    monkeypatch.setattr(evaluation,'wall_motion_prerequisite',lambda values:False)
    backend=SimpleNamespace(authorization=dict(mode=c.MODE,parameters=b.p,budgets=c.required_budget(c.MODE,b.p)),research=b.p,
        output=tmp_path,_evaluation_rows=rows,_evaluate_case=evaluate_case,resource_check=lambda *a:None,evaluate_r6_blocks=lambda:[{'fixture':'blocks'}])
    result=c.evaluate_registered(backend)
    assert [r[0] for r in calls]==list(evaluation.CASES) and seen==[(160,896)]
    assert all(r[1]==dict(direct_b0=False,evaluation_label='r6_candidate_u896_original') for r in calls)
    binding=json.loads((tmp_path/'r6_model_binding.json').read_text());assert binding['checkpoint']==model['checkpoint'] and binding['ics_clearance_margin_m']==.2
    wall=json.loads((tmp_path/'r6_original_wall_motion.json').read_text());assert len(wall['per_lane'])==32 and not wall['all32']
    assert not result['scientific_acceptance'] and result['original_seed']==20260909


def test_dispatch_and_fresh_resource_gate_route(stage,monkeypatch):
    from training_workloads.unified_h1.live_adapter import _dispatch
    from training_workloads.unified_h1.resource_admission import select_gate
    from training_workloads.unified_h1 import robustness_prelaunch
    seen=[];monkeypatch.setattr(c,'evaluate_registered',lambda b:seen.append(b))
    obj=Mock();_dispatch(obj,dict(mode=c.MODE,parameters=stage.p));assert seen==[obj]
    selected=object();monkeypatch.setattr(robustness_prelaunch,'selected',lambda *a:selected)
    assert select_gate({'mode':c.MODE},{},lambda:None) is selected


@pytest.mark.parametrize('fault',['selection','delta','actor','runtime','metric'])
def test_candidate_source_compatibility_rejects_changed_boundaries(compatibility,monkeypatch,fault):
    b,r,new=compatibility;value=dict(method=r['method']);model=dict(source_identity=r['origin_source_identity'],checkpoint=r['checkpoint'])
    monkeypatch.setattr(c.selection,'validate',lambda v:dict(model=model))
    r={k:v for k,v in r.items() if k not in ('independent_review','recipe')};r.update(schema_version='r6-candidate-compatibility/1',selection_sha256=c.base.object_sha(value))
    assert c.validate_compatibility_payload(r,value,new)==model['source_identity']
    if fault=='selection':r['selection_sha256']='f'*64
    elif fault=='delta':r['changed_product_sources']={}
    else:
        config={k:copy.deepcopy(v) for k,v in new.items() if k!='resolved_identity'}
        if fault=='runtime':config['runtime']['changed']=True
        else:
            section='workload' if fault=='metric' else 'product'
            name='training_workloads/unified_h1/evaluation.py' if fault=='metric' else 'isaaclab-training/instinctlab/observations/actor.py'
            path=Path(config[section]['path'])/name;path.write_text('unauthorized new bytes')
            from tools.test_r6_training_contract import pin
            config[section]['source_sha256'][name]=pin(path)['sha256']
        r['execution_config']=b.pin('changed-config.json',config)
        new=dict(config,resolved_identity=dict(new['resolved_identity'],config_sha256=r['execution_config']['sha256']))
        r['execution_source_identity']=new['resolved_identity']
        from training_workloads.unified_h1.robustness_training_contract import source_changes
        old=c.base.pinned_json(r['origin_config'])
        for section in ('product','workload'):r['changed_'+section+'_sources']=source_changes(old[section]['source_sha256'],config[section]['source_sha256'])
    with pytest.raises(PermissionError):c.validate_compatibility_payload(r,value,new)
