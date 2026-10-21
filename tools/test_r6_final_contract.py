"""CPU stage tests with explicit selection, source and simulator substitutes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools import test_r6_contract as baseline_fixture
from tools.test_r6_training_plan import case
from tools.test_r6_training_contract import compatibility
from training_workloads.unified_h1 import robustness_final_contract as c

@pytest.fixture
def stage(monkeypatch,request):
    b=baseline_fixture.ContractTests();b.setUp();request.addfinalizer(b.tearDown)
    b.p['r6']['split']='final';b.p['r6']['seed']=179123
    b.p['final_selection']=dict(schema_version='r6-final-selection/1',batch={'fixture':'released'},assignment_index=0)
    b.selected=dict(selection=dict(baseline_bindings=b.p['baseline_bindings']),
        comparison={k:b.p['r6'][k] for k in ('protocol','block_ids','seed','split')},
        model=dict(training_seed=20260908),method_pin=None,method=None,assignment={'fixture':'baseline'})
    monkeypatch.setattr(c.selection,'validate',lambda value:b.selected)
    b.card();return b


def test_budget_and_mode_isolation(stage):
    from training_workloads.unified_h1 import research_contract,robustness_candidate_contract
    p=stage.p;assert research_contract.validate_parameters(c.MODE,p)==p
    budget=research_contract.required_budget(c.MODE,p)
    assert budget['nontraining_transitions']==193600 and budget['explicit_resets']==21
    assert budget['training_transitions']==budget['optimizer_updates']==budget['checkpoint_writes']==0
    with pytest.raises(PermissionError):c.base.validate_parameters(c.base.MODE,p)
    with pytest.raises(PermissionError):robustness_candidate_contract.validate_parameters(robustness_candidate_contract.MODE,p)


@pytest.mark.parametrize('fault',['mode','split','seed','training_seed','original_seed','blocks','mixed','missing','card'])
def test_reject_drift(stage,fault):
    b=stage;mode=c.MODE
    if fault=='mode':mode=c.base.MODE
    elif fault=='split':b.p['r6']['split']='development'
    elif fault=='seed':b.p['r6']['seed']+=1
    elif fault=='training_seed':b.p['training_seed']+=1
    elif fault=='original_seed':b.p['evaluation_seed']+=1
    elif fault=='blocks':b.p['r6']['block_ids']=b.p['r6']['block_ids'][:-1]
    elif fault=='mixed':b.p['evaluation_selection']={}
    elif fault=='missing':del b.p['final_selection']
    elif fault=='card':b.p['experiment_id']='R6-DRIFT'
    if fault!='card':b.card()
    with pytest.raises(PermissionError):c.validate_parameters(mode,b.p)


def test_dedicated_authorization(stage,monkeypatch,tmp_path):
    from training_workloads.unified_h1 import authorization as a,config as cm
    p=stage.p;identity={k:'a'*40 for k in ('product_commit','product_tree','workload_commit','workload_tree')}
    config=dict(schema_version=cm.R6_FINAL_CONFIG_SCHEMA,future_entry={'modes':[c.MODE]},resolved_identity=identity,
        method=dict(version='unified-h1-formal-readiness-v1',proposal_sha256='c'*64))
    monkeypatch.setattr(a,'load_config',lambda:config);monkeypatch.setattr(a,'sha256_file',lambda p:'d'*64)
    seen=[];monkeypatch.setattr(c,'validate_compatibility',lambda *args:seen.append(args))
    attempt=tmp_path/'attempt';attempt.mkdir();path=tmp_path/'authorization.json'
    auth=dict(schema_version='unified-h1-r3-authorization.v2',status='AUTHORIZED',order_id=p['experiment_id']+'-R1-'+c.MODE.upper(),
        method_sha256='c'*64,config_sha256='d'*64,**identity,attempt=str(attempt),mode=c.MODE,budgets=c.required_budget(c.MODE,p),parameters=p)
    path.write_text(json.dumps(auth));assert a.load_r3_authorization(path,attempt,c.MODE)['mode']==c.MODE
    assert seen[0][1]==p['final_selection']
    auth['budgets']['optimizer_updates']=1;path.write_text(json.dumps(auth))
    with pytest.raises(PermissionError,match='budget'):a.load_r3_authorization(path,attempt,c.MODE)
    config['schema_version']=cm.R6_CANDIDATE_CONFIG_SCHEMA
    with pytest.raises(PermissionError,match='dedicated'):a.load_r3_authorization(path,attempt,c.MODE)


@pytest.mark.parametrize('method',[None,'confirmed'])
def test_exact_source_compatibility(compatibility,monkeypatch,method):
    b,r,new=compatibility;value={'fixture':'final selection'}
    selected=dict(model=dict(source_identity=r['origin_source_identity'],checkpoint=r['checkpoint']),method_pin=r['method'] if method else None)
    monkeypatch.setattr(c.selection,'validate',lambda v:selected)
    payload=c.compatibility_payload(r['origin_config'],r['execution_config'],value,new)
    assert payload['method']==selected['method_pin']
    assert c.validate_compatibility_payload(payload,value,new)==selected['model']['source_identity']
    record=dict(payload,independent_review=b.pin('final-compat-review.json',dict(decision='PASS_R6_FINAL_COMPATIBILITY',
        must_fix=[],reviewer='fixture-independent',payload_sha256=c.base.object_sha(payload))))
    assert c.validate_compatibility(b.pin('final-compat.json',record),value,new)==selected['model']['source_identity']
    for field,bad in [('method',{'wrong':True}),('selection_sha256','f'*64),('changed_product_sources',{}),('schema_version','r6-candidate-compatibility/1')]:
        drift=dict(payload);drift[field]=bad
        with pytest.raises(PermissionError):c.validate_compatibility_payload(drift,value,new)


def test_draft_capacity(stage,monkeypatch):
    from training_workloads.unified_h1 import extension_inputs
    monkeypatch.setattr(extension_inputs,'validate',lambda v:None)
    p={k:v for k,v in stage.p.items() if k not in ('experiment_card','output_cap_bytes')};p['extension_inputs']={}
    payload=64*1024**2+sum(sum(v.values())+16*1024**2 for v in p['r6']['output_quotas'].values())
    assert c.validate_draft(c.MODE,p,payload)==p
    with pytest.raises(PermissionError):c.validate_draft(c.MODE,p,payload-4096)


@pytest.mark.parametrize('trained',[False,True])
def test_complete_wrapper(stage,monkeypatch,tmp_path,trained):
    from training_workloads.unified_h1 import robustness_final_restore as restore_module,evaluation
    b=stage;update=896 if trained else 640;model=dict(checkpoint={'fixture':'checkpoint'},source_identity={'fixture':'source'},method_hashes={'fixture':'method'},update=update)
    selected=dict(b.selected,model=model,method={'ics_clearance_margin_m':.2} if trained else None)
    monkeypatch.setattr(restore_module,'restore',lambda *a:dict(selection=selected,metadata={'fixture':True},method_binding={'fixture':True}))
    rows={};calls=[]
    def evaluate_case(case,**kwargs):
        calls.append((case,kwargs));rows[case]=[{'fixture':case}]*32
        if case=='W':
            path=tmp_path/'evaluation_raw'/kwargs['evaluation_label']/'W.jsonl';path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'kind':'T0'})+'\n'+json.dumps({'root_state_w':[[0]*13 for _ in range(32)]})+'\n')
    seen=[];monkeypatch.setattr(evaluation,'evaluate_checkpoint',lambda traces,update:seen.append((len(traces),update)) or {'fixture':'scores'})
    monkeypatch.setattr(evaluation,'wall_motion_prerequisite',lambda values:False)
    backend=SimpleNamespace(authorization=dict(mode=c.MODE,parameters=b.p,budgets=c.required_budget(c.MODE,b.p)),research=b.p,
        output=tmp_path,_evaluation_rows=rows,_evaluate_case=evaluate_case,resource_check=lambda *a:None,evaluate_r6_blocks=lambda:[{'fixture':'all blocks'}])
    result=c.evaluate_registered(backend)
    assert [r[0] for r in calls]==list(evaluation.CASES) and seen==[(160,update)]
    assert all(r[1]==dict(direct_b0=False,evaluation_label=f'r6_final_u{update}_original') for r in calls)
    binding=json.loads((tmp_path/'r6_model_binding.json').read_text());assert binding['checkpoint']==model['checkpoint']
    assert binding['ics_clearance_margin_m']==(.2 if trained else .15)
    assert len(json.loads((tmp_path/'r6_original_wall_motion.json').read_text())['per_lane'])==32
    assert result['new_seed']==179123 and result['original_seed']==20260909 and not result['scientific_acceptance']


def test_dispatch_resource_and_native_routes(stage,monkeypatch):
    from training_workloads.unified_h1.live_adapter import _dispatch
    from training_workloads.unified_h1.resource_admission import select_gate
    from training_workloads.unified_h1 import robustness_prelaunch
    seen=[];monkeypatch.setattr(c,'evaluate_registered',lambda b:seen.append(b))
    obj=Mock();_dispatch(obj,dict(mode=c.MODE,parameters=stage.p));assert seen==[obj]
    marker=object();monkeypatch.setattr(robustness_prelaunch,'selected',lambda *a:marker)
    assert select_gate({'mode':c.MODE},{},lambda:None) is marker


def test_final_native_dispatch_exact_pins(monkeypatch,tmp_path):
    from rlflow import h1
    params=dict(native_volume={'fixture':'native'},result_volume={'fixture':'result'},output_cap_bytes=123)
    seen=[]
    monkeypatch.setattr(h1,'inspect_authority',lambda spec:seen.append(spec) or dict(authorization=dict(parameters=params)))
    spec=dict(backend='real',h1=dict(mode=c.MODE,attempt=str(tmp_path)))
    options=h1.native_dispatch_options(spec)
    assert seen==[spec] and options['native_volume_pin']==params['native_volume']
    assert options['result_volume_pin']==params['result_volume'] and options['job_output_cap_bytes']==123
    assert options['native_workload_output_root']==str(tmp_path/'results')
