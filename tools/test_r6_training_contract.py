"""Exact pinned CPU stage fixtures; kernel volume boundaries are substitutes."""
import copy
import hashlib
from pathlib import Path
import pytest
from tools.test_r6_training_plan import case,update_recipe
from training_workloads.unified_h1 import robustness_training_contract as c
from training_workloads.unified_h1 import robustness_training_plan as plan

ROOT=Path(__file__).resolve().parents[1]


def pin(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def review(case,decision,payload):
    return case.pin(decision+'.json',dict(decision=decision,must_fix=[],reviewer='CPU_FIXTURE_ONLY',payload_sha256=c.object_sha(payload)))


@pytest.fixture
def stage(case,monkeypatch):
    b=case
    from training_workloads.unified_h1.native_volume import MAXIMUM
    from training_workloads.unified_h1.result_volume import CONTROL_ALLOWANCE
    payload=b.plan['storage']['payload_bytes'];cap=payload+96*1024**2+MAXIMUM+CONTROL_ALLOWANCE
    capacity=dict(schema_version='r6-training-capacity/1',plan_sha256=c.object_sha(b.plan),payload_bytes=payload,
        job_output_cap_bytes=cap,budgets=plan.required_budget(b.plan))
    capacity['independent_review']=review(b,'PASS_R6_TRAINING_CAPACITY',capacity)
    p=dict(experiment_id=b.plan['experiment_id'],group='R1',training_seed=11,evaluation_seed=20260909,
        output_cap_bytes=cap,master_authority=pin(ROOT/'docs/experiments/r6-robustness-auto/MASTER_AUTHORITY_V2.json'),
        baseline_bindings=b.ev,checkpoint_compatibility=b.ev,capacity_review=b.pin('capacity.json',capacity),
        r6_training_plan=b.plan,r6_expected_mass_kg=1.181517923,native_volume=b.ev,result_volume=b.ev,extension_inputs=b.ev)
    p['experiment_card']=b.pin('card.json',dict(experiment_id=p['experiment_id'],profiles={'R1':dict(p)}))
    monkeypatch.setattr(c,'baseline',lambda p:dict(selected_model=b.plan['initialization']['checkpoint'],
        source_identity=b.identity,method_hashes={'baseline':'fixture'}))
    monkeypatch.setattr('training_workloads.unified_h1.native_volume.validate',lambda p:None)
    monkeypatch.setattr('training_workloads.unified_h1.extension_inputs.validate',lambda p:None)
    monkeypatch.setattr('training_workloads.unified_h1.result_volume.workload_write_cap',lambda p,**kw:payload)
    return b,p


def test_exact_stage_card_capacity_and_expanded_budget(stage):
    b,p=stage;assert c.validate_parameters(c.MODE,p)==p
    assert c.required_budget(c.MODE,p)==plan.required_budget(b.plan)
    assert c.required_budget(c.MODE,p)['explicit_resets']==2


@pytest.mark.parametrize('fault',['mode','seed','mass','cap','card','capacity','origin','legacy'])
def test_no_baseline_receipt_or_changed_plan_can_authorize_training(stage,fault):
    b,p=stage;mode=c.MODE
    if fault=='mode':mode='research-r6-evaluation'
    elif fault=='seed':p['training_seed']=20260908
    elif fault=='mass':p['r6_expected_mass_kg']=float('nan')
    elif fault=='cap':p['output_cap_bytes']-=4096
    elif fault=='card':p['experiment_card']=b.ev
    elif fault=='capacity':p['r6_training_plan']['wall_seconds']+=1
    elif fault=='origin':p['r6_training_plan']['initialization']['method_hashes']={'other':'fixture'}
    else:
        p['capacity_review']=b.pin('legacy-capacity.json',dict(decision='PASS_R6_PROTOCOL',must_fix=[]))
    with pytest.raises(PermissionError):c.validate_parameters(mode,p)


def test_payload_volume_cannot_be_short_or_wrong_layout(stage,monkeypatch):
    b,p=stage
    monkeypatch.setattr('training_workloads.unified_h1.result_volume.workload_write_cap',lambda p,**kw:1)
    with pytest.raises(PermissionError,match='PAYLOAD_VOLUME'):c.validate_parameters(c.MODE,p)


@pytest.fixture
def compatibility(case,tmp_path):
    b=case
    metric='training_workloads/unified_h1/evaluation.py'
    actor='isaaclab-training/instinctlab/observations/actor.py'
    adapter='isaaclab-training/instinctlab/observations/robustness_physics.py'
    configs=[]
    for name in ('old','new'):
        root=tmp_path/name
        sources={metric:'fixed evaluator',actor:'fixed actor',adapter:name+' adapter'}
        pins={}
        for relative,text in sources.items():
            f=root/relative;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(text);pins[relative]=pin(f)['sha256']
        configs.append(dict(actor={},training={},execution={},runtime={},versions={},
            method=dict(version='fixed',proposal_sha256='fixed',hashes={}),
            product=dict(path=str(root),source_sha256={k:v for k,v in pins.items() if k!=metric}),
            workload=dict(path=str(root),source_sha256={metric:pins[metric]})))
    old,new=configs;oldpin=b.pin('old-config.json',old);newpin=b.pin('new-config.json',new)
    origin=dict(b.identity,config_sha256=oldpin['sha256']);execution=dict(b.identity,config_sha256=newpin['sha256'])
    b.recipe['source_identity']=execution;update_recipe(b);b.plan['initialization']['source_identity']=origin
    record=dict(schema_version='r6-training-compatibility/1',origin_source_identity=origin,execution_source_identity=execution,
        origin_config=oldpin,execution_config=newpin,recipe=b.plan['recipe'],checkpoint=b.plan['initialization']['checkpoint'],
        method=b.recipe['method'],changed_product_sources=c.source_changes(old['product']['source_sha256'],new['product']['source_sha256']),
        changed_workload_sources={})
    record['independent_review']=review(b,'PASS_R6_TRAINING_COMPATIBILITY',record)
    return b,record,dict(new,resolved_identity=execution)


def test_exact_source_delta_and_method_need_their_own_review(compatibility):
    b,r,new=compatibility
    assert c.validate_compatibility(b.pin('compatibility.json',r),b.plan,new)==b.plan['initialization']['source_identity']
    r['independent_review']=b.pin('baseline-review.json',dict(decision='PASS_R6_EXECUTION_COMPATIBILITY',must_fix=[],reviewer='CPU_FIXTURE_ONLY'))
    with pytest.raises(PermissionError,match='independent review'):c.validate_compatibility(b.pin('compatibility.json',r),b.plan,new)


@pytest.mark.parametrize('fault',['recipe','delta','source_bytes','actor','runtime','metric'])
def test_source_drift_invariants_and_wrong_review_payload_rejected(compatibility,fault):
    b,r,new=compatibility
    if fault=='recipe':r['recipe']=b.ev
    elif fault=='delta':r['changed_product_sources']={}
    elif fault=='source_bytes':
        f=Path(new['product']['path'])/'isaaclab-training/instinctlab/observations/robustness_physics.py';f.write_text('drift')
    else:
        if fault=='runtime':new['runtime']={'upgraded':True}
        else:
            section='workload' if fault=='metric' else 'product'
            relative='training_workloads/unified_h1/evaluation.py' if fault=='metric' else 'isaaclab-training/instinctlab/observations/actor.py'
            f=Path(new[section]['path'])/relative;f.write_text('changed protected source')
            new[section]['source_sha256'][relative]=pin(f)['sha256']
        r['execution_config']=b.pin('new-config.json',{k:v for k,v in new.items() if k!='resolved_identity'})
        new['resolved_identity']['config_sha256']=r['execution_config']['sha256']
        r['execution_source_identity']=new['resolved_identity'];b.recipe['source_identity']=new['resolved_identity'];update_recipe(b);r['recipe']=b.plan['recipe']
        for section in ('product','workload'):
            old=c.pinned_json(r['origin_config']);r['changed_'+section+'_sources']=c.source_changes(old[section]['source_sha256'],new[section]['source_sha256'])
        r['independent_review']=review(b,'PASS_R6_TRAINING_COMPATIBILITY',{k:v for k,v in r.items() if k!='independent_review'})
    with pytest.raises(PermissionError):c.validate_compatibility(b.pin('compatibility.json',r),b.plan,new)


def test_training_authorization_requires_dedicated_schema_and_exact_binding(stage,monkeypatch,tmp_path):
    import json
    from training_workloads.unified_h1 import authorization as a,config as config_module
    b,p=stage;identity=b.recipe['source_identity'];config=dict(schema_version=config_module.R6_TRAINING_CONFIG_SCHEMA,
        future_entry={'modes':[c.MODE]},resolved_identity=identity,
        method=dict(version='unified-h1-formal-readiness-v1',proposal_sha256='c'*64))
    monkeypatch.setattr(a,'load_config',lambda:config)
    monkeypatch.setattr(a,'sha256_file',lambda path:'d'*64)
    checked=[];monkeypatch.setattr(c,'validate_compatibility',lambda *args:checked.append(args))
    attempt=tmp_path/'attempt';attempt.mkdir();path=tmp_path/'authorization.json'
    auth=dict(schema_version='unified-h1-r3-authorization.v2',status='AUTHORIZED',
        order_id=p['experiment_id']+'-R1-'+c.MODE.upper(),method_sha256='c'*64,config_sha256='d'*64,
        **{k:identity[k] for k in ('product_commit','product_tree','workload_commit','workload_tree')},
        attempt=str(attempt),mode=c.MODE,budgets=c.required_budget(c.MODE,p),parameters=p)
    path.write_text(json.dumps(auth));assert a.load_r3_authorization(path,attempt,c.MODE)['mode']==c.MODE
    assert checked and checked[0][1]==p['r6_training_plan']
    auth['budgets']['training_transitions']-=1;path.write_text(json.dumps(auth))
    with pytest.raises(PermissionError,match='budget'):a.load_r3_authorization(path,attempt,c.MODE)
    config['schema_version']=config_module.R6_CONFIG_SCHEMA
    with pytest.raises(PermissionError,match='dedicated'):a.load_r3_authorization(path,attempt,c.MODE)


def test_formal_dispatch_and_training_resource_gate_are_not_baseline(stage,monkeypatch):
    from unittest.mock import Mock
    from training_workloads.unified_h1.live_adapter import _dispatch
    from training_workloads.unified_h1.resource_admission import select_gate
    from training_workloads.unified_h1 import robustness_prelaunch
    b,p=stage;backend=Mock();_dispatch(backend,{'mode':c.MODE,'parameters':p})
    backend.train_r6.assert_called_once_with()
    selected=select_gate({'mode':c.MODE,'parameters':p},{'fixture':True},object())
    assert selected.func is robustness_prelaunch.run_gate and selected.keywords['stop_paths']


def test_control_native_options_preserve_dual_volume_for_training(stage,monkeypatch):
    from rlflow import h1
    b,p=stage;spec=dict(backend='real',h1=dict(mode=c.MODE,attempt='/CPU/attempt'))
    monkeypatch.setattr(h1,'inspect_authority',lambda spec:dict(authorization={'parameters':p}))
    options=h1.native_dispatch_options(spec)
    assert options['native_volume_pin']==p['native_volume']
    assert options['result_volume_pin']==p['result_volume']
    assert options['job_output_cap_bytes']==p['output_cap_bytes']


def test_draft_requires_reviews_without_touching_unprovisioned_volumes(stage,monkeypatch):
    b,p=stage
    def forbidden(*a,**k):raise AssertionError('draft must not inspect absent volumes')
    monkeypatch.setattr('training_workloads.unified_h1.native_volume.validate',forbidden)
    monkeypatch.setattr('training_workloads.unified_h1.result_volume.workload_write_cap',forbidden)
    draft={k:v for k,v in p.items() if k not in c.MANAGED}
    assert c.validate_draft(c.MODE,draft)==draft
    draft['r6_training_plan']['wall_seconds']+=1
    with pytest.raises(PermissionError,match='CAPACITY'):c.validate_draft(c.MODE,draft)


def test_training_profile_frozen_program_uses_actual_draft_contract(stage,monkeypatch):
    import io,json,sys
    from tools.r6_training_profile import PROGRAM
    from training_workloads.unified_h1 import config
    b,p=stage;draft={k:v for k,v in p.items() if k not in c.MANAGED}
    cfg=dict(schema_version=config.R6_TRAINING_CONFIG_SCHEMA,resolved_identity=b.recipe['source_identity'])
    monkeypatch.setattr(config,'load_config',lambda:cfg)
    seen=[];monkeypatch.setattr(c,'validate_compatibility',lambda *a:seen.append(a))
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(draft)));out=io.StringIO()
    monkeypatch.setattr(sys,'stdout',out);monkeypatch.setattr(sys,'argv',['fixture',str(ROOT)])
    original=list(sys.path)
    try:exec(PROGRAM,{})
    finally:sys.path[:]=original
    result=json.loads(out.getvalue());assert result['source_identity']==b.recipe['source_identity']
    assert result['payload_bytes']==b.plan['storage']['payload_bytes']
    assert seen==[(draft['checkpoint_compatibility'],draft['r6_training_plan'],cfg)]
