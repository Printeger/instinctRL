"""Exact pinned selection records; explicit baseline identity fixture, no tensors/live."""
import copy
import json
import pytest
from tools.test_r6_training_plan import case
from training_workloads.unified_h1 import robustness_evaluation_selection as s


@pytest.fixture
def selection(case,monkeypatch):
    b=case;attempt=s.validate_pin(b.ev).parent/'attempt';(attempt/'results/research-r6-training/checkpoints').mkdir(parents=True)
    b.checkpoint=b.pin('attempt/results/research-r6-training/checkpoints/u896.pt',{'synthetic_checkpoint':True})
    original=s.validate_pin
    monkeypatch.setattr(s,'validate_pin',lambda pin:s.training.validate_pin(pin) if pin==b.plan['initialization']['checkpoint'] else original(pin))
    baseline=dict(selected_model=b.ev,source_identity=b.identity,method_hashes={'baseline':'fixture'})
    monkeypatch.setattr(s.base,'baseline',lambda value:baseline)
    protocol=b.pin('protocol.json',dict(frozen=True,blocks=[dict(id=str(i)) for i in range(16)]))
    b.comparison=dict(schema_version='r6-paired-comparison/1',id='CPU-PAIR',candidate_id=b.recipe['candidate_id'],hypothesis='compare policy with same controller',
        protocol=protocol,split='development',seed=17,block_ids=[str(i) for i in range(16)],method=b.recipe['method'],baseline_checkpoint_sha256=s.base.BASELINE_SHA)
    auth=b.pin('auth.json',dict(mode='research-r6-training',attempt=str(attempt),parameters={'r6_training_plan':b.plan},budgets=s.training.required_budget(b.plan)))
    spec_value=dict(pins={'authorization':auth},h1=dict(mode='research-r6-training',attempt=str(attempt)))
    spec=b.pin('spec.json',spec_value);closure=b.pin('closure.json',dict(job_id='CPU',spec_sha256=s.base.object_sha(spec_value),native_status='FAILED',native_exit_code=-9))
    inventory=b.pin('inventory.json',dict(job_id='CPU',spec=spec,closure=closure,output_inventory_complete=True,
        files=[dict(pin,bytes=s.validate_pin(pin).stat().st_size) for pin in (b.checkpoint,auth)]))
    postrun=b.pin('postrun.json',dict(job_id='CPU',closure=closure,inventory=inventory,postrun_complete=True))
    b.report=dict(schema_version='r6-training-analysis/1',authorization=auth,plan_sha256=s.base.object_sha(b.plan),scientific_acceptance=False,issues=[],
        inputs=[dict(pin,bytes=s.validate_pin(pin).stat().st_size) for pin in (b.checkpoint,auth,b.plan['initialization']['checkpoint'])],
        summary=dict(issue_count=0,probability_gate_recomputed=True,rng_initialization_audit={'status':'PASS'},checkpoints=[dict(pin=b.checkpoint,update=896,metadata_verified=True,
            state_audit=dict(actor_critic_state_loaded=True,adam_state_loaded=True))]))
    b.binding=dict(schema_version='r6-bound-training-analysis/1',scientific_acceptance=False,training_plan_sha256=s.base.object_sha(b.plan),
        spec=spec,authorization=auth,closure=closure,inventory=inventory,postrun=postrun,job_id='CPU',native_status='FAILED',native_exit_code=-9)
    b.selection=dict(schema_version='r6-evaluation-selection/1',phase='development',role='trained_candidate',baseline_bindings=b.ev,
        checkpoint=b.checkpoint,training_plan=b.plan,method=b.recipe['method'])
    def save():
        b.binding['report']=b.pin('report.json',b.report)
        b.selection.update(training_analysis=b.pin('binding.json',b.binding),comparison=b.pin('comparison.json',b.comparison))
        body={k:v for k,v in b.selection.items() if k!='independent_review'}
        b.selection['independent_review']=b.pin('selection-review.json',dict(decision='PASS_R6_EVALUATION_SELECTION',must_fix=[],reviewer='fixture-independent',payload_sha256=s.base.object_sha(body)))
    b.save=save;save();return b


def test_candidate_selection_is_bound_but_not_dispatch_or_learning_acceptance(selection):
    result=s.validate(selection.selection)
    assert result['model']['checkpoint']==selection.checkpoint and result['model']['update']==896
    assert result['model']['ppo_config']['permutation_seed']==11
    assert not result['dispatch_authorized'] and not result['scientific_acceptance']


def test_control_keeps_u640_and_shares_comparison_method_and_stream(selection):
    b=selection;candidate=s.validate(b.selection);control=copy.deepcopy(b.selection)
    control.update(role='baseline_control',checkpoint=b.ev,training_plan=None,training_analysis=None)
    control['independent_review']=b.pin('control-review.json',dict(decision='PASS_R6_EVALUATION_SELECTION',must_fix=[],reviewer='fixture-independent',
        payload_sha256=s.base.object_sha({k:v for k,v in control.items() if k!='independent_review'})))
    result=s.validate(control)
    assert result['model']['update']==640 and result['comparison']==candidate['comparison'] and result['method']==candidate['method']


@pytest.mark.parametrize('fault',['final','method','checkpoint','node','issues','gate','rng','adam','inventory','native','stream','blocks','review','missing_input','changed_input'])
def test_selection_rejects_mismatches_even_with_fresh_fixture_review(selection,fault):
    b=selection
    if fault=='missing_input':b.report['inputs'].pop()
    if fault=='changed_input':b.report['inputs'][0]['sha256']='f'*64
    if fault=='final':b.selection['phase']='final'
    if fault=='method':b.selection['method']=b.pin('other-method.json',s.base.pinned_json(b.recipe['method']))
    if fault=='checkpoint':b.selection['checkpoint']=b.ev
    if fault=='node':b.report['summary']['checkpoints'][0]['update']=897
    if fault=='issues':b.report['issues']=[{'reason':'corrupt'}]
    if fault=='gate':b.report['summary']['probability_gate_recomputed']=False
    if fault=='rng':b.report['summary']['rng_initialization_audit']['status']='FAILED'
    if fault=='adam':b.report['summary']['checkpoints'][0]['state_audit']['adam_state_loaded']=False
    if fault=='inventory':
        v=s.base.pinned_json(b.binding['inventory']);v['files']=[];b.binding['inventory']=b.pin('bad-inventory.json',v)
    if fault=='native':b.binding['native_exit_code']=0
    if fault=='stream':b.comparison['split']='final'
    if fault=='blocks':b.comparison['block_ids'].pop()
    b.save()
    if fault=='review':b.selection['independent_review']=b.pin('rejected.json',dict(decision='MUST_FIX',must_fix=['fixture'],reviewer='fixture-independent'))
    with pytest.raises((PermissionError,KeyError)):s.validate(b.selection)


def test_loaded_metadata_cannot_substitute_seed_or_clock(selection):
    b=selection;r=s.validate(b.selection);model=r['model'];u=model['update']
    metadata=dict(counters=dict(update=u,transition=u*512,global_policy_step=u*16,actor_optimizer=u*8,critic_optimizer=u*8),training_seed=11,
        curriculum=dict(version='curriculum-h1-r6-v1',completed_policy_steps=u*16,recipe_sha256=b.plan['recipe']['sha256'],origin_checkpoint_sha256=s.base.BASELINE_SHA,branch_seed=11))
    assert s.verify_loaded(r,metadata)==model
    metadata['training_seed']=12
    with pytest.raises(PermissionError):s.verify_loaded(r,metadata)


def test_selection_draft_retains_evidence_checks_but_is_not_accepted(selection):
    b=selection;draft={k:v for k,v in b.selection.items() if k!='independent_review'}
    result=s.validate_draft(draft)
    assert result['model']==s.validate(b.selection)['model'] and not result['dispatch_authorized']
    with pytest.raises(PermissionError):s.validate(draft)
    with pytest.raises(PermissionError):s.validate_draft(b.selection)
    draft['checkpoint']=b.ev
    with pytest.raises(PermissionError):s.validate_draft(draft)


def test_frozen_selection_review_program_rechecks_unreviewed_evidence(selection,monkeypatch):
    import io,sys
    from pathlib import Path
    from tools import r6_candidate_review_inputs as tool
    from training_workloads.unified_h1 import config
    b=selection;draft={k:v for k,v in b.selection.items() if k!='independent_review'}
    draft_pin=b.pin('draft-selection.json',draft);root=Path(draft_pin['path']).parent/'snapshot'
    cp=root/'training_workloads/unified_h1/candidate_config.json';cp.parent.mkdir(parents=True);cp.write_text('{}')
    cfg=dict(schema_version=config.R6_CANDIDATE_CONFIG_SCHEMA,resolved_identity=b.identity)
    monkeypatch.setattr(config,'load_config',lambda:cfg)
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(dict(kind='selection',selection=draft_pin))))
    out=io.StringIO();monkeypatch.setattr(sys,'stdout',out);monkeypatch.setattr(sys,'argv',['fixture',str(root)])
    before=list(sys.path)
    try:exec(tool.PROGRAM,{})
    finally:sys.path[:]=before
    result=json.loads(out.getvalue());assert result['payload']==draft and result['source_identity']==b.identity
    with pytest.raises(PermissionError):s.validate(result['payload'])
