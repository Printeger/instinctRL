"""Synthetic complete raw index and paired selections; no replay/physics or live acceptance."""
import copy
import json
from pathlib import Path
import pytest
from tools.test_r6_training_plan import case
from training_workloads.unified_h1 import robustness_confirmation as c
from training_workloads.unified_h1 import robustness_evaluation_selection as selection
from training_workloads.unified_h1 import robustness_training_plan as training


@pytest.fixture
def ready(case,tmp_path,monkeypatch):
    b=case
    def pin(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        return b.pin(name,value)
    protocol=pin('protocol.json',dict(frozen=True,blocks=[dict(id=str(i),domain='target' if i<12 else 'stress') for i in range(16)]))
    split=pin('split.json',dict(schema_version='r6-split-manifest/1',confirmation_training_seeds=b.recipe['confirmation_seeds'],final={'seed_realization_generated':False}))
    review=pin('protocol-review.json',dict(decision='PASS_R6_PROTOCOL',must_fix=[],reviewer='fixture-independent',
        protocol_sha256=protocol['sha256'],evidence_pins=[protocol,split]))
    comparison=pin('comparison.json',{'fixture':'same paired method and stream'})
    selected={};pairs={};reports={};bindings={}
    for role,u in [('baseline_control',640),('trained_candidate',896)]:
        value=dict(role=role,method=b.recipe['method'],comparison=comparison,baseline_bindings=b.ev,training_plan=b.plan if role=='trained_candidate' else None)
        model=dict(checkpoint=b.ev,update=u);selected[role]=dict(selection=value,model=model,comparison=dict(protocol=protocol,seed=17))
        output=tmp_path/role/'results/research-r6-candidate-evaluation';output.mkdir(parents=True)
        request=dict(protocol=protocol,split='development',seed=17,run_id=role,block_ids=[str(i) for i in range(16)])
        auth=pin(role+'/auth.json',dict(mode='research-r6-candidate-evaluation',attempt=str(output.parent.parent),parameters=dict(r6=request,evaluation_selection=value)))
        spec_value=dict(pins={'authorization':auth},h1=dict(mode='research-r6-candidate-evaluation',attempt=str(output.parent.parent)))
        spec=pin(role+'/spec.json',spec_value)
        label=f'r6_candidate_u{u}_original';model_binding=pin(str(output/'r6_model_binding.json'),dict(checkpoint=b.ev,method=b.recipe['method'],
            evaluation_selection_sha256=c.base.object_sha(value),original_evaluation_label=label))
        files=[auth,model_binding]
        for i in range(16):
            for name in ('motion_traces.json','result.json','measurements.bin','audit.bin'):files.append(pin(str(output/'r6'/str(i)/name),{}))
        for name in ('Z','T','S','X','W'):files.append(pin(str(output/'evaluation_raw'/label/(name+'.jsonl')),{}))
        for name in ('r6_original_suite.json','r6_original_wall_motion.json'):files.append(pin(str(output/name),{}))
        inputs=[dict(p,bytes=Path(p['path']).stat().st_size) for p in files]
        close=pin(role+'/close.json',dict(event_key=role,job_id=role,spec_sha256=c.base.object_sha(spec_value),native_status='FAILED',native_exit_code=-9))
        inventory=pin(role+'/inventory.json',dict(event_key=role,job_id=role,spec=spec,closure=close,output_inventory_complete=True,files=inputs))
        post=pin(role+'/post.json',dict(job_event_key=role,job_id=role,closure=close,inventory=inventory,postrun_complete=True,postrun_queries_used=2,receipts=[pin(role+f'/postrun{i}.json',{}) for i in range(2)]))
        report=dict(schema_version='r6-motion-replay/1',scientific_acceptance=False,authorization=auth,candidate_binding_issues=[],run_id=role,seed=17,split='development',
            candidate=dict(selection_sha256=c.base.object_sha(value),checkpoint=b.ev,update=u,original_label=label),
            blocks=[dict(block_id=str(i),domain='target' if i<12 else 'stress',total=32,replay_complete=True,saved_result_equal=True,
                measurement_replay={'bit_exact_delivery':True},physics_replay={'staging_reconstructed':True},scoring={'successes':0}) for i in range(16)],
            summary=dict(original=dict(replay_complete=True,saved_result_equal=True,saved_wall_equal=True,all_gates_pass=True,wall_motion=dict(all32=True,per_lane=[True]*32))),
            inputs=inputs+[dict(protocol,bytes=Path(protocol['path']).stat().st_size)])
        report_pin=pin(role+'/report.json',report);reports[role]=report
        bound=dict(schema_version='r6-bound-motion-analysis/1',scientific_acceptance=False,issues=[],protocol=protocol,spec=spec,authorization=auth,
            closure=close,inventory=inventory,postrun=post,report=report_pin,model_binding=model_binding,event_key=role,job_id=role,native_status='FAILED',native_exit_code=-9)
        pairs[role]=pin(role+'/bound.json',bound);bindings[role]=bound
    monkeypatch.setattr(selection,'validate',lambda value:copy.deepcopy(selected[value['role']]))
    evidence=dict(schema_version='r6-recipe-selection-evidence/1',protocol=protocol,split_manifest=split,protocol_review=review,paired_evaluations=pairs,
        checkpoint_selection=dict(kind='terminal_checkpoint',common_start_update=640,target_update=640+b.recipe['branch_updates'],select_using_holdout=False))
    b.evidence=evidence;b.reports=reports;b.bindings=bindings;b.selected=selected;b.write=pin
    b.check=lambda:c.validate_selection_evidence(pin('evidence.json',evidence),b.plan['recipe'],b.recipe)
    return b


def test_complete_paired_evidence_can_be_reviewed_even_when_new_target_is_negative(ready):
    result=ready.check()
    assert not result['scientific_acceptance'] and not result['holdout_released']
    assert all(r['scoring']['successes']==0 for r in ready.reports['trained_candidate']['blocks'])


@pytest.mark.parametrize('fault',['missing_role','seed','holdout','rule','method','pair','recipe','original','wall','block','measurement','inputs','native','raw','event','postrun'])
def test_freeze_rejects_incomplete_unpaired_or_contaminated_evidence(ready,fault):
    b=ready;b.check();e=b.evidence;r=b.reports['trained_candidate'];bound=b.bindings['trained_candidate']
    if fault=='missing_role':e['paired_evaluations'].pop('baseline_control')
    if fault=='seed':b.recipe['confirmation_seeds']=[21,21,23]
    if fault=='holdout':e['split_manifest']=b.write('revealed.json',dict(schema_version='r6-split-manifest/1',confirmation_training_seeds=[21,22,23],final={'seed_realization_generated':True}))
    if fault=='rule':e['checkpoint_selection']['select_using_holdout']=True
    if fault=='method':b.selected['trained_candidate']['selection']['method']=b.ev
    if fault=='pair':b.selected['trained_candidate']['selection']['comparison']=b.ev
    if fault=='recipe':b.selected['trained_candidate']['selection']['training_plan']=dict(b.plan,recipe=b.ev)
    if fault=='original':r['summary']['original']['all_gates_pass']=False
    if fault=='wall':r['summary']['original']['wall_motion']['per_lane'][31]=False
    if fault=='block':r['blocks'].pop()
    if fault=='measurement':r['blocks'][0]['measurement_replay']['bit_exact_delivery']=False
    if fault=='inputs':r['inputs'].pop(2)
    if fault=='native':bound['native_exit_code']=0
    if fault=='event':bound['event_key']='different'
    if fault=='postrun':
        post=json.loads(Path(bound['postrun']['path']).read_text());post['receipts']=[post['receipts'][0]]*2
        bound['postrun']=b.write('bad-postrun.json',post)
    if fault=='raw':Path(r['inputs'][2]['path']).write_text('changed')
    bound['report']=b.write('changed-report.json',r)
    e['paired_evaluations']['trained_candidate']=b.write('changed-bound.json',bound)
    with pytest.raises((PermissionError,ValueError)):b.check()


def test_confirmation_plan_requires_full_selection_evidence_and_freeze_review(ready):
    b=ready;ep=b.write('evidence.json',b.evidence)
    freeze=dict(schema_version='r6-recipe-freeze/1',recipe=b.plan['recipe'],confirmation_seeds=b.recipe['confirmation_seeds'],selection_evidence=ep)
    review=b.write('freeze-review.json',dict(decision='PASS_R6_RECIPE_FREEZE',must_fix=[],reviewer='fixture-independent',payload_sha256=c.base.object_sha(freeze)))
    plan=dict(b.plan,phase='confirmation',training_seed=21,selection_freeze=b.write('freeze.json',dict(freeze,independent_review=review)))
    assert training.validate(plan)==plan
    plan['training_seed']=11
    with pytest.raises(PermissionError):training.validate(plan)
    plan['training_seed']=21
    bad=b.write('bad-review.json',dict(decision='MUST_FIX',must_fix=['fixture'],reviewer='fixture-independent'))
    plan['selection_freeze']=b.write('rejected.json',dict(freeze,independent_review=bad))
    with pytest.raises(PermissionError):training.validate(plan)
