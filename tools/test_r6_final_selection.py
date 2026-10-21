"""Pinned complete checkpoint chains; baseline and plan validation boundaries explicit."""
import copy
import pytest
from tools.test_r6_evaluation_selection import selection
from tools.test_r6_training_plan import case
from training_workloads.unified_h1 import robustness_final_selection as f
from training_workloads.unified_h1 import robustness_evaluation_selection as development
from training_workloads.unified_h1.robustness_holdout import commitment,reveal


@pytest.fixture
def ready(selection,monkeypatch):
    b=selection;recipe=copy.deepcopy(b.recipe);recipe['branch_updates']=256
    recipe_pin=b.pin('final-recipe.json',recipe);recipe_freeze=b.ev
    monkeypatch.setattr(f.training,'validate',lambda plan:copy.deepcopy(plan))
    models={'baseline':b.ev};streams=[];b.plans=[];b.reports=[]
    for i,seed in enumerate(recipe['confirmation_seeds']):
        name='final-'+str(i);plan=dict(b.plan,phase='confirmation',training_seed=seed,recipe=recipe_pin,selection_freeze=recipe_freeze)
        (f.validate_pin(b.ev).parent/name/'results/research-r6-training/checkpoints').mkdir(parents=True)
        b.plans.append(plan);checkpoint=b.pin(name+'/results/research-r6-training/checkpoints/u896.pt',{'fixture':seed})
        attempt=str(f.validate_pin(checkpoint).parents[3])
        auth=b.pin(name+'/auth.json',dict(mode='research-r6-training',attempt=attempt,parameters={'r6_training_plan':plan},budgets=f.training.required_budget(plan)))
        spec_value=dict(pins={'authorization':auth},h1=dict(mode='research-r6-training',attempt=attempt))
        spec=b.pin(name+'/spec.json',spec_value);closure=b.pin(name+'/closure.json',dict(job_id=name,spec_sha256=f.base.object_sha(spec_value),native_status='FAILED',native_exit_code=-9))
        inventory=b.pin(name+'/inventory.json',dict(job_id=name,spec=spec,closure=closure,output_inventory_complete=True,
            files=[dict(pin,bytes=development.validate_pin(pin).stat().st_size) for pin in (checkpoint,auth)]))
        post=b.pin(name+'/post.json',dict(job_id=name,closure=closure,inventory=inventory,postrun_complete=True))
        report=copy.deepcopy(b.report);report.update(authorization=auth,plan_sha256=f.base.object_sha(plan),
            inputs=[dict(pin,bytes=development.validate_pin(pin).stat().st_size) for pin in (checkpoint,auth,plan['initialization']['checkpoint'])])
        report['summary']['checkpoints'][0]['pin']=checkpoint;b.reports.append(report)
        analysis=b.pin(name+'/analysis.json',dict(schema_version='r6-bound-training-analysis/1',scientific_acceptance=False,
            training_plan_sha256=f.base.object_sha(plan),spec=spec,authorization=auth,closure=closure,inventory=inventory,postrun=post,
            report=b.pin(name+'/report.json',report),job_id=name,native_status='FAILED',native_exit_code=-9))
        context=b.pin(name+'/context.json',dict(baseline_profile=b.pin(name+'/profile.json',dict(r6_training_plan=plan,baseline_bindings=b.ev))))
        streams.append(dict(context=context,analysis=analysis,checkpoint=checkpoint,plan_sha256=f.base.object_sha(plan),training_seed=seed,target_update=896))
        models['seed'+str(i+1)]=checkpoint
    b.freeze=dict(schema_version='r6-final-freeze/2',branch='THREE_CONTINUATIONS',protocol=b.comparison['protocol'],recipe=recipe_pin,
        selection_rule=dict(kind='terminal_checkpoint',common_start_update=640,target_update=896,select_using_holdout=False),models=models,
        evidence=dict(baseline_bindings=b.ev,recipe_freeze=recipe_freeze,streams=streams))
    def save():
        freeze_pin=b.pin('final-freeze.json',b.freeze)
        review=b.pin('final-review.json',dict(decision='PASS_R6_FINAL_FREEZE',must_fix=[],reviewer='fixture-independent',freeze_sha256=freeze_pin['sha256']))
        public=commitment('12'*32,protocol_sha256=b.freeze['protocol']['sha256'],freeze_sha256=freeze_pin['sha256']);realization=reveal(public,'12'*32)
        pairs=[('baseline',i) for i in range(3)] if b.freeze['branch']=='BASELINE_SUFFICIENT' else [(role,0) for role in ('baseline','seed1','seed2','seed3')]
        b.batch=dict(schema_version='r6-final-batch-inputs/1',freeze=freeze_pin,freeze_review=review,holdout_event_key='fixture',publication=b.pin('public.json',public),
            publication_commit='a'*40,realization=b.pin('realization.json',realization),protocol=b.freeze['protocol'],scientific_acceptance=False,dispatch_authorized=False,
            assignments=[dict(model_role=role,checkpoint=b.freeze['models'][role],realization_index=i,evaluation_seed=realization['realization_seeds'][i],split='final') for role,i in pairs])
        return dict(schema_version='r6-final-selection/1',batch=b.pin('batch.json',b.batch),assignment_index=1)
    b.final_save=save;b.value=save();return b


def test_confirmed_models_use_exact_terminal_and_common_final_stream(ready):
    b=ready;results=[f.validate(dict(b.value,assignment_index=i)) for i in range(4)]
    assert [x['model']['training_seed'] for x in results]==[20260908,21,22,23]
    assert [x['model']['update'] for x in results]==[640,896,896,896]
    assert len({x['comparison']['seed'] for x in results})==1
    assert results[0]['method']is None
    assert all(x['comparison']['split']=='final' and not x['dispatch_authorized'] for x in results)


def test_baseline_sufficient_three_streams_keep_original_model_method(ready):
    b=ready;b.freeze.update(branch='BASELINE_SUFFICIENT',recipe=b.ev,models={'baseline':b.ev},
        selection_rule=dict(kind='unchanged_baseline',update=640,training_transitions=0,final_realization_indices=[0,1,2],
            all_original_and_registered_blocks_each_stream=True,select_using_holdout=False))
    value=b.final_save();rows=[f.validate(dict(value,assignment_index=i)) for i in range(3)]
    assert len({r['comparison']['seed'] for r in rows})==3
    assert all(r['model']['checkpoint']==b.ev and r['method']is None for r in rows)


@pytest.mark.parametrize('fault',['seed','role','index','phase','assignment_missing','nonce','review','commit','model','rule','seed_order','terminal','metadata','development_selector'])
def test_final_selection_rejects_substitution(ready,fault):
    b=ready;value=b.value
    if fault=='seed':b.batch['assignments'][1]['evaluation_seed']+=1
    if fault=='role':b.batch['assignments'][1]['model_role']='seed2'
    if fault=='index':value['assignment_index']=True
    if fault=='phase':b.batch['assignments'][1]['split']='development'
    if fault=='assignment_missing':b.batch['assignments'].pop()
    if fault=='nonce':
        x=f.base.pinned_json(b.batch['realization']);x['nonce']='34'*32;b.batch['realization']=b.pin('changed-realization.json',x)
    if fault=='review':b.batch['freeze_review']=b.pin('rejected.json',dict(decision='MUST_FIX',must_fix=['fixture'],reviewer='fixture'))
    if fault=='commit':b.batch['publication_commit']='not-a-commit'
    if fault in ('model','rule','seed_order','terminal'):
        if fault=='model':b.freeze['models']['seed1']=b.ev
        if fault=='rule':b.freeze['selection_rule']['select_using_holdout']=True
        if fault=='seed_order':b.freeze['evidence']['streams'].reverse()
        if fault=='terminal':b.freeze['selection_rule']['target_update']=897
        value=b.final_save()
    if fault=='metadata':
        stream=b.freeze['evidence']['streams'][0];bound=f.base.pinned_json(stream['analysis']);report=b.reports[0]
        report['summary']['checkpoints'][0]['state_audit']['adam_state_loaded']=False
        bound['report']=b.pin('bad-report.json',report);stream['analysis']=b.pin('bad-analysis.json',bound);value=b.final_save()
    if fault=='development_selector':value=b.selection
    else:value=dict(value,batch=b.pin('batch.json',b.batch))
    with pytest.raises((PermissionError,ValueError)):f.validate(value)
