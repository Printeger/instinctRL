"""Real Journal, pinned files and output inventory; frozen execution boundaries mocked."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from tools import r6_final_models as f
from tools import r6_controller as controller
from tools.test_r6_profile_preparation import case
from tools.r6_recovery import RecoveryError,file_pin


@pytest.fixture
def ready(case,monkeypatch):
    b=case;baseline=b.write('baseline.pt',{'fixture':'u640'})
    b.recipe=b.write('final-recipe.json',dict(source_identity={'workload_commit':'a'*40},
        branch_updates=256,confirmation_seeds=[21,22,23]))
    rule=dict(kind='terminal_checkpoint',common_start_update=640,target_update=896,select_using_holdout=False)
    selection=b.write('final-selection.json',dict(protocol=b.context,checkpoint_selection=rule))
    freeze=b.write('recipe-freeze.json',dict(recipe=b.recipe,selection_evidence=selection))
    snapshot=b.write('final-snapshot.json',dict(config=b.context,source_identity={'workload_commit':'a'*40}))
    q=b.write('final-input.json',dict(snapshot=snapshot,recipe=b.recipe,recipe_review=b.context))
    contexts=[];records=[];plans=[];b.reports=[];b.inventory=[]
    for i in range(3):
        plan=dict(experiment_id='R6-CONFIRM-'+str(i),phase='confirmation',recipe=b.recipe,
            selection_freeze=freeze,training_seed=21+i,start_update=640,target_update=896,
            initialization=dict(kind='COMMON_U640_NEW_STREAM',checkpoint=baseline,
                source_identity={'fixture':'baseline-source'},method_hashes={'fixture':'baseline-method'}))
        plans.append(plan)
        draft=b.write(f'final-profile-{i}.json',dict(r6_training_plan=plan,baseline_bindings=b.context))
        contexts.append(b.write(f'final-context-{i}.json',dict(baseline_profile=draft,snapshot_receipt=snapshot)))
        attempt=b.root/f'attempt-{i}';model=attempt/'results/research-r6-training/checkpoints/u896.pt'
        model.parent.mkdir(parents=True);model.write_text('distinct trained fixture '+str(i));pin=file_pin(model)
        report=dict(summary={'checkpoints':[dict(update=896,pin=pin)]});b.reports.append(report)
        inventory=dict(files=[dict(pin,bytes=model.stat().st_size)]);b.inventory.append(inventory)
        bound=b.write(f'final-bound-{i}.json',dict(report=b.write(f'final-report-{i}.json',report),
            authorization=b.write(f'final-auth-{i}.json',dict(mode='research-r6-training',attempt=str(attempt),parameters={'r6_training_plan':plan})),
            inventory=b.write(f'final-inventory-{i}.json',inventory),native_status='FAILED',native_exit_code=-9))
        records.append(dict(job_event_key='job-'+str(i),context=contexts[i],analysis=bound,scientific_acceptance=False))
    bundle=b.write('final-bundle.json',{'bundle':dict(plans=plans,freeze=freeze,baseline_bindings=b.context)})
    with b.j.writer():
        req=dict(inputs=q);event=dict(goal_id=b.j.read()['goal_id'],event_id='fixture-confirmation',job_id='fixture',
            input_sha256=f.sha(req),role=f.confirmation.ROLE)
        key,_=b.j.intent(event,kind='cpu',request=req,resources={'output_actual_plus_unreleased_reservations_bytes':f.confirmation.CAP},purpose='confirmation_delivery',free_disk_bytes=500*1024**3)
        prepared=dict(status='THREE_CONFIRMATION_CONTEXTS_READY_NOT_RESERVED',event_key=key,contexts=contexts,snapshot=snapshot,freeze=freeze,plans=bundle)
        p=b.j.directory/'confirmation-preparation'/key/'result.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(prepared))
        initial=b.write('controller-initial.json',dict(plan_pin=b.context,master_pin=b.context,review_executable=file_pin(b.builder)))
        request=b.write('controller-input.json',dict(initial_request=initial,profile_context=b.context))
        b.controller,_=controller.register(b.j,request,b.store,free_disk_bytes=500*1024**3)
        state=b.j.read();state['events'][b.controller]['cursor']=dict(phase='FINAL_MODEL_PREPARATION',confirmation=dict(preparation=prepared,contexts=contexts,results=records));b.j.save(state)
    b.records=records;b.plans=plans;b.baseline=baseline
    monkeypatch.setattr(f,'verify_frozen',Mock())
    monkeypatch.setattr(f.training,'recipe',lambda pin,review:json.loads(f.checked(pin).read_text()))
    monkeypatch.setattr(f.training,'validate',lambda plan:plan)
    monkeypatch.setattr(f.base,'baseline',lambda p:dict(selected_model=baseline,
        source_identity={'fixture':'baseline-source'},method_hashes={'fixture':'baseline-method'}))
    monkeypatch.setattr(f.confirmation,'collect',lambda j,key,context,store:copy.deepcopy(next(r for r in records if r['job_event_key']==key)))
    return b


def run(b):
    with b.j.writer():return f.prepare(b.j,b.controller,b.store,free_disk_bytes=500*1024**3)


def test_exact_three_terminal_models_and_replay_no_new_charge(ready):
    b=ready;result=run(b);budget=b.j.read()['budget'];assert run(b)==result and b.j.read()['budget']==budget
    freeze=json.loads(f.checked(result['freeze']).read_text())
    assert freeze['models']['baseline']==b.baseline and set(freeze['models'])=={'baseline','seed1','seed2','seed3'}
    assert all(s['native_status']=='FAILED' and s['native_exit_code']==-9 for s in freeze['evidence']['streams'])
    assert not result['holdout_released'] and not result['scientific_acceptance']
    with b.j.writer():assert f.validate(b.j,freeze,b.store)==freeze


@pytest.mark.parametrize('fault',['missing','duplicate','context','collection','store','seed','rule','inventory','node','model_drift'])
def test_incomplete_or_changed_confirmation_never_creates_final_payload(ready,fault,monkeypatch):
    b=ready
    with b.j.writer():
        state=b.j.read();seq=state['events'][b.controller]['cursor']['confirmation']
        if fault=='missing':seq['results'].pop()
        if fault=='duplicate':seq['results'][1]=seq['results'][0]
        if fault=='context':seq['contexts'][0]=b.context
        if fault=='store':state['events'][b.controller]['request']['store_root']='/wrong-store'
        b.j.save(state)
    if fault=='collection':monkeypatch.setattr(f.confirmation,'collect',lambda *a:{})
    if fault=='seed':
        r=json.loads(f.checked(b.recipe).read_text());r['confirmation_seeds']=[22,21,23]
        monkeypatch.setattr(f.training,'recipe',lambda *a:r)
    if fault=='rule':
        r=json.loads(f.checked(b.recipe).read_text());r['branch_updates']=512
        monkeypatch.setattr(f.training,'recipe',lambda *a:r)
    if fault in ('inventory','node'):
        bound=json.loads(f.checked(b.records[0]['analysis']).read_text())
        if fault=='inventory':
            b.inventory[0]['files'][0]['bytes']+=1
            bound['inventory']=b.write('changed-inventory.json',b.inventory[0])
        else:
            b.reports[0]['summary']['checkpoints'][0]['update']=895
            bound['report']=b.write('changed-report.json',b.reports[0])
        b.records[0]['analysis']=b.write('changed-bound.json',bound)
        with b.j.writer():
            state=b.j.read();state['events'][b.controller]['cursor']['confirmation']['results'][0]=copy.deepcopy(b.records[0]);b.j.save(state)
    if fault=='model_drift':Path(b.reports[0]['summary']['checkpoints'][0]['pin']['path']).write_text('changed')
    with pytest.raises((RecoveryError,KeyError)):run(b)
    assert not list((b.j.directory/'final-models').rglob('freeze.json'))


def test_review_cannot_substitute_an_earlier_model(ready):
    b=ready;out=run(b);freeze=json.loads(f.checked(out['freeze']).read_text());freeze['models']['seed1']=b.baseline
    with b.j.writer(),pytest.raises(RecoveryError,match='FREEZE_EVIDENCE_DRIFT'):f.validate(b.j,freeze,b.store)


def test_stop_prevents_any_final_preparation(ready):
    b=ready;before=b.j.read();(b.j.directory/'STOP').touch()
    with pytest.raises(RecoveryError,match='STOP'):run(b)
    assert b.j.read()==before
