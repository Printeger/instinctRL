"""Local confirmation keeps the existing scientific freeze checks; CPU fixtures."""
import copy
import pytest
from tools.test_r6_training_plan import case
from tools.test_r6_confirmation import ready, frozen
from training_workloads.unified_h1 import robustness_confirmation as confirmation
from training_workloads.unified_h1 import robustness_training_plan as training


def test_selection_rejects_unevaluated_declared_terminal_node(ready,monkeypatch):
    b=ready
    b.selected['trained_candidate']['model']['update']=896
    monkeypatch.setattr(confirmation,'evaluation_evidence',
        lambda pin,protocol: b.selected[next(role for role,p in b.evidence['paired_evaluations'].items() if p==pin)])
    with pytest.raises(PermissionError,match='EVALUATED_ENDPOINT'):
        b.check()


def test_three_local_plans_keep_full_recipe_and_reviewed_evidence(frozen):
    arguments=dict(frozen.arguments,recipe_review=None,local=True)
    result=confirmation.build_plans(**arguments)
    assert [p['training_seed'] for p in result['plans']]==[21,22,23]
    for plan in result['plans']:
        assert plan['schema_version']=='r6-local-training-plan/1'
        assert 'recipe_review' not in plan
        assert plan['phase']=='confirmation' and plan['start_update']==640 and plan['target_update']==1664
        assert training.validate(plan)==plan
        wrong=copy.deepcopy(plan);wrong['selection_freeze']=None
        with pytest.raises((PermissionError,ValueError,TypeError)):training.validate(wrong)
    assert result['training_job_reservations']['training_transitions']==3*1024*512
    assert not result['scientific_acceptance'] and not result['dispatch_authorized']


def test_publication_checks_real_git_bytes_without_a_store():
    from pathlib import Path
    import subprocess
    from tools.r6_evidence import file_pin
    from training_workloads.unified_h1.direct_selection import verify_publication
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    pin=file_pin(Path('training_workloads/unified_h1/robustness_holdout.py'))
    verify_publication(head,pin)
    with pytest.raises(PermissionError):verify_publication('not-a-commit',pin)


def test_confirmation_resources_use_reserved_confirmation_budget(frozen):
    from tools.r6_direct.run import reserve
    ledger=dict(caps={'x':100},historical_charged={'x':10},new_used={'x':0},
        protected_confirmation_delivery_repair={'x':80},new_reservations=[])
    with pytest.raises(ValueError):reserve(copy.deepcopy(ledger),'development',{'x':11})
    result=reserve(copy.deepcopy(ledger),'confirmation',{'x':80},purpose='confirmation_delivery')
    assert result['historical_charged']==ledger['historical_charged']
    with pytest.raises(ValueError):reserve(result,'too-large',{'x':11},purpose='confirmation_delivery')


@pytest.mark.parametrize('fault',[None,'original','wall','block','measurement','raw','counter'])
def test_local_pair_evidence_preserves_scientific_checks(ready,monkeypatch,fault):
    from pathlib import Path
    from tools.r6_direct import evidence
    from tools import r6_candidate_analysis
    from training_workloads.unified_h1.direct_selection import evaluation_evidence
    b=ready;role='trained_candidate';selected=b.selected[role]
    selected['selection']['schema_version']='r6-local-evaluation-selection/1'
    bound=b.bindings[role];auth=confirmation.base.pinned_json(bound['authorization'])
    report=copy.deepcopy(b.reports[role])
    report['candidate']['selection_sha256']=confirmation.base.object_sha(selected['selection'])
    output=Path(auth['attempt'])/'results/research-r6-candidate-evaluation'
    ctx=dict(contract=dict(operation='evaluate',parameters=auth['parameters'],budgets={'nontraining_transitions':193600}),
             output=output,state={'counters':dict(nontraining_transitions=193600,training_transitions=0,real_optimizer_updates=0)})
    # Exact process closure and model restore are tested separately. Exercise the
    # unchanged complete replay/score/measurement requirements here with synthetic raw.
    monkeypatch.setattr(evidence,'validate_binding',lambda pin:(bound,ctx,report))
    monkeypatch.setattr(r6_candidate_analysis,'validate_binding',lambda binding,selection:None)
    if fault=='original':report['summary']['original']['all_gates_pass']=False
    if fault=='wall':report['summary']['original']['wall_motion']['per_lane'][0]=False
    if fault=='block':report['blocks'].pop()
    if fault=='measurement':report['blocks'][0]['measurement_replay']['bit_exact_delivery']=False
    if fault=='raw':report['inputs'].pop(2)
    if fault=='counter':ctx['state']['counters']['real_optimizer_updates']=1
    protocol=selected['comparison']['protocol']
    if fault is None:
        assert evaluation_evidence(b.ev,protocol)==selected
    else:
        with pytest.raises((PermissionError,ValueError)):evaluation_evidence(b.ev,protocol)
