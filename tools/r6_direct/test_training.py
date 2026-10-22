"""CPU contract and orchestration checks; no physics or actual PPO updates."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.test_r6_training_plan import case
from tools.test_r6_training_execution import setup, run
from tools.test_r6_training_contract import stage
from tools.test_r6_training_rng import preserve_global_rng
from training_workloads.unified_h1 import robustness_training_plan as plans
from training_workloads.unified_h1.direct_contract import training_runtime_contract, training_inputs
from tools.r6_direct.run import run_resources


def local_contract(plan, identity):
    plan['schema_version']='r6-local-training-plan/1'
    plan.pop('recipe_review')
    baseline=json.loads(Path('docs/experiments/r6-direct-training/BASELINE_CONTRACT.json').read_text())
    return training_runtime_contract(dict(experiment_id=plan['experiment_id'],group='R1',
        training_seed=plan['training_seed'],evaluation_seed=20260909,
        baseline_bindings=baseline['parameters']['baseline_bindings'],r6_training_plan=plan,
        r6_expected_mass_kg=1.181517923,output_cap_bytes=20250099712,source_identity=identity))


def test_local_training_reserves_actual_updates_and_checkpoints(case):
    contract=local_contract(case.plan,case.identity)
    result=run_resources(contract)
    assert result['training_transitions']==256*512
    assert result['training_checkpoint_writes']==1
    assert result['major_scientific_candidates']==1
    assert result['nontraining_transitions']==0
    assert result['live_wall_seconds']==case.plan['wall_seconds']+190
    wrong=copy.deepcopy(contract);wrong['budgets']['training_transitions']=0
    with pytest.raises(ValueError):run_resources(wrong)


def test_local_plan_is_explicit_and_cannot_enter_legacy_or_confirmation(case):
    contract=local_contract(case.plan,case.identity)
    with pytest.raises(ValueError,match='LEGACY_PLAN_REQUIRED'):
        training_inputs(SimpleNamespace(authorization=contract),case.plan)
    wrong=copy.deepcopy(case.plan);wrong['recipe_review']={'invented':True}
    with pytest.raises(PermissionError):plans.validate(wrong)
    wrong=copy.deepcopy(case.plan);wrong['phase']='confirmation'
    with pytest.raises(PermissionError,match='LOCAL_CONFIRMATION'):plans.validate(wrong)
    wrong=copy.deepcopy(case.plan);wrong['training_seed']=22
    with pytest.raises(PermissionError,match='UNREGISTERED_SEED'):plans.validate(wrong)


def test_local_loaded_execution_keeps_rng_reset_and_checkpoint_order(setup):
    s=setup;s.b.runtime_contract=local_contract(s.p,s.identity)
    s.b.authorization=None;s.b.execution_mode='local-direct'
    run(s)
    assert s.b.authorization is None
    assert [e[1] for e in s.events if e[0]=='reset']==[640*16,641*16]
    assert [e[1] for e in s.events if e[0]=='checkpoint']==[642]
    assert s.b.journal.evidence_finalized
    assert all(loop.kw['optimizer_step_origin']==(5120,5120) for loop in s.loops)


def test_legacy_dispatch_rejects_local_plan(stage):
    from training_workloads.unified_h1.robustness_training_contract import validate_parameters
    case,parameters=stage
    case.plan['schema_version']='r6-local-training-plan/1'
    case.plan.pop('recipe_review')
    with pytest.raises(PermissionError,match='LEGACY_PLAN_REQUIRED'):
        validate_parameters('research-r6-training',parameters)
