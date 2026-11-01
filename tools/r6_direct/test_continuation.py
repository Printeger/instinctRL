"""Synthetic closed-run evidence; no simulator, GPU queries or training."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from tools.test_r6_training_plan import case
from tools.r6_direct import evidence as e
from tools.r6_evidence import file_pin
from training_workloads.unified_h1 import direct_contract as contracts
from training_workloads.unified_h1 import robustness_contract as base
from training_workloads.unified_h1 import robustness_evaluation_selection as selection


@pytest.fixture
def closed(case, tmp_path):
    b = case
    baseline = json.loads(Path('docs/experiments/r6-direct-training/BASELINE_CONTRACT.json').read_text())
    origin = base.baseline(baseline['parameters'])
    b.plan.update(schema_version='r6-local-training-plan/1', experiment_id='R6-DIRECT-CPU-CONTINUATION',
                  initialization=dict(kind='COMMON_U640_NEW_STREAM', checkpoint=origin['selected_model'],
                                      source_identity=origin['source_identity'], method_hashes=origin['method_hashes']))
    b.plan.pop('recipe_review')
    b.contract = contracts.training_runtime_contract(dict(experiment_id=b.plan['experiment_id'], group='R1',
        training_seed=11, evaluation_seed=20260909, baseline_bindings=baseline['parameters']['baseline_bindings'],
        r6_training_plan=b.plan, r6_expected_mass_kg=1.181517923, output_cap_bytes=20250099712, source_identity=b.identity))
    b.attempt = tmp_path/'attempt'; b.root = tmp_path/'run'
    b.output = b.attempt/'results/research-r6-training'
    (b.output/'checkpoints').mkdir(parents=True); b.root.mkdir()
    b.cp = b.pin('attempt/results/research-r6-training/checkpoints/u896.pt', {'explicit_synthetic_tensor_fixture': True})
    b.inputs = b.pin('inputs.json', dict(run_id=b.plan['experiment_id'], contract=b.pin('contract.json', b.contract),
        attempt_dir=str(b.attempt), output_root=str(b.root), snapshot=str(tmp_path)))
    plan = e.read(b.inputs)
    parent = dict(pid=99999991, starttime='1', boot_id='fixture'); child = dict(parent, pid=99999992)
    b.receipt = dict(status='CLOSED', parent=parent, owned_processes=[child], remaining=[], exit_code=-15,
                     error="TimeoutError('DIRECT_CLEANUP_TIMEOUT')")
    b.pin('run/receipt.json', b.receipt)
    b.pin('run/launch.json', dict(parent=parent, input=b.inputs))
    b.pin('run/RUN_INPUTS.json', dict(plan=plan, input=b.inputs, execution_mode='local-direct'))
    b.counters = dict(training_transitions=131072, nontraining_transitions=0, real_optimizer_updates=256,
                      completed_updates=256, real_checkpoint_writes=1)
    b.state = dict(status='PASS_PENDING_CLOSE', error=None, identity=child, counters=b.counters)
    b.pin('attempt/state.json', b.state)
    b.pin('attempt/CPU_MODULES.json', dict(execution_mode='local-direct', loaded_modules=['torch'],
        forbidden_modules=[], forbidden_pythonpath=[]))
    b.report = dict(schema_version='r6-training-analysis/1', scientific_acceptance=False,
        plan_sha256=base.object_sha(b.plan), issues=[], inputs=[dict(p, bytes=Path(p['path']).stat().st_size)
            for p in (b.cp, origin['selected_model'])], summary=dict(issue_count=0, probability_gate_recomputed=True,
                rng_initialization_audit={'status': 'PASS'}, expected_counters=b.counters, observed_counters=b.counters,
                checkpoints=[dict(pin=b.cp, update=896, metadata_verified=True,
                    state_audit=dict(actor_critic_state_loaded=True, adam_state_loaded=True))]))
    b.comparison = b.pin('comparison.json', dict(schema_version='r6-paired-comparison/1', id='CPU-PAIR',
        candidate_id=b.recipe['candidate_id'], hypothesis='CPU wiring only; no scientific result',
        protocol=baseline['parameters']['r6']['protocol'], split='development', seed=2026101901,
        block_ids=baseline['parameters']['r6']['block_ids'], method=b.recipe['method'], baseline_checkpoint_sha256=base.BASELINE_SHA))
    b.selection = dict(schema_version='r6-local-evaluation-selection/1', phase='development', role='trained_candidate',
        baseline_bindings=baseline['parameters']['baseline_bindings'], checkpoint=b.cp, training_plan=b.plan,
        method=b.recipe['method'], comparison=b.comparison)
    def save():
        ctx=e.context(b.inputs)
        e.add_inputs(b.report, b.inputs, ctx)
        report=b.pin('report.json', b.report)
        b.bound=b.pin('binding.json', e.bind(b.inputs, report))
        b.selection['training_analysis']=b.bound
    b.save=save; b.baseline=baseline; save()
    return b


def test_closed_training_to_exact_evaluation_without_legacy_authorization(closed):
    b=closed
    selected=selection.validate(b.selection)
    assert selected['model']['checkpoint']==b.cp
    p=copy.deepcopy(b.baseline['parameters'])
    p.update(experiment_id='R6-DIRECT-EVAL-CPU', training_seed=11, evaluation_selection=b.selection)
    p['r6']['run_id']=p['experiment_id']
    contract=contracts.evaluation_runtime_contract(p)
    assert contract['budgets']['nontraining_transitions']==193600
    assert contract['budgets']['training_transitions']==0
    assert 'authorization' not in e.read(b.bound) and e.read(b.bound)['native_exit_code']==-15
    assert contracts.validate_runtime_contract(contract)==contract
    wrong=copy.deepcopy(contract);wrong['budgets']['nontraining_transitions']-=32
    with pytest.raises(ValueError):contracts.validate_runtime_contract(wrong)
    wrong=copy.deepcopy(p);wrong['r6']['seed']+=1
    with pytest.raises(ValueError,match='COMPARISON'):contracts.evaluation_runtime_contract(wrong)
    wrong=copy.deepcopy(p);wrong['evaluation_selection']['schema_version']='r6-evaluation-selection/1'
    with pytest.raises(ValueError,match='LOCAL_SELECTION'):contracts.evaluation_runtime_contract(wrong)


@pytest.mark.parametrize('fault', ['report', 'gate', 'rng', 'adam', 'counter', 'model', 'receipt', 'failed', 'dependency'])
def test_direct_selection_rejects_incomplete_or_changed_evidence(closed, fault):
    b=closed
    if fault=='report':b.report['issues']=[{'reason':'missing stream'}]
    if fault=='gate':b.report['summary']['probability_gate_recomputed']=False
    if fault=='rng':b.report['summary']['rng_initialization_audit']['status']='FAIL'
    if fault=='adam':b.report['summary']['checkpoints'][0]['state_audit']['adam_state_loaded']=False
    if fault=='counter':
        b.state=copy.deepcopy(b.state);b.state['counters']['real_optimizer_updates']=0
        b.pin('attempt/state.json',b.state)
    if fault=='failed':
        b.state.update(status='FAIL_PENDING_CLOSE',error='fixture failure');b.pin('attempt/state.json',b.state)
    if fault=='receipt':
        b.receipt['remaining']=[b.receipt['owned_processes'][0]];b.pin('run/receipt.json',b.receipt)
    if fault=='dependency':
        b.pin('attempt/CPU_MODULES.json',dict(execution_mode='local-direct',loaded_modules=['rlflow.store'],forbidden_modules=[],forbidden_pythonpath=[]))
    if fault=='model':Path(b.cp['path']).write_text('changed model')
    with pytest.raises((ValueError,PermissionError,RuntimeError)):
        b.save();selection.validate(b.selection)


def test_local_training_analysis_reports_missing_data_instead_of_success(closed):
    from tools.r6_training_analysis import analyze
    report,summary=analyze(closed.output,closed.inputs,direct=True)
    assert summary['issue_count']>0 and not summary['scientific_acceptance']
    assert report['run_inputs']==closed.inputs and 'authorization' not in report
    assert closed.inputs['path'] in {p['path'] for p in report['inputs']}


def test_all_direct_entry_imports_with_scheduler_imports_blocked():
    script='''
from tools.r6_direct.dependencies import install
install()
import importlib
for name in ('tools.r6_direct.run','tools.r6_direct.worker','tools.r6_direct.analyze',
             'tools.r6_direct.prepare','tools.r6_direct.collect','tools.prepare_r6_video_trajectories',
             'tools.render_r6_isaac_video','tools.analyze_r6_training','tools.analyze_r6_candidate',
             'tools.analyze_r6_final','tools.analyze_r6_evaluation'):
    importlib.import_module(name)
from tools.r6_direct.source import audit_modules
assert not audit_modules()['forbidden_modules']
try: importlib.import_module('rlflow.store')
except ImportError as error: assert 'DIRECT_FORBIDDEN' in str(error)
else: raise AssertionError('scheduler import was permitted')
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='.:isaaclab-training')
    subprocess.run([sys.executable,'-B','-c',script],env=env,check=True,timeout=60)
