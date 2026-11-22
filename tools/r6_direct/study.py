"""Serial execution and evidence handoff for the frozen third-training comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.import_legacy import ROOT, pin, process_identity, stop_paths
from tools.r6_direct.io import atomic
from tools.r6_direct.lifecycle import writer
from tools.r6_direct.resource_policy import load_run_ledger
from tools.r6_direct.settle import settle_time
from tools.r6_direct.storage import SYSTEM_ROOT, evidence_bytes
from tools.r6_evidence import canonical, persist_exact
from training_workloads.unified_h1.direct_selection import training_model, verify_publication


DOCS = ROOT / 'docs/experiments/third-training'
FORMAL = DOCS / 'formal'
DIAGNOSTIC_TEMPLATE = ROOT / ('docs/experiments/r6-direct-training/continuation-20261101'
                              '/probability-cuda-06/plan.json')
PYTHON = '/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def publish(message, *paths):
    files = [str(p.relative_to(ROOT)) for p in paths]
    if git('diff', '--cached', '--name-only'):
        raise ValueError('study requires an empty staging area')
    if not git('status', '--porcelain', '--', *files):
        return git('rev-parse', 'HEAD')
    subprocess.run(['git', '-C', ROOT, 'add', '--', *files], check=True)
    subprocess.run(['git', '-C', ROOT, 'diff', '--cached', '--check'], check=True)
    subprocess.run(['git', '-C', ROOT, 'commit', '-m', message], check=True)
    subprocess.run(['git', '-C', ROOT, 'push', 'origin', 'main'], check=True)
    return git('rev-parse', 'HEAD')


def store(path, data):
    if path.exists():
        if json.loads(path.read_bytes()) != data:
            raise ValueError(f'preexisting study evidence differs: {path}')
    else:
        persist_exact(path, canonical(data) + b'\n')
    return pin(path)


def guard():
    if (paths := stop_paths()):
        raise RuntimeError(f'STOP {paths}')


def analysis_command():
    return [PYTHON, '-B', '-m', 'tools.r6_direct.analyze']


def run_command(args, *, live=False):
    code = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               CUDA_VISIBLE_DEVICES='', PYTHONPATH=f'{code}:{code / "isaaclab-training"}')
    result = subprocess.run(args, cwd=code, env=env, check=not live)
    return result.returncode


def latest_receipt(freeze):
    _, ledger, _, _ = load_run_ledger('R6-DIRECT-THIRD-STUDY', freeze['common']['resource_policy'])
    rows = [r for r in ledger['new_reservations'] if 'live_starts' in r['resources']]
    return rows[-1]['receipt']


def run_closed(input_path, plan, module, flag):
    root = Path(plan['output_root'])
    receipt_path = root / 'receipt.json'
    if root.exists():
        launch_path = root / ('intent.json' if flag == '--plan' else 'launch.json')
        launch = json.loads(launch_path.read_bytes())
        parent = launch['parent']
        while process_identity(parent['pid']) == parent:
            time.sleep(5)
    else:
        guard()
        print(f'Starting {plan["run_id"]}', flush=True)
        run_command(['/usr/bin/python3', '-B', '-m', module, flag, str(input_path)],live=True)
    result = read(pin(receipt_path))
    if result['status'] != 'CLOSED' or result['remaining']:
        raise RuntimeError(f'Unclosed job {plan["run_id"]}')
    if any(process_identity(p['pid']) == p for p in [result['parent'], *result['owned_processes']]):
        raise RuntimeError(f'Live process after closure {plan["run_id"]}')
    return result


def settle(plan):
    guard()
    run_id = plan['run_id']
    with writer():
        path, ledger, _, _ = load_run_ledger(run_id,plan['resource_policy'])
        row, = (r for r in ledger['new_reservations'] if r['id'] == run_id)
        if 'settlement' not in row:
            row['closed_evidence_bytes'] = evidence_bytes(plan['attempt_dir'], plan['native_dir'])
            settle_time(ledger, run_id)
            ledger['status'] = 'FORMAL_RUNNING'
            atomic(path, ledger)


def input_path(freeze, job, *, payload):
    name = job['run_id']
    path = FORMAL / f'{name}-inputs.json'
    if path.exists():
        plan = read(pin(path))
        if plan['contract'] != job['contract'] or plan['run_id'] != name:
            raise ValueError('third-training preexisting plan drift')
        return path, plan
    attempt = SYSTEM_ROOT / name
    common = freeze['common']
    native = Path(read(common['native_volume'])['target'])
    plan = dict(common, run_id=name, attempt_dir=str(attempt), output_root=str(attempt / 'control'),
                native_dir=str(native / name), contract=job['contract'], result_payload_cap_bytes=payload,
                predecessor_receipt=latest_receipt(freeze))
    store(path, plan)
    publish(f'Freeze input for {name}', path, DOCS / 'RESOURCE_LEDGER.json')
    return path, plan


def analyze(input_path, dest, *, supplemental=None):
    guard()
    if not dest.exists():
        extra = [] if supplemental is None else ['--probability-diagnostic', str(supplemental)]
        run_command([*analysis_command(), '--inputs', str(input_path), '--destination', str(dest), *extra])
    return validate_binding(pin(dest / 'binding.json'))


def probability_replay(freeze, inputs, report):
    name = inputs['run_id'] + '-DIAGNOSTIC'
    path = FORMAL / f'{name}-inputs.json'
    attempt = SYSTEM_ROOT / name
    template = read(pin(DIAGNOSTIC_TEMPLATE))
    init = read(inputs['contract'])['parameters']['r6_training_plan']
    old_name = f'pilot_diagnostic/update_{init["start_update"]+1}_full_input.pt'
    retained, = (row['pin'] for row in report['streams'] if row['logical_name'] == old_name)
    request = dict(contract=inputs['contract'], retained=retained, backend='isaaclab_seeded_reset')
    if path.exists():
        plan = read(pin(path))
        if plan['probability_replay'] != request:
            raise ValueError('third-training probability inputs changed')
    else:
        plan = dict(template, run_id=name, system_results=True,
                    launcher_script=pin(Path(__file__).with_name('cuda_diagnostic.py')),
                    native_dir=str(Path(inputs['native_dir']).with_name(name)),
                    attempt_dir=str(attempt), output_root=str(attempt / 'control'),
                    probability_replay=request, predecessor_receipt=latest_receipt(freeze),
                    resource_policy=freeze['common']['resource_policy'])
        store(path, plan)
        publish(f'Freeze CUDA probability replay for {inputs["run_id"]}', path, DOCS / 'RESOURCE_LEDGER.json')
    receipt = run_closed(path, plan, 'tools.r6_direct.cuda_diagnostic', '--plan')
    settle(plan)
    if receipt['exit_code'] != 0 or receipt['error'] is not None:
        raise RuntimeError(f'CUDA probability gate failed {name}')
    root = attempt / 'control'
    result = pin(Path(plan['native_dir']) / 'probability_result.json')
    if pin(attempt / 'probability_result.json')['sha256'] != result['sha256']:
        raise ValueError('third-training diagnostic result mirror drift')
    sealed = attempt / 'sealed-evidence.json'
    store(sealed, dict(schema_version='r6-probability-sealed-evidence/1', plan=pin(path),
        run_inputs=pin(root / 'RUN_INPUTS.json'), intent=pin(root / 'intent.json'),
        receipt=pin(root / 'receipt.json'), result=result, source_trace=plan['source_trace'],
        origin=init['initialization']['checkpoint']))
    return sealed


def train(freeze, job):
    contract = read(job['contract'])
    path, inputs = input_path(freeze, job, payload=contract['parameters']['output_cap_bytes'])
    run_closed(path, inputs, 'tools.r6_direct.run', '--inputs')
    attempt = Path(inputs['attempt_dir'])
    bound, _, report = analyze(path, attempt / 'analysis-cpu')
    if report['issues']:
        if any(row['input'] != 'first_update_probability_replay' for row in report['issues']):
            raise RuntimeError(f'Training audit failed {attempt / "analysis-cpu"}')
        supplemental = probability_replay(freeze, inputs, report)
        bound, _, report = analyze(path, attempt / 'analysis', supplemental=supplemental)
    terminal, = [row for row in report['summary']['checkpoints'] if row['update'] == 768]
    binding = bound['report']['path'].replace('/report.json', '/binding.json')
    binding_pin = pin(binding)
    plan = contract['parameters']['r6_training_plan']
    model = training_model(plan, binding_pin, terminal['pin'], read(freeze['methods'][job['arm']]))
    settle(inputs)
    print(f'Accepted {inputs["run_id"]} {model["checkpoint"]["sha256"]}', flush=True)
    return dict(job, model=model, training_plan=plan, training_analysis=binding_pin)


def model_freeze(freeze, models):
    path = FORMAL / 'MODELS.json'
    template = read(freeze['evaluation_template'])
    published = store(path, dict(freeze=pin(FORMAL / 'FREEZE.json'), models=models, origin=freeze['origin']))
    commit = publish('Commit all third-training checkpoints before holdout reveal', path, DOCS / 'RESOURCE_LEDGER.json')
    verify_publication(commit, published, pin(FORMAL / 'FREEZE.json'))
    nonce = (SYSTEM_ROOT / '.third-holdout-nonce').read_text().strip()
    if hashlib.sha256(nonce.encode()).hexdigest() != Path(freeze['holdout_commitment']['path']).read_text().strip():
        raise ValueError('held-out nonce does not match the frozen commitment')
    seed = int.from_bytes(hashlib.sha256(('instinctrl-third-holdout/1 ' + nonce).encode()).digest()[:4],'big') % 2**31
    reveal = FORMAL / 'HOLDOUT.json'
    if reveal.exists():
        prior = read(pin(reveal))
        verify_publication(prior['publication_commit'],published,pin(FORMAL / 'FREEZE.json'))
        commit=prior['publication_commit']
    store(reveal, dict(models=published, publication_commit=commit, nonce=nonce, seed=seed))
    baseline = dict(arm='baseline',pair=0,model=dict(checkpoint=freeze['origin'], training_seed=20260908),
                    training_analysis=None,training_plan=None)
    template = template['parameters']
    evaluations = []
    from tools.r6_direct.prepare import build
    for m in [baseline, *models]:
        arm = m['arm']
        method = freeze['methods']['control' if arm == 'baseline' else arm]
        run_id = f'R6-DIRECT-THIRD-EVAL-{arm.upper()}-{m["pair"]:02}'
        r6 = dict(template['r6'], run_id=run_id, seed=seed)
        comparison = dict(read(template['evaluation_selection']['comparison']),
            id=f'THIRD-HOLDOUT-{arm}', candidate_id='R6-CANDIDATE-THIRD-'+arm.upper(), seed=seed, method=method)
        comparison_path = FORMAL / f'{run_id}-comparison.json'
        selected = dict(template['evaluation_selection'], method=method,
            role='baseline_control' if arm == 'baseline' else 'trained_candidate',
            checkpoint=m['model']['checkpoint'], training_plan=m['training_plan'],
            training_analysis=m['training_analysis'], comparison=store(comparison_path, comparison))
        params = dict(template, experiment_id=run_id, r6=r6, output_cap_bytes=freeze['evaluation_payload_bytes'],
                      evaluation_selection=selected, training_seed=m['model']['training_seed'])
        contract = FORMAL / f'{run_id}-contract.json'
        contract_pin = store(contract, build('contract', dict(operation='evaluate',parameters=params)))
        evaluations.append(dict(arm=arm,pair=m['pair'],run_id=run_id,contract=contract_pin))
    publish('Reveal third-training holdout and freeze seven evaluation contracts', reveal,
            *sorted(FORMAL.glob('R6-DIRECT-THIRD-EVAL-*-comparison.json')),
            *sorted(FORMAL.glob('R6-DIRECT-THIRD-EVAL-*-contract.json')))
    return evaluations


def evaluate(freeze, job):
    path, inputs = input_path(freeze, job, payload=freeze['evaluation_payload_bytes'])
    run_closed(path, inputs, 'tools.r6_direct.run', '--inputs')
    bound, _, report = analyze(path, Path(inputs['attempt_dir']) / 'analysis')
    if report['candidate_binding_issues']:
        raise RuntimeError(f'Evaluation evidence failed {job["run_id"]}')
    from tools.r6_direct.study_report import score_model
    result=dict(job,report=bound['report'])
    score=score_model(result)
    store(Path(inputs['attempt_dir'])/'paired-scoring.json',score)
    settle(inputs)
    if not score['safety_pass']:
        raise RuntimeError(f'Frozen hard guardrail failed {job["run_id"]}; see paired-scoring.json')
    print(f'Audited {inputs["run_id"]}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args()
    guard()
    freeze_pin = pin(FORMAL / 'FREEZE.json')
    verify_publication(git('rev-parse','third-training-formal-inputs'), freeze_pin)
    freeze = read(freeze_pin)
    models = [train(freeze, job) for job in freeze['jobs']]
    jobs = model_freeze(freeze,models)
    evaluations = [evaluate(freeze,job) for job in jobs]
    from tools.r6_direct.study_report import summarize
    result = summarize(freeze,evaluations)
    final = FORMAL / 'RESULTS.json'
    store(final,result)
    publish('Publish third-training paired evaluation and final resource ledger',final,DOCS/'RESOURCE_LEDGER.json')
    print(final,flush=True)


if __name__ == '__main__':
    main()
