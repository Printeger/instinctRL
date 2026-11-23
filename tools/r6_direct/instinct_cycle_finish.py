"""Wait for the one frozen training stream, independently analyze, then evaluate it."""

import json
import os
from pathlib import Path
import subprocess
import time

from tools.r6_direct.import_legacy import ROOT, pin, process_identity, stop_paths
from tools.r6_direct.instinct_cycle import FORMAL, HISTORICAL, EVAL, DIAG, active_training_inputs, load, save

PYTHON = '/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11'
SYSTEM_PYTHON = '/usr/bin/python3'


def guard():
    if paths := stop_paths():
        raise RuntimeError(f'STOP {paths}')


def wait_closed(inputs, *, minutes):
    root = Path(inputs['output_root'])
    deadline = time.monotonic() + minutes * 60
    while time.monotonic() < deadline:
        guard()
        receipt_path = root / 'receipt.json'
        if receipt_path.exists():
            receipt = load(receipt_path)
            if receipt['status'] != 'CLOSED' or receipt['remaining']:
                raise RuntimeError(f'Unclosed job: {root}')
            if not any(process_identity(p['pid']) == p for p in [receipt['parent'], *receipt['owned_processes']]):
                return receipt
        time.sleep(5)
    raise TimeoutError(f'Exact job did not close: {root}')


def command(snapshot, module, *args, system=False, check=True):
    guard()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONPATH=f'{snapshot}:{snapshot / "isaaclab-training"}')
    argv = [SYSTEM_PYTHON if system else PYTHON, '-B', '-P', '-m', f'tools.r6_direct.{module}', *map(str,args)]
    print(' '.join(argv), flush=True)
    subprocess.run(argv, cwd=snapshot, env=env, check=check)


def freeze(tag, message, *paths):
    guard()
    files = [str(p.relative_to(ROOT)) for p in paths]
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True)
    if staged.strip():
        raise RuntimeError('Cannot freeze while other changes are staged')
    subprocess.run(['git', 'add', '--', *files], check=True, cwd=ROOT)
    subprocess.run(['git', 'diff', '--cached', '--check'], check=True, cwd=ROOT)
    subprocess.run(['git', '-c', 'commit.gpgsign=false', 'commit', '-m', message, '--', *files], check=True, cwd=ROOT)
    subprocess.run(['git', 'tag', '-a', tag, '-m', message], check=True, cwd=ROOT)


def analyze(snapshot, inputs, output, *extra):
    if not output.exists():
        command(snapshot, 'analyze', '--inputs', inputs, '--destination', output, *extra)
    return load(output / 'report.json')


def recover_probability(snapshot, train_path, attempt):
    command(snapshot, 'instinct_cycle', 'diagnostic')
    diag_path = FORMAL / (DIAG + '-inputs.json')
    freeze('instinct-first-cycle-probability-inputs', 'Freeze exact same-device probability replay',
           diag_path, FORMAL / 'RESOURCE_LEDGER.json')
    command(snapshot, 'cuda_diagnostic', '--plan', diag_path, system=True, check=False)
    diag = load(diag_path)
    wait_closed(diag, minutes=3)
    command(snapshot, 'instinct_cycle', 'seal')
    return analyze(snapshot, train_path, attempt / 'analysis',
                   '--probability-diagnostic', Path(diag['attempt_dir']) / 'sealed-evidence.json',
                   '--reuse-cpu-binding', attempt / 'analysis-cpu/binding.json')


def main():
    train = active_training_inputs()
    train_path = FORMAL / (train['run_id'] + '-inputs.json')
    snapshot, attempt = Path(train['snapshot']), Path(train['attempt_dir'])
    print('Waiting for exact trained workload:', train['run_id'], flush=True)
    wait_closed(train, minutes=90)
    report = analyze(snapshot, train_path, attempt / 'analysis-cpu')
    if report['issues']:
        if [r['input'] for r in report['issues']] != ['first_update_probability_replay']:
            raise RuntimeError(f'Training audit has other unresolved issues: {attempt / "analysis-cpu"}')
        report = recover_probability(snapshot, train_path, attempt)
    if report['issues']:
        raise RuntimeError('Training audit remains incomplete')
    command(snapshot, 'instinct_cycle', 'evaluate')
    eval_path = FORMAL / (EVAL + '-inputs.json')
    freeze('instinct-first-cycle-formal-eval-inputs', 'Freeze the trained checkpoint and full paired evaluation',
           eval_path, FORMAL / 'EVAL-contract.json', FORMAL / 'EVAL-comparison.json', FORMAL / 'RESOURCE_LEDGER.json')
    command(snapshot, 'run', '--inputs', eval_path, system=True, check=False)
    evaluation = load(eval_path)
    wait_closed(evaluation, minutes=3)
    evaluate_report = analyze(snapshot, eval_path, Path(evaluation['attempt_dir']) / 'analysis')
    if evaluate_report['candidate_binding_issues']:
        raise RuntimeError('Candidate evaluation evidence is incomplete')
    command(snapshot, 'instinct_cycle', 'close')
    pin_file = FORMAL / 'FORMAL_EVIDENCE.json'
    save(pin_file, dict(training_report=pin(attempt / ('analysis' if (attempt / 'analysis').exists()
                        else 'analysis-cpu') / 'report.json'),
                        evaluation_report=pin(Path(evaluation['attempt_dir']) / 'analysis/report.json'),
                        references=pin(HISTORICAL / 'RESULTS.json'),
                        ledger=pin(FORMAL / 'RESOURCE_LEDGER.json')))
    print(json.dumps(pin(pin_file), indent=2), flush=True)


if __name__ == '__main__':
    main()
