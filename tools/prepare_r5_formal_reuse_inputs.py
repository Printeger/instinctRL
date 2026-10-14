"""Prepare a new research card against an existing immutable source snapshot.

CPU only. No approval, release, new snapshot or existing evidence mutation.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/experiments/r5-formal-readiness-auto'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_id')
    parser.add_argument('--parent', required=True)
    args = parser.parse_args()
    from rlflow.store import Store
    from rlflow.releases import verify
    from tools.r5_formal_commander import idle
    from training_workloads.unified_h1.budget import required_mode_budget
    control = ROOT / 'coordination/research'
    idle(Store(control))
    directory = BASE / 'experiments' / args.experiment_id
    cardpath = directory / 'EXPERIMENT_CARD.json'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert subprocess.check_output(['git', 'show', commit + ':' + str(cardpath.relative_to(ROOT))], cwd=ROOT) == cardpath.read_bytes()
    card = read(cardpath)
    assert card['experiment_id'] == args.experiment_id
    parentpath = BASE / 'experiments' / args.parent / 'launch_inputs.proposed.json'
    parent = read(parentpath)
    baseline = parent['baseline']
    snapshot = Path(baseline['snapshot'])
    configpath = snapshot / 'training_workloads/unified_h1/candidate_config.json'
    config = read(configpath)
    assert sha(configpath) == baseline['config_sha256']
    assert subprocess.check_output(['git', '-C', str(snapshot), 'rev-parse', 'HEAD'], text=True).strip() == baseline['snapshot_commit']
    assert subprocess.check_output(['git', '-C', str(snapshot), 'rev-parse', 'HEAD^{tree}'], text=True).strip() == baseline['snapshot_tree']
    for section in ['product', 'workload']:
        for relative, digest in config[section]['source_sha256'].items():
            assert sha(Path(config[section]['path']) / relative) == digest, relative
    from training_workloads.unified_h1.b0_handoff import _candidate_diff
    assert hashlib.sha256(_candidate_diff(snapshot, baseline['snapshot_commit'])).hexdigest() == baseline['snapshot_diff_sha256']
    verify(Path(baseline['tool_release_path']))
    rows = []
    for group, profile in card['profiles'].items():
        for mode in ['research-training', 'research-evaluation']:
            training = mode == 'research-training'
            row = deepcopy(next(r for r in parent['entries'] if r['group'] == group and r['mode'] == mode))
            spec, auth = row['experiment_proposal'], row['authorization_proposal']
            parameters = {**deepcopy(profile), 'experiment_card': {'path': str(cardpath), 'sha256': sha(cardpath)}}
            budget = required_mode_budget('research-training', parameters)
            if not training:
                count = len(parameters['evaluation_updates'])
                budget.update(training_transitions=0, nontraining_transitions=count * 6400,
                              optimizer_updates=0, checkpoint_writes=0, explicit_resets=count,
                              automatic_resets=count * 6400, wall_seconds=1800)
            suffix = baseline['snapshot_commit'][:12]
            attempt = control / 'attempts' / f'{args.experiment_id}-{group}-{mode.upper()}-{suffix}'
            trainattempt = control / 'attempts' / f'{args.experiment_id}-{group}-RESEARCH-TRAINING-{suffix}'
            assert not attempt.exists()
            authpath = attempt / 'commands/AUTHORIZATION.json'
            if not training:
                parameters['checkpoints'] = [{'path': str(trainattempt / f'results/research-training/checkpoints/u{u}.pt'), 'sha256': None, 'update': u} for u in parameters['evaluation_updates']]
            auth.update(order_id=f'{args.experiment_id}-{group}-{mode.upper()}', attempt=str(attempt), mode=mode, budgets=budget, parameters=parameters)
            spec.update(id=attempt.name, seed=profile['training_seed'] if training else profile['evaluation_seed'],
                        argv=[config['execution']['python_path'], '-B', '-m', 'training_workloads.unified_h1.formal_unified_h1', '--authorization', str(authpath), '--attempt', str(attempt), '--mode', mode],
                        resources={'starts': 1, 'wall_seconds': budget['wall_seconds'] + 10, 'gpu_seconds': budget['wall_seconds'] + 10, 'transitions': budget['training_transitions'] + budget['nontraining_transitions']},
                        conditions=[{'evidence_id': 'E-R5-FORMAL-MASTER-DELEGATION-20261014', 'sha256': profile['master_authority']['sha256']}])
            spec['h1']['attempt'] = str(attempt)
            spec['pins']['authorization'] = {'path': str(authpath), 'sha256': None}
            row.pop('expected_authorization', None)
            if training:
                path = directory / f'inputs/{group}-training-AUTH.proposed.json'
                save(path, auth)
                spec['pins']['authorization']['sha256'] = sha(path)
                row['expected_authorization'] = {'path': str(path.relative_to(directory)), 'sha256': sha(path)}
            row.update(experiment_proposal=spec, authorization_proposal=auth)
            rows.append(row)
    packet = {**deepcopy(parent), 'experiment_id': args.experiment_id, 'entries': rows,
              'experiment_card': {'path': str(cardpath), 'sha256': sha(cardpath)},
              'status': 'PREPARED_REUSED_FROZEN_SOURCE_PENDING_EXACT_REVIEW',
              'parent_packet': {'path': str(parentpath), 'sha256': sha(parentpath)},
              'generator_sha256': sha(Path(__file__))}
    save(directory / 'launch_inputs.proposed.json', packet)
    save(directory / 'preparation-verification.json', {'status': 'PASS_CPU_PREPARATION_ONLY', 'input_commit': commit, 'baseline': baseline, 'all_pinned_source_files_verified': True, 'new_snapshots': 0, 'GPU_queries': 0, 'live_starts': 0})
    print(json.dumps({'packet_sha256': sha(directory / 'launch_inputs.proposed.json'), 'source_commit': baseline['snapshot_commit']}))


if __name__ == '__main__':
    main()
