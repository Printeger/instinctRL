"""Freeze one committed W Goal config/source under the new ledger, CPU only.

No job, approval, GPU query, dependency change or scheduler is created here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/experiments/r5-w-closure-auto'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    from rlflow.store import Store
    from tools.r5_w_commander import idle, authority, checked, pin
    from training_workloads.unified_h1.config import PRODUCT_SOURCES, WORKLOAD_SOURCES
    control = ROOT / 'coordination/research'
    store = Store(control)
    idle(store)
    initial_ledger = read(BASE/'goal_budget.json')
    stagepath = checked(initial_ledger['stage_allocation_history'][-1])
    master, stage = authority(store, stagepath)
    assert initial_ledger['master_authority'] == pin(BASE/'MASTER_AUTHORITY.json')
    assert initial_ledger['caps'] == master['master_caps']
    configpath = args.config.resolve()
    assert BASE in configpath.parents and configpath.is_file() and not configpath.is_symlink()
    commit = git('rev-parse', 'HEAD')
    assert subprocess.check_output(['git', 'show', commit + ':' + str(configpath.relative_to(ROOT))], cwd=ROOT) == configpath.read_bytes()
    config = read(configpath)
    snapshot = Path(config['workload']['path'])
    assert snapshot.is_absolute() and snapshot.resolve() == snapshot and snapshot.parent == control / 'workspaces'
    assert not snapshot.exists() and config['product']['path'] == str(snapshot)
    for section, sources in [('product', PRODUCT_SOURCES), ('workload', WORKLOAD_SOURCES)]:
        spec = config[section]
        assert spec['candidate_commit'] == spec['candidate_tree'] == 'SELF' and spec['branch'] == ''
        assert set(spec['source_sha256']) == sources
        for relative, expected in spec['source_sha256'].items():
            data = subprocess.check_output(['git', 'show', commit + ':' + relative], cwd=ROOT)
            assert hashlib.sha256(data).hexdigest() == expected, relative
    ledgerpath = BASE / 'goal_budget.json'
    ledger = read(ledgerpath)
    reservation = 2 * 1024**3
    entries = sum(e['output_cap_bytes'] for e in ledger['reservations']['live_entries'])
    snapshots = sum(e['reserved_bytes'] for e in ledger['reservations']['source_snapshots'])
    assert ledger['stage_allocation_history'][-1] == pin(stagepath)
    total = entries + snapshots + reservation + 2 * 1024**3
    assert total <= ledger['caps']['output_peak_and_reservations_bytes']
    assert total <= stage['cumulative_caps']['output_peak_and_reservations_bytes']
    assert shutil.disk_usage(ROOT).free > 30 * 1024**3 + reservation
    record = {'path': str(snapshot), 'reserved_bytes': reservation, 'status': 'RESERVED_BEFORE_CLONE',
              'source_commit': commit, 'config_sha256': sha(configpath)}
    ledger['reservations']['source_snapshots'].append(record)
    save(ledgerpath, ledger)
    subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout', '--local', str(ROOT), str(snapshot)], check=True)
    subprocess.run(['git', '-C', str(snapshot), '-c', 'core.hooksPath=/dev/null', 'checkout', '--detach', commit], check=True)
    (snapshot / 'training_workloads/unified_h1/candidate_config.json').write_bytes(configpath.read_bytes())
    tree = subprocess.check_output(['git', '-C', str(snapshot), 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    from training_workloads.unified_h1.b0_handoff import _candidate_diff
    overlay = hashlib.sha256(_candidate_diff(snapshot, commit)).hexdigest()
    # This is a stdlib-only config/source verifier, never an environment import.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(snapshot) + ':' + str(snapshot / 'isaaclab-training'))
    result = subprocess.run([config['execution']['python_path'], '-B', '-c',
        'import json; from training_workloads.unified_h1.config import load_config; '
        'print(json.dumps(load_config()["resolved_identity"], sort_keys=True))'],
        cwd=snapshot, env=env, capture_output=True, text=True, timeout=60, check=True)
    identity = json.loads(result.stdout)
    assert identity == {kind + '_' + key: value for kind in ['product', 'workload']
                        for key, value in [('commit', commit), ('tree', tree)]} | {'config_sha256': sha(configpath)}
    size = sum(p.stat().st_size for p in snapshot.rglob('*') if p.is_file())
    assert size <= reservation
    for path in sorted(snapshot.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)
    snapshot.chmod(snapshot.stat().st_mode & ~0o222)
    record.update(status='FROZEN_READ_ONLY', observed_file_bytes=size)
    # Re-read because the original GPU run may have been observed/settled by the
    # sole writer between separate commands. No concurrent writer is allowed.
    ledger = read(ledgerpath)
    matches = [r for r in ledger['reservations']['source_snapshots'] if r['path'] == str(snapshot)]
    assert len(matches) == 1
    matches[0].update(record)
    save(ledgerpath, ledger)
    output = configpath.parent / 'readonly-snapshot.json'
    assert not output.exists()
    save(output, {'status': 'CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY', 'snapshot': str(snapshot),
                  'snapshot_commit': commit, 'snapshot_tree': tree, 'config_sha256': sha(configpath),
                  'snapshot_diff_sha256': overlay, 'read_only': True, 'source_identity': identity,
                  'snapshot_observed_bytes': size, 'snapshot_reserved_bytes': reservation,
                  'config_stdout': result.stdout, 'config_stderr': result.stderr,
                  'GPU_queries': 0, 'live_starts': 0, 'training_updates': 0})
    print(json.dumps({'source_commit': commit, 'source_identity': identity, 'output': str(output)}))


if __name__ == '__main__':
    main()
