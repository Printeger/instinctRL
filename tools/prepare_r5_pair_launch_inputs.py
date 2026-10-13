"""Bind the accepted R5 pilot proposal to a fresh paired experiment.

This writes reviewable project inputs only. It does not write actual attempt
authorizations, approve a proposal, probe a GPU, start a simulator or train.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / 'docs/diagnostics/r5-readiness'
OUTPUT = ROOT / 'docs/experiments/r5-paired-pilot-20261013'
SOURCE = 'ef68ebb5682b85aa83044fdea1c13b7d0edbf325'
STAGE = 'INSTINCTRL-R5-PAIRED-PILOT-20261013'
FUTURE_SHA = '0154b4b2c9a693c99bf590e91256e092a22ba8fb07599dd66ca71d564b03113a'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def main():
    original = READINESS / 'pilot_future_proposals.r3.json'
    assert digest(original) == FUTURE_SHA
    future = json.loads(original.read_text())
    ready = json.loads((READINESS / 'pilot_launch_manifest.json').read_text())
    assert ready['status'] == 'READY_FOR_PAIRED_PILOT_WITH_FIRST_ROLLOUT_GATE'
    review = ready['independent_final_review']
    assert digest(READINESS / review['path']) == review['sha256']
    snapshot = Path(future['baseline']['snapshot'])
    assert subprocess.check_output(['git', '-C', str(snapshot), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
    config_path = snapshot / 'training_workloads/unified_h1/candidate_config.json'
    assert digest(config_path) == ready['shared_baseline']['config_sha256']
    config = json.loads(config_path.read_text())
    source_checks = []
    for section in ('product', 'workload'):
        for name, expected in config[section]['source_sha256'].items():
            path = Path(config[section]['path']) / name
            assert not path.is_symlink() and digest(path) == expected
            source_checks.append({'path': str(path), 'sha256': expected})
    release = Path(future['baseline']['tools_release_path'])
    release_record = json.loads((release / 'release.json').read_text())
    assert release_record['id'] == ready['tools_release_content_id']
    for name, expected in release_record['files'].items():
        assert digest(release / name) == digest(ROOT / 'tools/rlflow' / name) == expected
    rows = []
    for previous in future['groups']:
        row = deepcopy(previous)
        group, mode = row['group'], row['mode']
        spec, auth = row['experiment_proposal'], row['authorization_proposal']
        old_attempt = auth['attempt']
        new_attempt = old_attempt.removesuffix('20261012') + '20261013'
        assert not Path(old_attempt).exists() and not Path(new_attempt).exists()
        spec['id'] = spec['id'].removesuffix('20261012') + '20261013'
        spec['stage_id'] = STAGE
        spec['argv'] = [value.replace(old_attempt, new_attempt) for value in spec['argv']]
        spec['h1']['attempt'] = new_attempt
        spec['pins']['authorization']['path'] = str(Path(new_attempt) / 'commands/AUTHORIZATION.json')
        auth['attempt'] = new_attempt
        row['output'] = str(Path(new_attempt) / 'results' / mode)
        row['authorization_issued'] = False
        if mode == 'pilot-training':
            expected_auth = dict(auth, status='AUTHORIZED')
            filename = f'inputs/{group}-training-authorization.proposed.json'
            write(OUTPUT / filename, expected_auth)
            spec['pins']['authorization']['sha256'] = digest(OUTPUT / filename)
            row['expected_authorization'] = {'path': filename, 'sha256': spec['pins']['authorization']['sha256'],
                                             'status': 'PROPOSED_BYTES_ONLY_NOT_ISSUED'}
            row['missing_bindings'] = ['current independent pre-live review predicate', 'exact Commander approval',
                                       'actual AUTH bytes issued only after independent acceptance']
        else:
            for checkpoint in auth['parameters']['checkpoints']:
                checkpoint['path'] = checkpoint['path'].replace('20261012', '20261013')
            row['missing_bindings'] = ['accepted actual u16/u64 checkpoints and source identity',
                                       'independent actual-training evidence predicate',
                                       'exact Commander evaluation approval and actual AUTH bytes']
        spec['conditions'] = [{'evidence_id': 'E-R5-ENTRY6-FINAL-INDEPENDENT-READINESS', 'sha256': review['sha256']}]
        row['postrun_nvidia_smi_calls_maximum'] = 2
        row['postrun_query_boundary'] = 'After terminal receipt and independently observed owned quiescence; two audit calls total, not two sample-provider pairs.'
        row['evidence_collection'] = 'Existing generic H1 collector plus complete original job and entire attempt/results inventory. No specialized B0 seal claimed.'
        rows.append(row)
    packet = {'schema_version': 'instinctrl-r5-paired-pilot-launch-inputs/1',
              'status': 'UNISSUED_PENDING_INDEPENDENT_PRELIVE_AND_COMMANDER',
              'stage_id': STAGE, 'started_from_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'user_authority': 'New active Goal requests both64-update groups and u16/u64 original Z evaluations after new Commander authorization.',
              'baseline': ready['shared_baseline'], 'tools_release_content_id': ready['tools_release_content_id'],
              'frozen_proposal': {'path': str(original), 'sha256': FUTURE_SHA},
              'accepted_readiness_review': {'path': str(READINESS / review['path']), 'sha256': review['sha256']},
              'first_rollout_gate': ready['first_rollout_gate'], 'single_variable': 'actor LR3e-4 versus1e-4',
              'future_totals': future['future_total'], 'entries': rows,
              'no_automatic_retries': True, 'max_training_update_per_group': 64,
              'unchanged': ['reward', 'network', 'controller', 'curriculum', 'algorithm', 'original metric-h1-v1 thresholds'],
              'known_process_limit': 'Native app.close may time out at9.5sec. FAILED process is preserved separately from data/scientific acceptance; INCOMPLETE counters retain full reservation.',
              'evaluation_release_policy': 'Only after independently accepted actual training data, gate and checkpoint source/hash bindings. Training native close failure is disclosed and independently judged; never relabel process PASS.',
              'stop_paths': ready['stop_paths'], 'generator_sha256': digest(Path(__file__))}
    write(OUTPUT / 'launch_inputs.proposed.json', packet)
    write(OUTPUT / 'preparation-verification.json',
          {'schema_version': 'instinctrl-r5-pair-input-verification/1', 'result': 'PASS_STATIC_ONLY',
           'source_checks': source_checks, 'tool_release_files_checked': len(release_record['files']),
           'snapshot_commit': SOURCE, 'config_sha256': digest(config_path), 'frozen_proposal_sha256': FUTURE_SHA,
           'fresh_entries': [row['experiment_proposal']['id'] for row in rows],
           'available_disk_bytes': shutil.disk_usage(ROOT).free,
           'actual_authorizations_written': 0, 'GPU_queries': 0, 'simulation_starts': 0, 'optimizer_updates': 0})
    print(json.dumps({'status': packet['status'], 'output': str(OUTPUT), 'entries': len(rows), 'source_checks': len(source_checks)}))

if __name__ == '__main__':
    main()
