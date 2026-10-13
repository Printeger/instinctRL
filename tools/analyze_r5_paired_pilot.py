"""CPU analysis of actual R5 pilot evidence; never performs optimizer updates.

Process, counter completeness and scientific data coverage are reported
separately. Original Z thresholds and all32 lanes are retained.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import torch
from training_workloads.unified_h1.evaluation import replay_raw_case, evaluate_episode, case_outcome

SOURCE = 'ef68ebb5682b85aa83044fdea1c13b7d0edbf325'
CHECKPOINTS = ('init0000', 'u1', 'u2', 'u8', 'u16', 'u32', 'u64')

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text()) if path.is_file() else None

def jsonl(path):
    if not path.is_file():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]

def stats(values):
    value = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if value.numel() == 0:
        return {'status': 'UNAVAILABLE', 'reason': 'NO_ACTUAL_SAMPLES'}
    if not bool(torch.isfinite(value).all()):
        return {'status': 'UNAVAILABLE', 'reason': 'NONFINITE_ACTUAL_SAMPLES', 'count': value.numel()}
    return {'status': 'AVAILABLE', 'count': value.numel(), 'mean': float(value.mean()),
            'std': float(value.std(unbiased=False)), 'min': float(value.min()), 'max': float(value.max()),
            'p95': float(torch.quantile(value, .95)), 'rms': float(value.square().mean().sqrt())}

def vectors(value):
    tensor = torch.as_tensor(value).detach().cpu()
    assert tensor.ndim == 2 and tensor.shape[1] == 3
    return {'units': 'm/s', 'frame': 'body', 'xyz': [stats(tensor[:, i]) for i in range(3)],
            'horizontal_norm': stats(tensor[:, :2].norm(dim=-1)), 'norm': stats(tensor.norm(dim=-1))}

def training(attempt, group):
    output = attempt / 'results/pilot-training'
    status = read(output / 'status.json')
    gate = read(output / 'pilot_diagnostic/first_update_gate.json')
    failure = read(output / 'pilot_diagnostic/first_update_failure.json')
    updates = jsonl(output / 'learning_log/updates.jsonl')
    minibatches = jsonl(output / 'learning_log/minibatches.jsonl')
    checkpoint_rows = []
    for name in CHECKPOINTS:
        path = output / 'checkpoints' / (name + '.pt')
        if not path.is_file():
            checkpoint_rows.append({'label': name, 'status': 'MISSING'})
            continue
        payload = torch.load(path, map_location='cpu', weights_only=True)
        expected_update = 0 if name == 'init0000' else int(name[1:])
        assert payload['counters']['update'] == expected_update
        assert payload['counters']['transition'] == expected_update * 512
        assert payload['counters']['global_policy_step'] == expected_update * 16
        assert payload['counters']['actor_optimizer'] == payload['counters']['critic_optimizer'] == expected_update * 8
        assert payload['source_identity']['product_commit'] == payload['source_identity']['workload_commit'] == SOURCE
        assert payload['training_seed'] == 20260908
        assert payload['ppo_config']['actor_learning_rate'] == (3e-4 if group == 'P0' else 1e-4)
        assert payload['ppo_config']['critic_learning_rate'] == 3e-4
        assert payload['curriculum']['completed_policy_steps'] == expected_update * 16
        checkpoint_rows.append({'label': name, 'status': 'VERIFIED_ACTUAL', 'path': str(path),
                                'sha256': digest(path), 'bytes': path.stat().st_size,
                                'counters': payload['counters'], 'source_identity': payload['source_identity'],
                                'network_config': payload['network_config'], 'ppo_config': payload['ppo_config']})
        del payload
    probe_path = output / 'pilot_diagnostic/first_update_probe.pt'
    probes = []
    if probe_path.is_file():
        probe = torch.load(probe_path, map_location='cpu', weights_only=True)
        assert probe['sample_count'] == 512 and probe['correction_heads'] == [1, 2, 3]
        assert probe['correction_units'] == 'm/s'
        for ordinal, row in enumerate(probe['minibatches'], 1):
            assert torch.equal(row['indices'], torch.arange(512))
            pre, post = row, row['after']
            probes.append({'minibatch': ordinal,
                'analytic_beta_kl_pre': stats(pre['analytic_beta_kl']),
                'analytic_beta_kl_post': stats(post['analytic_beta_kl']),
                'legacy_ppo_approximate_kl_pre': stats(pre['legacy_ppo_approximate_kl']),
                'legacy_ppo_approximate_kl_post': stats(post['legacy_ppo_approximate_kl']),
                'ratio_pre': stats(pre['ratio']), 'ratio_post': stats(post['ratio']),
                'issued_v_corr_b': vectors(pre['issued_v_corr_b']),
                'mean_v_corr_b_pre': vectors(pre['mean_v_corr_b']), 'mean_v_corr_b_post': vectors(post['mean_v_corr_b']),
                'action_mean_pre': [stats(pre['action_mean'][:, i]) for i in range(4)],
                'action_mean_post': [stats(post['action_mean'][:, i]) for i in range(4)],
                'parameter_delta_l2_pre': pre['parameter_delta_before'], 'parameter_delta_l2_post': pre['parameter_delta_after'],
                'unclipped_gradient_norm': pre['unclipped_gradient_norm'], 'clipped_gradient_norm': pre['clipped_gradient_norm']})
        assert len(probes) == 8
        del probe
    per_update = []
    for row in updates:
        update = row['completed_updates']
        records = [record for record in minibatches if record['update'] == update]
        assert len(records) == 8
        assert row['actor_optimizer_steps'] == row['critic_optimizer_steps'] == update * 8
        per_update.append({'update': update, 'minibatches': len(records),
                           'approximate_kl': stats([record['approximate_kl'] for record in records]),
                           'actor_pre_clip_gradient_norm': stats([record['actor_pre_clip_gradient_norm'] for record in records]),
                           'statistics': row['statistics'], 'coverage': row['coverage']})
    # Actual executed corrections from all returned training steps, rather
    # than substituting action-space sample-minus-mean for a physical vector.
    executed = {}
    coordinates = set()
    compact_path = output / 'learning_log/compact_rollouts.jsonl'
    if compact_path.is_file():
        with compact_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                coordinate = (row['update'], row['step'])
                assert coordinate not in coordinates
                coordinates.add(coordinate)
                assert 1 <= coordinate[0] <= 64 and 0 <= coordinate[1] < 16
                pre = row['pre_reset']; control = pre['control']
                sample = executed.setdefault(row['update'], {'correction': [], 'final_command': [],
                                                            'velocity': [], 'alpha': [], 'ics_beta': [], 'resets': 0})
                for target, value in [('correction', control['v_corr_b']), ('final_command', control['v_final_b']),
                                      ('velocity', pre['task']['actual_velocity_b'])]:
                    assert torch.as_tensor(value).shape == (32, 3)
                    sample[target].extend(value)
                assert torch.as_tensor(row['command_b']).shape == (32, 3)
                assert not bool(torch.as_tensor(row['command_b']).ne(0).any()), 'original1024-stepzero warmup changed'
                sample['alpha'].extend(torch.as_tensor(control['alpha']).reshape(-1).tolist())
                sample['ics_beta'].extend(torch.as_tensor(control['ics_metrics']['beta']).reshape(-1).tolist())
                sample['resets'] += int(torch.as_tensor(row['terminated']).logical_or(torch.as_tensor(row['truncated'])).sum())
    executed_rows = [{'update': update, 'actual_step_lane_samples': len(row['correction']),
                      'issued_v_corr_b': vectors(row['correction']), 'executed_v_final_b': vectors(row['final_command']),
                      'actual_velocity_b': vectors(row['velocity']), 'alpha': stats(row['alpha']),
                      'ics_beta': stats(row['ics_beta']), 'reset_lanes': row['resets']}
                     for update, row in sorted(executed.items())]
    complete = (len(updates) == 64 and [row['completed_updates'] for row in updates] == list(range(1, 65))
                and all(row['status'] == 'VERIFIED_ACTUAL' for row in checkpoint_rows)
                and bool(gate and gate['status'] == 'PASS' and gate['sample_count'] == 512)
                and len(probes) == 8)
    if complete:
        assert len(minibatches) == 512
        assert len(coordinates) == 1024 and len(executed_rows) == 64
        assert all(row['actual_step_lane_samples'] == 512 for row in executed_rows)
    return {'group': group, 'mode': 'pilot-training', 'source_commit': SOURCE,
            'actual_raw_status': status, 'first_real_rollout_gate': gate, 'diagnostic_failure': failure,
            'scientific_training_coverage_complete': complete, 'completed_update_records': len(updates),
            'completed_minibatch_records': len(minibatches), 'checkpoints': checkpoint_rows,
            'actual_first_update_probe': probes, 'per_update': per_update,
            'executed_training_corrections': executed_rows,
            'learning_failure_records': jsonl(output / 'learning_log/failures.jsonl'),
            'episodes': jsonl(output / 'learning_log/episodes.jsonl'),
            'partial_episode_records': jsonl(output / 'learning_log/partial_episodes.jsonl'),
            'partial_rollout_boundaries': jsonl(output / 'learning_log/partial_rollout_boundaries.jsonl'),
            'process_acceptance': 'NOT_GRANTED_BY_THIS_CPU_DATA_ANALYSIS',
            'gate_scope': 'Actual production beforebothoptimizers gate and first-update recorded probe. No CPU gate replaces it.'}

def evaluation(attempt, group):
    output = attempt / 'results/pilot-evaluation'
    cases = {}
    for update in (16, 64):
        raw = output / 'evaluation_raw' / f'pilot_{group}_u{update}' / 'Z.jsonl'
        if not raw.is_file():
            cases[str(update)] = {'status': 'MISSING_ACTUAL_RAW', 'raw_path': str(raw)}
            continue
        traces = replay_raw_case(str(raw), case='Z')
        assert len(traces) == 32 and [trace.env_index for trace in traces] == list(range(32))
        rows = [evaluate_episode(trace) for trace in traces]
        gate = case_outcome([bool(row['success']) for row in rows], ['d'+str(i % 8) for i in range(32)],
                            expected_subgroups=['d'+str(i) for i in range(8)])
        producer = read(output / f'pilot_{group}_u{update}_Z.json')
        normalized_rows = json.loads(json.dumps(rows))
        assert producer and producer['episodes'] == normalized_rows
        cases[str(update)] = {'status': 'ACTUAL_ORIGINAL_Z_REPLAYED', 'raw_path': str(raw), 'raw_sha256': digest(raw),
                              'episodes': rows, 'original_gate': gate,
                              'evidence_valid_episodes': sum(row['evidence_valid'] for row in rows)}
    return {'group': group, 'mode': 'pilot-evaluation', 'source_commit': SOURCE,
            'actual_raw_status': read(output / 'status.json'), 'u16_u64': cases,
            'coverage_complete': all(row['status'] == 'ACTUAL_ORIGINAL_Z_REPLAYED' for row in cases.values()),
            'all_original_gates_pass': all(row.get('original_gate', {}).get('pass', False) for row in cases.values()),
            'metric': 'Original metric-h1-v1 including sensorT0/startup/safety/resets/all32lanes/subgroups; no threshold edits.',
            'scope': 'Z development evaluations only; process acceptance is separate.'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['training', 'evaluation'], required=True)
    parser.add_argument('--group', choices=['P0', 'PLR'], required=True)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve existing analysis: ' + str(args.output))
    torch.set_num_threads(2)
    result = (training if args.mode == 'training' else evaluation)(args.attempt.resolve(), args.group)
    result.update(schema_version='instinctrl-r5-pair-actual-data-analysis/1',
                  analysis_tool_sha256=digest(Path(__file__)), CPU_only=True,
                  analysis_optimizer_updates=0, missing_samples_filled=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'group': args.group, 'mode': args.mode,
                      'coverage_complete': result.get('scientific_training_coverage_complete', result.get('coverage_complete')),
                      'output': str(args.output)}))

if __name__ == '__main__':
    main()
