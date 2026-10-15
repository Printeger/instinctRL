"""CPU-only evidence checks for the already frozen formal 257..512 segment."""
from pathlib import Path
import argparse
import json
import torch
from tools.analyze_r5_formal_research import bound_entry
from tools.analyze_r5_paired_pilot import digest, read, jsonl
from training_workloads.unified_h1.evaluation import CASES, evaluate_checkpoint, replay_raw_case
from training_workloads.unified_h1.formal_segment import method_hashes

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/experiments/r5-formal-readiness-auto'
CASE = 'R5-FORMAL-MEAN10-FIRST-SEGMENT-01'
NODES = [272, 288, 320, 384, 512]


def verify_inventory(directory, group, mode):
    inventory = read(directory / 'live' / f'{group}-{mode}' / 'INVENTORY.json')
    expected = set()
    for root in inventory['roots'].values():
        expected.update(str(p) for p in Path(root).rglob('*') if p.is_file())
    assert expected == {r['original'] for r in inventory['files']}, 'incomplete inventory'
    for row in inventory['files']:
        p = Path(row['original'])
        assert not p.is_symlink() and p.stat().st_size == row['bytes'] and digest(p) == row['sha256']
        if row['archived']:
            assert digest(Path(row['archived'])) == row['sha256']
    return {'objects': len(expected), 'all_original_and_copy_hashes': 'PASS'}


def completion(raw, receipt, output, *, training):
    """Only the exact known-close exception may accompany complete science."""
    assert raw['primary_error'] is None and not raw['cleanup_errors']
    assert raw['status'] in ['PASS', 'PASS_PENDING_CLOSE']
    if receipt['status'] != 'SUCCEEDED':
        close = read(output / 'cleanup_timeout.json')
        assert receipt['status'] == 'FAILED' and receipt['exit_code'] == -9
        assert close['status'] == 'HARD_TIMEOUT' and close['cleanup'] == 'app.close'
        assert close['timeout_seconds'] == 9.5
    c = raw['counters']
    expected = {'training_transitions': 131072 if training else 0,
                'nontraining_transitions': 0 if training else 164800,
                'completed_updates': 256 if training else 0,
                'real_optimizer_updates': 256 if training else 0,
                'actor_optimizer_steps': 2048 if training else 0,
                'critic_optimizer_steps': 2048 if training else 0,
                'real_checkpoint_writes': 5 if training else 0}
    assert all(c[k] == v for k, v in expected.items()), 'incomplete formal execution'
    assert c['synthetic_samples'] == c['synthetic_optimizer_updates'] == c['temporary_checkpoint_writes'] == 0
    if training:
        gate = read(output / 'pilot_diagnostic/first_update_gate.json')
        assert gate['status'] == 'PASS' and gate['sample_count'] == 512 and gate['minibatches'] == 8
        assert gate['starting_update'] == 256 and gate['first_absolute_update'] == 257
        assert gate['float32_atol'] == 1e-6 and gate['float32_rtol'] == 1e-5
        assert gate['policy_rng_consumed'] is False
        assert not (output / 'pilot_diagnostic/first_update_failure.json').exists()
    return expected


def validate_update_logs(updates, mini, weight):
    assert [r['completed_updates'] for r in updates] == list(range(257, 513))
    assert [(r['update'], r['minibatch_ordinal']) for r in mini] == [(u, i) for u in range(257, 513) for i in range(8)]
    assert all(r['actor_optimizer_steps'] == r['critic_optimizer_steps'] == r['completed_updates']*8 for r in updates)
    assert all(isinstance(r['actor_auxiliary_loss'], (int, float)) and 0 <= r['actor_auxiliary_loss'] < float('inf') for r in mini)
    if weight == 0: assert all(r['actor_auxiliary_loss'] == 0 for r in mini)


def verify_probe(data, probe, mini, p, origin):
    from instinctlab.training.ppo import deterministic_minibatches
    assert probe['sample_count'] == 512 and probe['correction_heads'] == [1, 2, 3] and len(probe['minibatches']) == 8
    assert data['update'] == 257
    first = probe['minibatches'][0]
    for a, b in [('actions', 'action'), ('advantages', 'advantage'), ('returns', 'return')]:
        assert torch.equal(data[a], first[b])
    assert torch.allclose(data['old_log_probabilities'], first['log_probability'], atol=1e-6, rtol=1e-5)
    assert all(torch.equal(r['indices'], torch.arange(512)) for r in probe['minibatches'])
    indices = deterministic_minibatches(512, 64, 1, origin['ppo_config']['permutation_seed'] + 256)
    for i, ids in enumerate(indices):
        mean = probe['minibatches'][i]['action_mean'][ids]
        zero = data['policy']['state_vec'][ids, 45:48].norm(dim=-1) <= .001
        expected = p['mean_prior_weight']*(zero*((mean[:, 1:4]-.5)/.1).square().sum(-1)).mean()
        assert torch.isclose(expected, torch.tensor(mini[i]['actor_auxiliary_loss']), atol=1e-6, rtol=1e-5)


def training_analysis(directory, group):
    attempt, auth, config, binding = bound_entry(directory, group, 'research-formal-training')
    p = auth['parameters']; out = attempt / 'results/research-formal-training'
    receipt = read(ROOT / 'coordination/research/jobs' / binding['job_id'] / 'receipt.json')
    counts = completion(read(out / 'status.json'), receipt, out, training=True)
    inventory = verify_inventory(directory, group, 'research-formal-training')
    init = read(out / 'formal_initialization.json')
    assert init['origin'] == p['initialization_checkpoint'] and init['phase'] == p
    assert init['initial_global_policy_step'] == 4096 and init['restored_adam_steps_each'] == 2048
    assert init['physical_history_anchor_actuator_continuity_claimed'] is False
    aux = read(out/'actor_mean_prior_binding.json')
    assert aux['actual_weight'] == p['mean_prior_weight'] and aux['hook_installed'] and aux['actor_only']
    assert aux['mean_prior_contract'] == p['mean_prior_contract'] and not aux['runtime_action_replacement']
    checkpoints = []
    for update in NODES:
        path = out / 'checkpoints' / f'u{update}.pt'
        cp = torch.load(path, map_location='cpu', weights_only=True)
        assert cp['counters'] == {'update': update, 'transition': update*512,
            'global_policy_step': update*16, 'actor_optimizer': update*8, 'critic_optimizer': update*8}
        assert cp['source_identity'] == {k: auth[k] for k in ['product_commit', 'product_tree', 'workload_commit', 'workload_tree', 'config_sha256']} and cp['training_seed'] == p['training_seed']
        assert cp['method_hashes'] == method_hashes(config, p)
        assert cp['curriculum'] == {'version': 'curriculum-h1-v1', 'completed_policy_steps': update*16}
        assert cp['ppo_config']['actor_learning_rate'] == p['actor_lr']
        assert cp['ppo_config']['critic_learning_rate'] == p['critic_lr']
        for key in ['actor_optimizer', 'critic_optimizer']:
            assert {int(v['step']) for v in cp[key]['state'].values()} == {update*8}
        checkpoints.append({'group': group, 'update': update, 'path': str(path), 'sha256': digest(path)})
        from instinctlab.training.checkpoint_h1 import _finite_tree
        _finite_tree(cp, 'actual_formal_checkpoint')
        del cp
    updates = jsonl(out / 'learning_log/updates.jsonl')
    mini = jsonl(out / 'learning_log/minibatches.jsonl')
    validate_update_logs(updates, mini, p["mean_prior_weight"])
    from training_workloads.unified_h1.schedule import command_at
    coordinates = []; resets = 0
    for row in jsonl(out / 'learning_log/compact_rollouts.jsonl'):
        u, step = row['update'], row['step']; coordinates.append((u, step))
        cmd = torch.tensor(row['command_b'], dtype=torch.float32)
        assert torch.equal(cmd, command_at((u-1)*16+step)), 'mixed clock/command mismatch'
        corr = torch.tensor(row['pre_reset']['control']['v_corr_b'], dtype=torch.float64)
        actual = torch.tensor(row['reward_components']['reward_zero_extra'], dtype=torch.float64).reshape(32)
        expected = -.05 * (cmd.norm(dim=-1) <= .001) * (corr/.1).square().sum(-1)
        assert torch.isfinite(corr).all() and torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
        resets += int(torch.tensor(row['terminated']).logical_or(torch.tensor(row['truncated'])).sum())
    assert coordinates == [(u, s) for u in range(257, 513) for s in range(16)]
    # Saved outputs only: do not run an extra actor or sample on CPU.
    from instinctlab.training.ppo import deterministic_minibatches
    assert digest(Path(p['initialization_checkpoint']['path'])) == p['initialization_checkpoint']['sha256']
    origin = torch.load(p['initialization_checkpoint']['path'], map_location='cpu', weights_only=True)
    data = torch.load(out/'pilot_diagnostic/update_257_full_input.pt', map_location='cpu', weights_only=True)
    probe = torch.load(out/'pilot_diagnostic/first_update_probe.pt', map_location='cpu', weights_only=True)
    verify_probe(data, probe, mini, p, origin)
    from tools.analyze_r5_paired_pilot import stats, vectors
    diagnostics = [{'ordinal':i+1, 'analytic_KL_pre':stats(r['analytic_beta_kl']),
        'analytic_KL_post':stats(r['after']['analytic_beta_kl']),
        'mean_v_corr_b_pre':vectors(r['mean_v_corr_b']), 'mean_v_corr_b_post':vectors(r['after']['mean_v_corr_b'])}
        for i, r in enumerate(probe['minibatches'])]
    learning = {k:stats([r[k] for r in mini]) for k in ['actor_auxiliary_loss','approximate_kl','actor_pre_clip_gradient_norm']}
    return {'status': 'PASS_CPU_ACTUAL_TRAINING', 'binding': binding, 'coverage_complete': True,
            'inventory': inventory, 'additional_counters': counts, 'automatic_reset_lanes': resets,
            'accepted_checkpoint_candidates': checkpoints, 'first_absolute_update': 257, 'first_update_distribution_diagnostics':diagnostics,
            'all_minibatch_statistics':learning, 'learning_update_statistics':updates,
            'CPU_only': True, 'actor_forwards': 0, 'optimizer_calls': 0,
            'limits': 'Independent model must additionally inspect actual retained gate/probe correspondence before acceptance.'}


def evaluation_analysis(directory, group):
    attempt, auth, config, binding = bound_entry(directory, group, 'research-formal-evaluation')
    out = attempt / 'results/research-formal-evaluation'
    receipt = read(ROOT / 'coordination/research/jobs' / binding['job_id'] / 'receipt.json')
    counts = completion(read(out / 'status.json'), receipt, out, training=False)
    inventory = verify_inventory(directory, group, 'research-formal-evaluation')
    nodes = {}
    for ck in auth['parameters']['checkpoints']:
        assert digest(Path(ck['path'])) == ck['sha256']
        traces = []
        for case in CASES:
            raw = out/'evaluation_raw'/f'formal_{group}_u{ck["update"]}'/f'{case}.jsonl'
            rows = jsonl(raw)
            assert rows[0]['kind'] == 'T0' and rows[0]['seed'] == auth['parameters']['evaluation_seed']
            steps = [r for r in rows if r['kind'] == 'STEP']
            assert [r['step'] for r in steps] == list(range(CASES[case].steps))
            traces.extend(replay_raw_case(str(raw), case=case))
        result = json.loads(json.dumps(evaluate_checkpoint(traces, update=ck['update']), allow_nan=False))
        assert result == read(out/f'formal_{group}_u{ck["update"]}.json'), 'original score replay mismatch'
        nodes[str(ck['update'])] = result
    assert sorted(map(int, nodes)) == NODES
    return {'status': 'PASS_CPU_ORIGINAL_EVALUATION_REPLAY', 'binding': binding,
            'coverage_complete': True, 'inventory': inventory, 'nontraining_transitions': counts['nontraining_transitions'],
            'nodes': nodes, 'any_original_Z_regression': any(not n['cases']['Z']['pass'] for n in nodes.values()),
            'CPU_only': True, 'optimizer_calls': 0}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['training', 'evaluation'])
    parser.add_argument('group', choices=['R0', 'R1']); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    result = globals()[args.action + '_analysis'](BASE/'experiments'/CASE, args.group)
    from rlflow.store import atomic
    atomic(args.output, (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())
    print(json.dumps({'status': result['status'], 'path': str(args.output), 'sha256': digest(args.output)}))

if __name__ == '__main__': main()
