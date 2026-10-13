"""Finite CPU attribution from the accepted real R5 pair; no optimizer calls.

Selected real rollout windows and retained inputs stay distinct from CPU
fixed-input checkpoint probes. Reset differencing never crosses a boundary.
"""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'docs/experiments/r5-paired-pilot-20261013'
OUT = ROOT / 'docs/experiments/r5-zero-correction-20261014'
UPDATES = (1, 2, 16, 64)
W = 0.05

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)

def stats(value):
    value = tensor(value).reshape(-1)
    assert value.numel() and bool(torch.isfinite(value).all())
    return {'n': value.numel(), 'signed_mean': float(value.mean()), 'std': float(value.std(unbiased=False)),
            'rms': float(value.square().mean().sqrt()), 'p05': float(torch.quantile(value, .05)),
            'p50': float(torch.quantile(value, .50)), 'p95': float(torch.quantile(value, .95)),
            'min': float(value.min()), 'max': float(value.max())}

def vector_stats(value):
    value = tensor(value).reshape(-1, 3)
    return {'xyz': [stats(value[:, axis]) for axis in range(3)],
            'horizontal_norm': stats(value[:, :2].norm(dim=-1)), 'norm': stats(value.norm(dim=-1))}

def corr(left, right):
    left, right = tensor(left).reshape(-1), tensor(right).reshape(-1)
    left, right = left - left.mean(), right - right.mean()
    denominator = left.norm() * right.norm()
    return float((left * right).sum() / denominator) if denominator > 0 else None

def group_windows(group, attempt):
    path = attempt / 'results/pilot-training/learning_log/compact_rollouts.jsonl'
    inventory = json.loads((OLD / f'live-{group}-training/INVENTORY.json').read_text())
    original = next(row for row in inventory['files'] if row['original'] == str(path))
    assert sha(path) == original['sha256'] and path.stat().st_size == original['bytes']
    selected = {update: [] for update in UPDATES}
    total_rows = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line); total_rows += 1
            if row['update'] in selected:
                selected[row['update']].append(row)
    assert total_rows == 1024
    result = {}
    for update, rows in selected.items():
        assert len(rows) == 16 and [row['step'] for row in rows] == list(range(16))
        stack = lambda fn: torch.stack([tensor(fn(row)) for row in rows])
        command = stack(lambda row: row['command_b'])
        sample = stack(lambda row: row['action_sample'])
        mean = stack(lambda row: row['action_mean'])
        issued = stack(lambda row: row['pre_reset']['control']['v_corr_b'])
        final = stack(lambda row: row['pre_reset']['control']['v_final_b'])
        final_world = stack(lambda row: row['pre_reset']['control']['v_final_w'])
        velocity = stack(lambda row: row['pre_reset']['task']['actual_velocity_b'])
        root = stack(lambda row: row['pre_reset']['root_state_w'])
        beta = stack(lambda row: row['pre_reset']['control']['ics_metrics']['beta'])
        anchor = [row['pre_reset']['control']['anchor_metrics'] for row in rows]
        zero = command.norm(dim=-1) <= 1e-3
        assert bool(zero.all()) and torch.allclose(issued, sample[:, :, 1:4] - .5, atol=6e-8, rtol=0)
        assert mean.shape == sample.shape == (16, 32, 4)
        terminated = torch.stack([torch.as_tensor(row['terminated'], dtype=torch.bool) for row in rows])
        truncated = torch.stack([torch.as_tensor(row['truncated'], dtype=torch.bool) for row in rows])
        reset = terminated | truncated
        components = {key: stack(lambda row, k=key: row['reward_components'][k]) for key in rows[0]['reward_components']}
        for name in ['reward_zero_extra', 'reward_governor_proxy', 'reward_final_proxy', 'reward_preservation']:
            assert not bool(components[name].ne(0).any())
        candidate_penalty = -W * zero * (issued.norm(dim=-1) / .1).square()
        segments = []
        for lane in range(32):
            start = 0
            for step in range(16):
                if bool(reset[step, lane]) or step == 15:
                    section = slice(start, step + 1)
                    displacement = root[step, lane, :3] - root[start, lane, :3]
                    segments.append({'lane': lane, 'first_step': start, 'last_step': step,
                                     'end_is_actual_reset': bool(reset[step, lane]),
                                     'samples': step + 1 - start,
                                     'mean_issued_v_corr_b': issued[section, lane].mean(dim=0).tolist(),
                                     'mean_actual_velocity_b': velocity[section, lane].mean(dim=0).tolist(),
                                     'mean_actual_velocity_w': root[section, lane, 7:10].mean(dim=0).tolist(),
                                     'world_displacement_from_first_observed_row_m': displacement.tolist(),
                                     'horizontal_displacement_from_first_observed_row_m': float(displacement[:2].norm()),
                                     'reference_limit': 'First returned row inside selected window; not sensor-T0, not across resets.'})
                    start = step + 1
        result[str(update)] = {
            'global_policy_step_range': [(update - 1) * 16, update * 16 - 1],
            'actual_lane_samples': 512, 'command_zero_samples': int(zero.sum()),
            'action_sample': [stats(sample[:, :, axis]) for axis in range(4)],
            'recorded_actor_mean': [stats(mean[:, :, axis]) for axis in range(4)],
            'issued_v_corr_b': vector_stats(issued), 'actor_mean_v_corr_b': vector_stats(mean[:, :, 1:4] - .5),
            'v_final_b': vector_stats(final), 'v_final_w': vector_stats(final_world),
            'actual_velocity_b': vector_stats(velocity), 'actual_velocity_w': vector_stats(root[:, :, 7:10]),
            'v_corr_vs_actual_velocity_b_rmse': vector_stats(issued - velocity),
            'same_step_control_then_response_correlation_xyz': [corr(issued[:, :, axis], velocity[:, :, axis]) for axis in range(3)],
            'ics_beta': stats(beta), 'reset_lanes': int(reset.sum()), 'segments': segments,
            'anchor_active': stats([row['active'] for row in anchor]),
            'anchor_usable_fraction': stats([row['usable_fraction'] for row in anchor]),
            'anchor_huber_loss': stats([row['huber_loss'] for row in anchor]),
            'reward_components': {name: stats(value) for name, value in components.items()},
            'preregistered_w005_extra_penalty_on_actual_actions': stats(candidate_penalty),
            'nonfailure_extra_penalty': stats(candidate_penalty[~reset.reshape(16, 32)]),
            'timing': 'Action/control staged before integration; velocity/root observed after returned policy step. dt0.1s; correlations are descriptive and include inertial lag.'}
    return {'raw_path': str(path), 'raw_sha256': original['sha256'], 'total_actual_rows': total_rows, 'windows': result}

def retained_gradient_evidence(group, attempt):
    output = attempt / 'results/pilot-training'
    probe_path = output / 'pilot_diagnostic/first_update_probe.pt'
    probe = torch.load(probe_path, map_location='cpu', weights_only=True)
    gradient_rows = []
    for ordinal, row in enumerate(probe['minibatches'], 1):
        # The final actor linear layer has eight output rows: alpha then beta.
        named = [(name, value) for name, value in row['unclipped_gradients'].items()
                 if name.startswith('trunk.') and value.ndim == 2 and value.shape[0] == 8]
        assert len(named) == 1
        name, gradient = named[0]
        gradient_rows.append({'minibatch': ordinal, 'recorded_gradient_parameter': name,
                              'raw_alpha_beta_head_gradient_l2': gradient.double().norm(dim=1).tolist(),
                              'actor_unclipped_gradient_norm': row['unclipped_gradient_norm'],
                              'actor_clipped_gradient_norm': row['clipped_gradient_norm'],
                              'parameter_delta_l2': row['parameter_delta_after'],
                              'actual_mean_v_corr_before_xyz': row['mean_v_corr_b'].double().mean(dim=0).tolist(),
                              'actual_mean_v_corr_after_xyz': row['after']['mean_v_corr_b'].double().mean(dim=0).tolist(),
                              'analytic_beta_kl_after': stats(row['after']['analytic_beta_kl'])})
    inputs = {}
    for update in (1, 2):
        path = output / f'pilot_diagnostic/update_{update}_full_input.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        alpha, beta = saved['old_alpha'].double(), saved['old_beta'].double()
        actions = saved['actions'].double().clamp(1e-6, 1 - 1e-6)
        advantage = saved['advantages'].double().reshape(512)
        returns = saved['returns'].double().reshape(512)
        whitened = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        score_a = actions.log() - torch.digamma(alpha) + torch.digamma(alpha + beta)
        score_b = torch.log1p(-actions) - torch.digamma(beta) + torch.digamma(alpha + beta)
        inputs[str(update)] = {'path': str(path), 'sha256': sha(path),
                              'actual_old_beta_mean_v_corr': vector_stats(alpha[:, 1:4] / (alpha[:, 1:4] + beta[:, 1:4]) - .5),
                              'actual_advantages': stats(advantage), 'actual_returns': stats(returns),
                              'advantage_vs_sampled_v_corr_correlation_xyz': [corr(advantage, actions[:, axis] - .5) for axis in (1, 2, 3)],
                              'weighted_Beta_score_alpha_heads': (whitened[:, None] * score_a).mean(dim=0).tolist(),
                              'weighted_Beta_score_beta_heads': (whitened[:, None] * score_b).mean(dim=0).tolist(),
                              'limit': 'Actual captured concentrations/actions/advantages; output-space score diagnostic, not reconstructed full minibatch optimizer or exact reward counterfactual.'}
        del saved
    return {'first_probe_path': str(probe_path), 'first_probe_sha256': sha(probe_path),
            'actual_first_update_gradients_and_migration': gradient_rows, 'actual_retained_inputs': inputs,
            'conditional_Beta_KL_u16_u64': 'Not recorded per actual sample. Existing per-minibatch approximateKL and actual action means remain available; no missing Beta state is invented.'}

def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'R5_ZERO_COMMAND_ATTRIBUTION.json'
    if target.exists():
        raise FileExistsError('preserve existing attribution')
    packet = json.loads((OLD / 'launch_inputs.proposed.json').read_text())
    groups = {}
    for group in ('P0', 'PLR'):
        row = next(row for row in packet['entries'] if row['group'] == group and row['mode'] == 'pilot-training')
        attempt = Path(row['authorization_proposal']['attempt'])
        groups[group] = {'actual_windows': group_windows(group, attempt),
                         'gradient_and_advantage_evidence': retained_gradient_evidence(group, attempt)}
    plr = groups['PLR']['actual_windows']['windows']['64']
    assert abs(plr['issued_v_corr_b']['xyz'][1]['signed_mean'] + .05559) < .001
    result = {'schema_version': 'instinctrl-r5-zero-command-attribution/1',
              'source_pair': str(OLD), 'old_source_commit': 'ef68ebb5682b85aa83044fdea1c13b7d0edbf325',
              'groups': groups, 'decision': 'SUPPORTED_PROXIMAL_RESIDUAL_BIAS_PROCEED_PREREGISTERED_SINGLE_VARIABLE',
              'candidate_weight_frozen_before_new_training': W,
              'formula': '-w * I[norm(command_b)<=1e-3] * (norm(v_corr_b)/0.1m_s)^2',
              'already_present_objectives': ['actual velocity', 'ray anchor', 'command smoothing', 'failure/safety penalties'],
              'gap': 'Direct residual correction proxy is disabled under zero-command g=0; zero-extra is exactly0 in accepted real windows.',
              'limits': ['No claim that all historical failures or forces are explained.',
                         'Sampled training and deterministic Z are different trajectories.',
                         'Correlations and fixed-window means are not closed-loop causal proof.',
                         'No exact unrecorded reward/optimizer counterfactual; missing per-sample Beta state at u16/u64 is explicit.'],
              'CPU_only': True, 'new_simulator_starts': 0, 'new_optimizer_updates': 0,
              'analysis_tool_sha256': sha(Path(__file__))}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'decision': result['decision'], 'weight': W,
                      'PLR_u64_vcorr_y': plr['issued_v_corr_b']['xyz'][1]['signed_mean'],
                      'PLR_u64_actual_v_y': plr['actual_velocity_b']['xyz'][1]['signed_mean'],
                      'PLR_u64_candidate_penalty_mean': plr['preregistered_w005_extra_penalty_on_actual_actions']['signed_mean']}))

if __name__ == '__main__':
    main()
