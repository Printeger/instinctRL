#!/usr/bin/env python3
"""Bounded two-environment M7 runtime evaluation with deterministic mean actions."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--output', type=Path, required=True)
PARSER.add_argument('--product-commit', required=True)
ARGS = PARSER.parse_args()
APP_LAUNCHER = AppLauncher(headless=True, enable_cameras=False)
SIMULATION_APP = APP_LAUNCHER.app

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instinctlab.envs import TaslabDirectEnv, TaslabDirectEnvCfg
from instinctlab.evaluation import RuntimeEvaluationCollector, RuntimeEvaluationConfig
from instinctlab.training.networks import HistoryActor

NUM_ENVS = 2
POLICY_STEPS = 8
PHYSICS_STEPS = 160
TIMEOUT_STEP = 4
SEED = 7002


def _tensor_hash(value: torch.Tensor) -> str:
    header = f'{value.dtype}|{tuple(value.shape)}|'.encode()
    payload = value.detach().contiguous().cpu().numpy().tobytes(order='C')
    return hashlib.sha256(header + payload).hexdigest()


def _actor_hash(actor: HistoryActor) -> str:
    digest = hashlib.sha256()
    for name, value in actor.state_dict().items():
        digest.update(name.encode())
        digest.update(_tensor_hash(value).encode())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _owned_policy(policy: OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    if tuple(policy) != ('lidar_grid', 'state_vec'):
        raise AssertionError('post-reset policy must retain exact actor-only key order')
    return OrderedDict((name, value.detach().clone()) for name, value in policy.items())


def _cpu_policy(policy: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value.detach().cpu().clone()) for name, value in policy.items())


def _field_hashes(policy: OrderedDict[str, torch.Tensor] | dict[str, torch.Tensor]) -> OrderedDict[str, str]:
    return OrderedDict((name, _tensor_hash(value)) for name, value in policy.items())


def main() -> None:
    if len(ARGS.product_commit) != 40:
        raise ValueError('product commit must be a full SHA')
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    cfg = TaslabDirectEnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    env = TaslabDirectEnv(cfg)
    observation, _ = env.reset(seed=SEED)
    actor = HistoryActor().to(env.device).eval()
    collector = RuntimeEvaluationCollector(
        observation['critic'].detach().clone(),
        RuntimeEvaluationConfig(num_envs=NUM_ENVS, expected_policy_steps=POLICY_STEPS),
    )
    actor_source = Path(__file__).resolve().parents[1] / 'instinctlab' / 'training' / 'networks.py'
    policy_field_hashes: list[OrderedDict[str, str]] = []
    concentration_hashes: list[OrderedDict[str, str]] = []
    timeout_evidence: OrderedDict[str, object] | None = None
    pending_timeout_policy: OrderedDict[str, torch.Tensor] | None = None
    for step in range(1, POLICY_STEPS + 1):
        if step == TIMEOUT_STEP:
            env.episode_length_buf[0] = env.max_episode_length - 1
        policy = observation['policy']
        audited = collector.prepare_policy(policy)
        if pending_timeout_policy is not None:
            if timeout_evidence is None or step != TIMEOUT_STEP + 1 or tuple(audited) != tuple(pending_timeout_policy):
                raise AssertionError('post-reset policy was not presented at the exact next actor call')
            binding = OrderedDict()
            for name in pending_timeout_policy:
                returned_value, next_value = pending_timeout_policy[name], audited[name]
                exact = (
                    returned_value.shape == next_value.shape
                    and returned_value.dtype == next_value.dtype
                    and returned_value.device == next_value.device
                    and torch.equal(returned_value, next_value)
                    and _tensor_hash(returned_value) == _tensor_hash(next_value)
                )
                if not exact:
                    raise AssertionError(f'post-reset {name} does not bind to the next audited actor mapping')
                binding[name] = OrderedDict((
                    ('shape', tuple(returned_value.shape)), ('dtype', str(returned_value.dtype)),
                    ('device', str(returned_value.device)), ('tensor_equal', True), ('hash_equal', True),
                ))
            timeout_evidence['next_audited_step'] = step
            timeout_evidence['next_audited_policy'] = _cpu_policy(_owned_policy(audited))
            timeout_evidence['next_audited_policy_field_hashes'] = _field_hashes(audited)
            timeout_evidence['post_reset_to_next_audited_binding'] = binding
            pending_timeout_policy = None
        policy_field_hashes.append(OrderedDict((
            ('lidar_grid', _tensor_hash(audited['lidar_grid'])),
            ('state_vec', _tensor_hash(audited['state_vec'])),
        )))
        with torch.no_grad():
            actor_output = actor(audited)
            action = actor_output.mean_action.detach().clone()
        concentration_hashes.append(OrderedDict((
            ('alpha', _tensor_hash(actor_output.alpha)), ('beta', _tensor_hash(actor_output.beta)),
        )))
        observation, reward, terminated, truncated, _ = env.step(action)
        pre_reset = env.extras['pre_reset']
        collector.record_transition(
            pre_action_policy=audited,
            action=action,
            pre_reset=pre_reset,
            returned_policy=observation['policy'],
            returned_critic=observation['critic'].detach().clone(),
            returned_episode_length=env.episode_length_buf.detach().clone(),
        )
        if step == TIMEOUT_STEP:
            if terminated.tolist() != [False, False] or truncated.tolist() != [True, False]:
                raise AssertionError('forced timeout did not produce the exact selective edge')
            pending_timeout_policy = _owned_policy(observation['policy'])
            timeout_evidence = OrderedDict((
                ('step', step),
                ('pre_reset_root_state_w', pre_reset['task']['root_state_w'][0].detach().cpu().clone()),
                ('post_reset_critic_root_state_w', observation['critic'][0, :13].detach().cpu().clone()),
                ('post_reset_policy', _cpu_policy(pending_timeout_policy)),
                ('post_reset_policy_field_hashes', _field_hashes(pending_timeout_policy)),
                ('pre_reset_episode_length', int(pre_reset['episode_length'][0].item())),
                ('post_reset_episode_length', int(env.episode_length_buf[0].item())),
                ('terminated', bool(terminated[0].item())), ('truncated', bool(truncated[0].item())),
            ))
    if timeout_evidence is None or pending_timeout_policy is not None:
        raise RuntimeError('timeout evidence or exact next-call policy binding was not captured')
    provenance = OrderedDict((
        ('product_commit', ARGS.product_commit), ('fixture_id', 'm7-runtime-two-env-eight-step-v1'),
        ('generator_id', 'run_m7_runtime_evaluation_smoke.py'), ('seed', SEED),
    ))
    evaluation = collector.finalize(provenance)
    output = OrderedDict((
        ('schema_version', 'm7.runtime-smoke.v1'),
        ('runtime_manifest', OrderedDict((
            ('num_envs', NUM_ENVS), ('policy_steps', POLICY_STEPS), ('physics_steps', PHYSICS_STEPS),
            ('physics_dt_s', 0.005), ('decimation', 20), ('policy_dt_s', 0.1),
            ('device', str(env.device)), ('headless', True), ('seed', SEED),
            ('timeout_step', TIMEOUT_STEP), ('actor_action', 'HistoryActor.forward.mean_action'),
            ('gradient_enabled_during_inference', False), ('training_performed', False),
            ('sampling_performed', False), ('checkpoint_loaded', False),
        ))),
        ('actor_manifest', OrderedDict((
            ('source_path', 'isaaclab-training/instinctlab/training/networks.py'),
            ('source_sha256', _file_hash(actor_source)), ('parameter_sha256', _actor_hash(actor)),
            ('parameter_manifest', actor.parameter_manifest()),
        ))),
        ('policy_field_hashes', tuple(policy_field_hashes)),
        ('concentration_hashes', tuple(concentration_hashes)),
        ('timeout_evidence', timeout_evidence),
        ('evaluation', evaluation),
        ('gates', OrderedDict((
            ('coverage_exact', len(policy_field_hashes) == POLICY_STEPS),
            ('physics_coverage_exact', POLICY_STEPS * cfg.decimation == PHYSICS_STEPS),
            ('single_selective_timeout', evaluation['gates']['reset_count'] == 1),
            ('all_finite', evaluation['gates']['all_finite']), ('all_gates_pass', True),
        ))),
    ))
    if not all(output['gates'].values()):
        raise AssertionError(f'runtime gates failed: {output["gates"]}')
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, ARGS.output)
    print({'status': 'PASS', 'policy_steps': POLICY_STEPS, 'physics_steps': PHYSICS_STEPS, 'reset_count': 1})
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
