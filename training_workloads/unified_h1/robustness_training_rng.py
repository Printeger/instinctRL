"""Explicit RNG fork after a verified u640 restore; no training authorization.

The caller must verify/load the frozen checkpoint, persist the returned tensor
record before rollout, and register the changed permutation seed in the method
and checkpoint configuration. This is not strict continuation of the old RNG.
"""
from dataclasses import asdict, replace
import random

import numpy as np
import torch

from .driver import optimizer_steps
from .robustness_contract import BASELINE_SHA


def branch_from_u640(trainer, curriculum_generator, *, seed, origin_sha256, metadata):
    from instinctlab.training.ppo import PPOConfig
    from instinctlab.training.checkpoint_h1 import _rng_payload
    if type(seed) is not int or not 0<=seed<2**31 or seed==20260908:
        raise ValueError('R6_BRANCH_SEED')
    if origin_sha256!=BASELINE_SHA or metadata.get('training_seed')!=20260908:
        raise ValueError('R6_BRANCH_ORIGIN')
    counters=metadata.get('counters',{})
    expected=dict(update=640,transition=640*512,global_policy_step=640*16,
                  actor_optimizer=640*8,critic_optimizer=640*8)
    if counters!=expected or any(type(counters[k]) is not int for k in expected):
        raise ValueError('R6_BRANCH_ORIGIN_COUNTERS')
    if (trainer.update_counter!=640 or optimizer_steps(trainer.actor_optimizer)!=5120
            or optimizer_steps(trainer.critic_optimizer)!=5120):
        raise ValueError('R6_BRANCH_ACTUAL_COUNTERS')
    if getattr(trainer,'_r6_rng_branch_applied',False):
        raise ValueError('R6_BRANCH_ALREADY_APPLIED')
    if not isinstance(trainer.config,PPOConfig) or str(curriculum_generator.device)!='cpu':
        raise ValueError('R6_BRANCH_CONFIGURATION')
    device=next(trainer.actor.parameters()).device
    if next(trainer.critic.parameters()).device!=device or device.type not in ('cpu','cuda'):
        raise ValueError('R6_BRANCH_DEVICE')
    old_config=trainer.config
    new_config=replace(old_config,permutation_seed=seed)
    # CPU checks never probe CUDA. A live CUDA trainer is already initialized.
    before=dict(torch=torch.get_rng_state().clone(),python=random.getstate(),
                numpy=np.random.get_state(),curriculum=curriculum_generator.get_state().clone())
    if device.type=='cuda':before['cuda']=[s.clone() for s in torch.cuda.get_rng_state_all()]
    original_rng=_rng_payload(curriculum_generator,trainer)
    try:
        torch.random.default_generator.manual_seed(seed)
        if device.type=='cuda':torch.cuda.manual_seed_all(seed)
        random.seed(seed);np.random.seed(seed);curriculum_generator.manual_seed(seed)
        trainer.config=new_config
        branched_rng=_rng_payload(curriculum_generator,trainer)
    except BaseException:
        trainer.config=old_config
        torch.set_rng_state(before['torch']);random.setstate(before['python']);np.random.set_state(before['numpy'])
        curriculum_generator.set_state(before['curriculum'])
        if 'cuda' in before:torch.cuda.set_rng_state_all(before['cuda'])
        raise
    trainer._r6_rng_branch_applied=True
    return dict(schema_version='r6-training-rng-branch/1',origin_sha256=origin_sha256,
        origin_training_seed=20260908,branch_training_seed=seed,starting_update=640,
        origin_ppo_config=asdict(old_config),branch_ppo_config=asdict(new_config),
        origin_rng=original_rng,branch_rng=branched_rng,
        physical_continuity=False,strict_rng_continuation=False,
        semantics='COMMON_U640_NEW_CONTINUATION_STREAM; FRESH_PHYSICAL_RESET_REQUIRED')
