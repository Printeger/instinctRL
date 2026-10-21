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


def prepare_plan_rng(trainer,curriculum_generator,*,plan,metadata):
    """Bind one restored trainer to a validated plan before any rollout.

    The caller must durably save the returned tensor record before proceeding.
    Resuming an R6 trajectory preserves every restored RNG; only a common-u640
    start creates a new stream. Neither path executes an optimizer or reset.
    """
    from . import robustness_training_plan as contract
    from .checkpoint_compatibility import pinned_json
    from instinctlab.training.checkpoint_h1 import _rng_payload
    from instinctlab.training.ppo import PPOConfig
    binding=contract.verify_loaded(plan,metadata)
    if getattr(trainer,'_r6_plan_rng_prepared',False):raise ValueError('R6_PLAN_RNG_ALREADY_PREPARED')
    if not isinstance(trainer.config,PPOConfig) or str(curriculum_generator.device)!='cpu':
        raise ValueError('R6_PLAN_CONFIGURATION')
    recipe=pinned_json(plan['recipe']);method=pinned_json(recipe['method'])
    if (trainer.update_counter!=plan['start_update']
            or optimizer_steps(trainer.actor_optimizer)!=plan['start_update']*8
            or optimizer_steps(trainer.critic_optimizer)!=plan['start_update']*8):
        raise ValueError('R6_PLAN_ACTUAL_COUNTERS')
    if any(getattr(trainer.config,k)!=method['ppo'][k] for k in ('minibatch_size','epoch_count')):
        raise ValueError('R6_PLAN_PPO_CLOCK')
    if binding['branch_new_rng']:
        record=branch_from_u640(trainer,curriculum_generator,seed=plan['training_seed'],
            origin_sha256=plan['initialization']['checkpoint']['sha256'],metadata=metadata)
    else:
        if trainer.config.permutation_seed!=plan['training_seed']:
            raise ValueError('R6_PLAN_RESUME_PERMUTATION')
        record=dict(schema_version='r6-training-rng-resume/1',training_seed=plan['training_seed'],
            starting_update=plan['start_update'],restored_rng=_rng_payload(curriculum_generator,trainer),
            strict_rng_continuation=True,physical_continuity=False)
    trainer._r6_plan_rng_prepared=True
    return dict(plan_sha256=contract.object_sha(plan),recipe=plan['recipe'],binding=binding,rng=record,
                evidence_must_be_persisted_before_rollout=True)
