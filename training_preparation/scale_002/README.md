# SCALE-002 CPU-only preparation

This package specifies and checks a proposal; it is not a training entry. The
recommended definition is a cumulative total of exactly 1,000,000 decimal
environment transitions. Starting from the accepted 128,000-transition
checkpoint therefore requires exactly 872,000 new transitions.

The lossless segment plan is 1,703 rollouts of `32 × 16 = 512` samples plus one
rollout of `32 × 2 = 64` samples. With minibatch size 64 and one epoch, these
produce 1,704 updates and 13,625 new steps for each optimizer. The restored
counter moves from 250 to 1,954 and each optimizer from step 2,000 to 15,625.
Every environment receives 27,250 policy steps and 545,000 policy-driven
physics steps at decimation 20.

The alternative meaning, one million *new* transitions, would finish at
1,128,000 cumulative transitions. It requires 1,953 full rollouts and one
64-sample tail, 1,954 new updates, counter 2,204, 15,625 new steps per
optimizer, 31,250 policy steps per environment and 625,000 physics steps per
environment. The proposal never labels 1,024,000 or 1,128,000 as cumulative
one million.

The recommended research change is explicit: replace the old plane-only,
zero-command distribution with 32 isolated copies of the EVAL-002-style corner
and a bounded two-phase command curriculum. This changes the training method
and input and needs user approval. The existing two-environment EVAL-002 rank
evidence does not prove visibility for the proposed 32-environment layout.
The successor entry must build the combined RayCaster mesh and matching wall
colliders, assert all actual `env_origins`, and establish affine rank 3 for
every initial hit cloud before the first training step.

The actor boundary remains the exact measured `lidar_grid` and `state_vec`.
Scene truth and privileged state remain outside the actor. Environment resets
reset task/history state for selected environments but do not restart the
global command clock. The checkpoint restores model, optimizers, counter and
CPU/CUDA RNG. It has no simulator/environment state, so the proposal starts a
fresh environment reset and makes no same-trajectory resume claim.

Run the declaration-only tests with the pinned interpreter:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B \
  tests/training_preparation/test_scale_002.py
```

Run the read-only identity, checkpoint, and public-interface audit with the
product package on `PYTHONPATH`:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/mint/rl_dev/worktrees/instinctRL-scale-002-prep:/home/mint/rl_dev/worktrees/instinctRL-blackwell/isaaclab-training \
  /home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B \
  -m training_preparation.scale_002.audit
```

Both commands are CPU-only. They do not import an AppLauncher entry, construct
an environment or trainer, call `env.step`, call `PPOTrainer.update`, perform
backpropagation, write a checkpoint, or start a simulator.
