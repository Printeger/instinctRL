# M6-003 PPO Tensor-Core Plan

## Goal

Add only the pure-PyTorch actor, separate critic, Beta distribution, GAE,
bounded PPO-Clip update, and strict checkpoint core required by Order M6-003.
The accepted environment is a read-only regression input and is not connected
to the new package.

## Sources

- Accepted product base: `2d414e744b116dd6e4d52a34b297306376ed8bf0`.
- Exact M4 policy contract: `lidar_grid [N,12,59,360]`, `state_vec [N,52]`.
- Exact M6 critic contract: one privileged float32 `[N,18]` branch.
- Formal read-only algorithm evidence at `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`:
  `ppo.py` SHA256 `b47777f6cdcb6b054f64ffcec6c0bf209da9d69f6de6d9717fc8c72d3054f3fe`
  and `utils.py` SHA256 `ac86be73f2a75dd6b66641747f1766dc0503403ca4d81db1c714ba7a2f15b04f`.

## Steps

1. Implement immutable validated configs and exact actor/critic interfaces.
2. Implement bounded Beta evaluation, hand-checkable GAE and PPO losses,
   deterministic full-coverage minibatching, finite/gradient/optimizer checks,
   and separate Adam optimizers.
3. Implement atomic versioned restricted-load checkpoints.
4. Add CPU/static/negative tests and a two-process CUDA synthetic smoke.
5. Replay accepted regression and integrity gates, then create exactly two
   commits and stop for Commander review.

## Exclusions

No collector, environment step/import from the PPO core, replay, simulator
training, evaluation, curriculum, randomization, recurrent actor, cost critic,
PPO-Lagrangian, CTBR, M7, 128k, 1M, tuning, merge, or push.
