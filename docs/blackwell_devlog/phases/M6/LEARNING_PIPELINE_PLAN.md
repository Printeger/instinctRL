# M6-004 Bounded Learning-Pipeline Plan

## Goal

Connect the accepted two-environment `TaslabDirectEnv` to the accepted PPO
tensor core through one fail-closed rollout adapter, while preserving the hard
actor boundary and all accepted implementations unchanged.

## Product boundary

- Add `training.rollout` for immutable configuration, detached transition
  ownership, pure-timeout terminal reconstruction, GAE assembly, and explicit
  `[T,N]` flattening.
- Export the adapter without changing networks, PPO, checkpoint, task, or env.
- Add pure/fake tests and one bounded real-environment smoke.
- Keep all general collection, training CLI, evaluation, randomization,
  curriculum, scaling, and simulator-state resume out of scope.

## Fixed dimensions

Each independent process uses `N=2`, two `T=4` segments, eight policy steps,
160 physics steps, 16 samples, two PPO updates, and one checkpoint reload. The
first public step forces env0 to the accepted pure-timeout edge; the remaining
seven steps prove finite continuation.

## Gates

Static actor-call isolation precedes unit tests. Unit and hand-computed adapter
tests precede bounded runtime. Runtime trees must pass independent semantic
inspection and exact full-tree comparison. Accepted regressions, source bytes,
frozen heads, packages, history, paths, and diff hygiene close the delivery.
