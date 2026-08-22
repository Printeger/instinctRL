# M6 Pure-Tensor Task Contract Plan

Order: `M6-001`, revision 0

Attempt: `M6-001-D0001`

Base: `acda9b1b30ca8aceaf6cf6eaca9d2c1d4bdaaa73`

## Goal

Add strict pure-tensor reward, termination, and critic-only privileged schemas
after accepted M5, with exact actor/control non-interference. This increment
does not add a neural network, PPO, environment wiring, rollout, checkpoint,
evaluation, or training.

## Sequence

1. Implement required actual-velocity reward inputs and auditable components.
2. Implement collision/height/timeout masks, truncation and reason precedence.
3. Implement fixed-order privileged critic tensor with scalar-first root state.
4. Add analytic, negative, gradient, isolation and complete CUDA tests.
5. Regress and byte-check accepted M2-M5 surfaces.
6. Record source provenance, contract, evidence, limitations and final report.
7. Create exactly the two ordered commits and run final fail-closed evidence.

## Guaranteed exclusions

No accepted M1-M5 edit; no goal semantics, CTBR, CMDP, Lagrangian/cost critic,
GRU requirement, actor/critic network, optimizer, PPO, collector, replay,
rollout, checkpoint, environment/simulator integration, evaluation, training,
M7 work, package/runtime change, merge, or push.
