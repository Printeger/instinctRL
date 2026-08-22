# M7-001 Evaluation Plan

## Goal

Establish a deterministic pure-tensor evaluation contract at accepted base
`033514d8fa7d8d2bb2ece5faa1521d528e32b0a4`. Audit the exact M4 actor mapping
without modification, isolate evaluation privilege, aggregate every `[T,N]`
sample exactly once, and emit a finite canonical JSON record.

## Source of truth

1. M4 `audit_actor_observation` and exact `lidar_grid/state_vec` mapping.
2. M5 anchor outputs and ICS `beta`, `emergency`, `worst_margin_m_s` semantics.
3. M6 actual-velocity reward diagnostics and independent termination masks,
   union, truncation, and reason precedence.
4. Architecture lock, active CONTEXT, and locked Handbook metric list.

## Implementation sequence

1. Define immutable configuration and result records.
2. Validate the exact ordered two-branch public boundary and evaluation schema.
3. Compute per-step values/masks without reduction or repair.
4. Aggregate with explicit numerator, denominator, valid count, total count,
   and unavailable states.
5. Create actor audit/source/provenance/metric/termination canonical JSON.
6. Prove hand arithmetic, negatives, isolation, ownership, serialization, two-
   process determinism, regressions, and repository integrity.

## Scope boundary

No simulator, rollout collection, production evaluator, policy-quality test,
observability algorithm, B0-B8 implementation/execution, training, M8, scale
run, package change, merge, or push is part of this plan.
