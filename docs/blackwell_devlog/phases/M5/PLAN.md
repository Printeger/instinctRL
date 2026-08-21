# M5 Pure-Tensor Control-Core Plan

Order: `M5-001`, revision 0

Attempt: `M5-001-D0001`

Base: `bff6f853eb6d95b12d4de7b95cbfa5a7f2283bbc`

## Goal

Add a deterministic pure-PyTorch control core over the exact audited M4 actor
boundary while isolating the controller quaternion to the final adapter.

## Source authorities

Accepted M3 ray geometry, accepted M4 history/audit, the architecture lock,
the four named Handbook sections, and the four frozen legacy modules/tests.
Legacy experimental gates and privileged safety paths are excluded.

## Expected changes

The new `instinctlab.control` package, one M5 semantic suite, CUDA smoke and
unchanged comparator, this phase's six records, and the bounded README update.

## Sequence

1. Strict governor and B0 decode over latest M4 state fields.
2. Stateful per-environment measurement anchor with atomic validation.
3. Exact mounted MID360 body rays and history-only ICS attenuation.
4. Strict scalar-first body-to-world adapter and atomic composed pipeline.
5. Positive, negative, analytic, source, non-interference, CUDA, and regression
   qualification followed by exact two-commit delivery.

## Tests and evidence

Run source/static checks first, then CPU unit tests, CUDA smoke/comparison, M4
and M3 regressions, and repository/evidence integrity. Preserve command metadata,
raw streams, hashes, full tensor outputs/caches, and final diff inventories.

## Risks and stop conditions

Fail closed for any M4 pair/domain violation, malformed action/quaternion,
state mismatch, or invalid reset. Stop for an external prerequisite, contract
contradiction, forbidden path, or required runtime/package/system change.

## Guaranteed exclusions

No actor network, controller runtime, simulator, articulation, reward, critic,
PPO, collector, checkpoint, rollout, evaluation, training, M6 work, M1-M4 edit,
package change, external-repository change, merge, or push.
