# M7-008 Revision 1 Import-Boundary Rework Final Report

## Finding F1 closure

Functional commit `666aafc1e3f2fa6c153d7c9ea2cda43e7f87de81`
restores a direct, exact import of `B0DirectConfig`, `B0DirectResult`, and
`execute_b0_direct_velocity` from `instinctlab.control.baseline_b0`. The
execution module does not import aggregate `instinctlab.control` or any other
control module.

The historical M7-007 test now permits exactly this one named evaluation source
and continues rejecting the direct B0 dependency in every other evaluation,
actor/observation, environment, task, and training module. The corrected module
SHA-256 is
`db7b73ee7e9c4b39689814530363bafb6a1ef48c2513a0208c0ff29c7073c108`;
the fourth capability and B0 READY/EXECUTED fixtures bind those exact bytes.

## Unchanged behavior

Manifest reparsing, fail-closed B0 READY validation, exact one adapter call,
READY-to-EXECUTED status/hash-only transition, tensor parity/ownership,
determinism, B1-B8 values/readiness, exports, and non-integration boundaries are
unchanged from D0001.

## Qualification

Fresh D0002 candidate 0001 is **PASS**: 50 expected command exits, 1,500 pytest
passes with zero skips, 17 pre-adapter rejects, two byte-identical independent-
process records, exact import/source/frozen/package integrity, D0001 evidence
immutability, full allowlist/history/diff hygiene, and clean state passed.

## Recommendation

`READY_FOR_REVIEW`

No environment/controller/runtime/scenario integration, baseline execution,
CUDA, simulator, training, M8, scale work, merge, push, or history rewrite was
performed.
