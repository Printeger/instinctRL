# M7-011 Exact B0 TASLAB Environment Final Report

## Scope completed

The accepted TASLAB direct environment has one exact factory-owned B0
command-only execution mode. It validates the requested configuration before
construction, revalidates the realized runtime configuration after
construction, owns exact `[N, 3]` commands before simulator mutation, advances
the existing Isaac Lab task lifecycle, invokes the accepted B0 run session
exactly once per physics step, and applies exactly one accepted Lee-controller
wrench. It does not invoke a second controller pipeline or expose simulator
state to the learned actor boundary.

The mode finalizes only at its exact positive horizon, emits the accepted
privacy-preserving lifecycle receipt plus scalar coverage, and clears command,
control, Lee and wrench caches. The B0 registry has the seventh exact
capability `m7.b0_taslab_environment`, bound to provider
`instinctlab.evaluation.baseline_environment.TaslabB0EvaluationEnv` and source
SHA-256
`b1ecdaa8b7d01ff0f9dd49891c2266908cea5a2540e8a91c33b9626464ded584`.
The preceding six capabilities and B1-B8 behavior remain unchanged.

D0003 repair commit `4eb7e067dcfe92ff1318679e8704c98728c035be`
deleted exactly one redundant root-level `oracle_diagnostics` payload key. It
did not change the summary diagnostics, per-physics-step diagnostics, or any
gate. The repaired runner SHA-256 is
`192d527b3bdb4e9f3e532bf0303f2de55f18ab175ab505333cfe66cace347d08`;
the exact comparator remained unchanged at
`1d5ab4ba37770b895968b53506efff2d71117b6c8460b88b13ad569220864158`.

## Correctness gates

| Gate | Result |
| --- | --- |
| exact factory ownership and post-construction configuration revalidation | PASS |
| exact command-only API and fail-closed configuration | PASS |
| one lifecycle and one wrench application per physics step | PASS |
| no duplicate pipeline and exact disabled control-core semantics | PASS |
| command ownership, actor privacy and failure-before-mutation | PASS |
| body/world and all five accepted Lee diagnostic identities | PASS: zero mismatches and zero maximum error |
| exact horizon, count coverage, receipt and cache clearing | PASS |
| two-process exact determinism and mutation-sensitive comparator | PASS |
| seven-binding readiness, stale-six rejection and B1-B8 preservation | PASS |
| M2-M7 regression, bounded CUDA regression and frozen integrity | PASS |

Accepted formal candidate `M7-011-D0003/formal_candidates/0001` recorded 1610
pytest passes, two explicitly authorized deselections, 41 command ledger
entries, two bounded headless CUDA environment processes, and exact equality
across 66 tensors / 45,496 elements. Its trajectory SHA-256 is
`97ba25dd50bfdaf89e6f5da1dfb58f23972f98dee7cfa9e7c146bfb0bfc65e96`.

## Boundary declaration

No collector, scenario matrix, new metric, performance ranking, sensor change,
reward, PPO, training, M8, scale, merge, push, frozen-runtime mutation, package
change, or Git configuration change was performed.

Recommendation: `READY_FOR_REVIEW`.

`NO NEXT-PHASE WORK WAS STARTED.`

`AWAITING COMMANDER REVIEW.`
