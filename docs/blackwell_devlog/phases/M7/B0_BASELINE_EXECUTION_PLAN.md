# M7-008 B0 Manifest Execution Plan

## Goal

Add one fail-closed evaluation-side boundary from an exact canonical B0 READY
manifest to the accepted M7-007 direct adapter. Return an immutable execution
envelope and bind B0 readiness to the exact new module bytes without runtime,
environment, controller, simulator, or scenario integration.

## Source locks

| Source | Accepted SHA-256 | Use |
| --- | --- | --- |
| `instinctlab/control/baseline_b0.py` | `5fa2b8d746eb231a1bb12f331fe1a76e8f988005e9a315e1dbf622e0bf1ed3d1` | Sole adapter invoked after manifest validation |
| `instinctlab/evaluation/baseline_execution.py` | `db7b73ee7e9c4b39689814530363bafb6a1ef48c2513a0208c0ff29c7073c108` | Fourth exact B0 readiness capability |
| M7-007 base | `fb9b9c4af7b0004c0b0647b80acef2e9e85e361b` | Accepted registry, manifest, and adapter contract |

## Writable files

Only the 12 product paths enumerated by Order M7-008 are writable. The first
commit contains implementation, exports, capability binding, tests, README,
plan, and contract. The second contains the four bounded outcome records.

## Implementation and tests

1. Reparse exact manifest canonical bytes and validate exact B0 identity,
   schemas, toggles, four ordered bindings, PASS audit, READY state, and no skip.
2. Validate exact tensor/config types, invoke only the accepted adapter once,
   and build the EXECUTED successor with the accepted manifest builder.
3. Prove only execution status and manifest hash change; preserve adapter
   parity, ownership, dtype/device, determinism, isolation, and fail atomicity.
4. Run focused suites twice, accepted pure regressions twice, independent-
   process comparison, exact source/import audits, and repository integrity.

## Stop conditions

Stop if work requires an unlisted path, changed accepted adapter/manifest
semantics, actor/research changes, packages or frozen inputs, runtime or
controller integration, CUDA/simulation, baseline execution, training, M8,
scale work, merge, push, or history rewrite.
