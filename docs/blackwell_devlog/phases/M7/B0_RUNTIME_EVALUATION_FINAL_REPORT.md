# M7-012 Exact B0 Runtime Evaluation Final Report

## Scope completed

The direct-import-only `instinctlab.evaluation.baseline_runtime` collector now
connects the accepted command-only B0 environment transition to every accepted
ordered evaluation input, exact normal-free observability and the unchanged M7
step/aggregate metrics. It finalizes only against the exact M7-011 environment
receipt and emits one owned `m7.b0-runtime-evaluation.v1` record.

The command is the only actor-facing input. No learned actor/action was
fabricated, the accepted learned `RuntimeEvaluationCollector` and canonical
learned audit path were not changed, and evaluation-only root/velocity/control/
task/observability state cannot feed back into command generation or actuation.
Raw rays, normals, mesh objects, actor-private lidar/history and live simulator
aliases do not appear in the result. Active row, detail, command, identity and
station-reference caches are cleared at finalization.

## Exact gates

| Gate | Result |
| --- | --- |
| Factory/config/manifest ownership and direct import isolation | PASS |
| One-pending-command atomic state machine | PASS |
| Exact ordered `[8,2,...]` evaluation inputs | PASS |
| Normal-free unique current-step observability | PASS |
| Unchanged metric tensors, scalar values and counts | PASS |
| Exact READY/EXECUTED receipt and coverage binding | PASS |
| Privacy, no-alias and post-finalize cache clearing | PASS |
| Eighth capability; preceding seven and B1-B8 preserved | PASS |
| Two exact independent CUDA processes and complete comparator | PASS |
| Accepted learned/default runtime and environment regressions | PASS |
| Source/frozen/allowlist/history/diff/clean integrity | PASS |

Each new CUDA process proved exactly two environments, eight policy rows, 16
metric samples, 160 physics/lifecycle/wrench calls, 320 environment samples and
339,840 rays, with zero reset/done, actor and training calls, one receipt and
one canonical record. The complete comparator found zero difference across 41
tensors and 3,360 elements; all eight required mutation classes were rejected.

## Capability binding

`m7.b0_runtime_evaluation` binds
`instinctlab.evaluation.baseline_runtime.B0RuntimeEvaluationCollector` to
source SHA-256
`c3f1e6f73f48bdcbf995c6f8178d5b93be2d6107af00d81b0ceac69c4a358e66`.
The exact canonical fixture READY and EXECUTED manifest hashes are respectively
`52cb42ba22b7c69c11fca5cd742a38a145a689ff295ec2b7c48df920bafa5b58`
and `9315d3d1c056769c0f7dc4dcc03c41165865a0c20a8d52ab7deddbd51165670e`.

## Recommendation and stop declaration

`READY_FOR_REVIEW`

`NO NEXT-PHASE WORK WAS STARTED.`

`AWAITING COMMANDER REVIEW.`
