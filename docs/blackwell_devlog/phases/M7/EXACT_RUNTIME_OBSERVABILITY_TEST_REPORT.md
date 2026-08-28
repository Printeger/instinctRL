# M7-005 Exact Runtime Observability Test Report

## Results

| Check | Result | Evidence |
| --- | --- | --- |
| Static scope/isolation audit | PASS | Exact allowlist; unchanged M7-003 core; no private world-ray/cache access in environment or runner; no actor/control/task/training source reference |
| Focused source/runtime/environment tests | PASS — 148 tests | Ordered source, ownership, adapter guard, exact mapping, terminal binding, aliases, stale/duplicate/skipped/reordered/step/reset/post-reset/mutation negatives |
| Relevant accepted M3-M7 pure regression, run 1 | PASS — 424 tests and 33 subtests | No accepted contract regression |
| Relevant accepted M3-M7 pure regression, run 2 | PASS — 424 tests and 33 subtests | Repeat result identical in count/status |
| M2 static | PASS — 10 tests | Platform static contract retained |
| Live process 1 | PASS | 2 envs, 8 policy steps, 16 transitions, 160 physics steps, 339,840 rays, one selective timeout |
| Live process 2 | PASS | Independent process with identical coverage |
| Live inspectors | PASS | Rank set `{1}`; exact sigma range `[0,0]`; valid/EXACT rows; hit max error `0`; distance max error `7.62939453125e-06 m` |
| Complete-tree comparator | PASS | 73 tensors, 1,022,282 elements, maximum absolute error `0` |
| Comparator mutations | PASS — 22 expected rejections | Every 15-detail field, three canonical observability fields, missing, extra, reordered, and nonfinite trees rejected |
| Frozen/package/formal-source integrity | PASS | R580 driver and pinned Python/PyTorch/Isaac/TorchRL/TensorDict; frozen Git trees and pip freeze unchanged |

## Acceptance mapping

- Same-state and terminal/reset binding: **PASS**. Source steps are exactly `[[1,1],...,[8,8]]`; row four reset mask is `[true,false]` and binds the forced terminal timeout.
- Exact metric semantics: **PASS**. Static-plane rank-one rows have `sigma_min=0`, validity true, and source `EXACT`.
- Detailed fields and finiteness: **PASS**. All declared serialized floating tensors are finite; raw infinities are absent from the complete output tree.
- Actor isolation: **PASS**. Static recursive scans and live before/after policy equality pass.
- Coverage and determinism: **PASS**. Both processes cover 339,840 rays and compare at error zero.

No training, sampling, gradients, optimizer, checkpoint, proxy, plot, ablation, or later-phase work was run.
