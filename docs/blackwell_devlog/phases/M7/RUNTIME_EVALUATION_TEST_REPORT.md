# Runtime Evaluation Test Report

Revision 1 formal candidate `M7-002-D0002/formal_candidates/0001`: **PASS**.
All 51 commands produced their expected exits.

| Check | Result |
| --- | --- |
| Revised runtime pure contract | PASS — 43 tests |
| Structured metadata | PASS — exact 23 fields × 9 ordered metadata keys, exact values/digest, malformed/reordered/mutated negatives, deep ownership |
| Runtime source negatives | PASS — extra/malformed/shape/dtype/device/nonfinite/domain/precedence/alias/stale/reset corruption fail closed |
| Post-reset policy binding | PASS — complete `lidar_grid` and `state_vec` retained and exactly bound to next audited actor-only call |
| M7-001 plus M3-M6 pure replay, twice | PASS — 242 tests and 33 subtests each |
| M2 static | PASS — 10 tests |
| Accepted M6 environment smoke | PASS — 2 envs, 32 policy / 640 physics steps |
| Two M7 runtime processes | PASS each — 2 envs, 8 policy / 160 physics steps |
| Semantic inspection | PASS each — 16 samples, 1 selective reset, terminal row and post-reset binding exact |
| Complete-tree comparison | PASS — 33 tensors, 1,020,410 elements, maximum numeric error 0 |
| Eight comparator mutations | PASS — every expected exit was nonzero |
| Formal/frozen/package assertions | PASS — exact heads/status bytes, kit SHA256, driver/core versions, and byte-identical frozen `pip freeze` |
| History/scope | PASS — D0001 commits preserved, one correction commit, exact seven-path rework allowlist |

The D0001 formal candidates and all D0001 evidence remain immutable. Simulator
outcome magnitudes and untrained-policy quality remain non-blocking diagnostics.
