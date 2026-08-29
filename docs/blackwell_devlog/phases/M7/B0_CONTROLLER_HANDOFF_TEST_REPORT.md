# M7-009 B0 TASLAB Controller Handoff Test Report

## Formal candidate 0001

Status: **PASS**.

| Check | Result | Evidence |
| --- | --- | --- |
| Pure fixture, negative, call-order, and source auditors | PASS | ledger 005-008 |
| Frozen source/runtime/package/driver integrity | PASS | ledger 009 |
| Focused handoff, twice | PASS, 29 each | ledger 010, 014 |
| Focused manifest, twice | PASS, 37 each | ledger 011, 015 |
| Focused reverse-import adapter, twice | PASS, 53 each | ledger 012, 016 |
| Focused isolation, twice | PASS, 72 each | ledger 013, 017 |
| M2 static | PASS, 10 | ledger 018 |
| Accepted M7 pure suites, twice | PASS, 411 each | ledger 019, 021 |
| Relevant M3-M6 regressions, twice | PASS, 203 each | ledger 020, 022 |
| Headless CUDA smoke, two independent processes | PASS, 256 steps each | ledger 023-024 |
| Exact complete-record comparison | PASS, 72 tensors / 25,856 elements | ledger 025 |
| History, diff, allowlist, diff-check, clean | PASS | ledger 026-031 |

The candidate records 1,620 pytest passes, zero skips, and exactly two
deselections of the Order-accepted obsolete M6 whole-control-directory
assertion (once per regression repetition). Every formal command returned its
expected exit. Both CUDA runs passed all finite, topology, quaternion,
clipping, rotor-domain, thrust-bound, manifest, step-completeness, and reset-
replay gates; their normalized complete tensor records compare exactly.

No simulator acceptance threshold concerns tracking, displacement, collision,
clearance, reward, or performance.
