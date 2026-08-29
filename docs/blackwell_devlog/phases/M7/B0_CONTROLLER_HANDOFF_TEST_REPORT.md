# M7-009 Revision 1 B0 Controller Handoff Test Report

## D0002 formal candidate 0001

Status: **PASS**.

| Check | Result | Evidence |
| --- | --- | --- |
| D0001 evidence immutability | PASS, 102 files | ledger 005 |
| Executor and Commander smoke-gate probes | PASS | ledger 006-007 |
| Protected source/functional scope/frozen integrity | PASS | ledger 008-009 |
| Focused handoff/smoke contract, twice | PASS, 30 each | ledger 010-011 |
| M2 static | PASS, 10 | ledger 012 |
| Accepted M7 pure suites, twice | PASS, 412 each | ledger 013, 015 |
| Relevant M3-M6 regressions, twice | PASS, 203 each | ledger 014, 016 |
| Headless CUDA smoke, two independent processes | PASS, 256 steps each | ledger 017-018 |
| Exact oracle and closed-thrust validation | PASS, all six errors/counts zero | ledger 019 |
| Unchanged complete-record comparator | PASS, 124 tensors / 33,536 elements | ledger 020 |
| History, original/rework diff, diff-check, clean | PASS | ledger 021-028 |

The candidate records 1,300 pytest passes and zero skips. The only
deselections are the Order-accepted obsolete M6 whole-control-directory
assertion, once per regression repetition. Every formal command returned its
expected exit.

Both CUDA summaries and bundles record exact zero for body-clip, world-
rotation, and complete Lee maximum absolute error and mismatch count. The
separate body norm bound, both physical thrust inequalities, prior topology/
finite/domain/reset/manifest gates, and aggregate gate are true. Only the
original handoff Lee rotor output actuated the wrench adapter.
