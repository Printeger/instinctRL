# M7-008 Revision 1 Import-Boundary Rework Test Report

## D0002 formal candidate 0001

Status: **PASS**.

| Check | Result | Evidence |
| --- | --- | --- |
| Exact single-exception import probe | PASS | ledger 006 |
| D0001 evidence immutability | PASS, 351 files | ledger 005 |
| Focused execution, twice | PASS, 36 each | ledger 032, 035 |
| Focused direct adapter, twice | PASS, 53 each | ledger 033, 036 |
| Focused isolation, twice | PASS, 72 each | ledger 034, 037 |
| M2 static | PASS, 10 | ledger 038 |
| Accepted M7 pure suites, twice | PASS, 381 each | ledger 039, 041 |
| Relevant M3-M6 regressions, twice | PASS, 203 each | ledger 040, 042 |
| Seventeen manifest rejection cases | PASS, expected exit 2 | ledger 013-029 |
| Two independent process records | PASS, byte-identical | ledger 007-009 |
| Source/frozen/package/allowlist/diff checks | PASS | ledger 030-031, 043-050 |

Candidate 0001 records **1,500 pytest passes and zero skips**. The exact
Order-identified obsolete M6 whole-control-directory assertion is the sole
deselection; the M7 exact source/path auditors replace it. No M7 or behavior
test was deselected, and all 50 commands matched their expected exit code.

Commander probe semantics are exact: the execution module imports
`instinctlab.control.baseline_b0` once, does not import aggregate
`instinctlab.control`, and no other prohibited module has the direct B0 import.
