# M7-008 B0 Manifest Execution Test Report

## Formal candidate 0001

Status: **REJECTED**.

| Check | Result | Evidence |
| --- | --- | --- |
| Focused execution, twice | PASS, 36 each | candidate 0001 raw logs 030 and 032 |
| Updated isolation, twice | PASS, 72 each | candidate 0001 raw logs 031 and 033 |
| M2 static | PASS, 10 | candidate 0001 raw log 034 |
| 17 manifest rejection cases | PASS | candidate 0001 ledger entries 012-028 |
| Two independent process records | PASS, byte-identical | candidate 0001 ledger entries 005-007 |
| Source/frozen/package integrity | PASS | candidate 0001 ledger entries 029 and preceding source audit |
| Accepted M7 pure suite, first pass | FAIL, 380 passed / 1 failed | candidate 0001 raw log 035 |

The sole failure is the accepted M7-007 reverse-import assertion, which still
forbids every evaluation module from importing the accepted B0 adapter. M7-008
requires the new execution module to be the sole such evaluation-side import.
No test was deselected. Candidate 0001 remains immutable; a bounded nonsemantic
import-boundary repair and fresh candidate are required.
