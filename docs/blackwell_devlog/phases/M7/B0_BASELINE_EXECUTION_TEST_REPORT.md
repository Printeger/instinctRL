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

## Formal candidate 0002

Status: **PASS**.

| Check | Result |
| --- | --- |
| Focused execution, twice | PASS, 36 each |
| Updated isolation, twice | PASS, 72 each |
| M2 static | PASS, 10 |
| Accepted M7 pure suites, twice | PASS, 381 each |
| Relevant M3-M6 regressions, twice | PASS, 203 each |
| Manifest rejection matrix | PASS, 17 expected exits |
| Independent processes | PASS, 2 byte-identical records |
| Source/import/frozen/package/history/diff checks | PASS |

Total recorded pytest results: **1394 passed, 0 skipped**. The obsolete M6
whole-control-directory assertion was the sole deselection and was replaced by
the M7 exact-path/source audit. No M7 test was deselected. Candidate 0002 ran
44 commands, all with actual exit equal to expected exit.
