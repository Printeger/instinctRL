# M7-014 Test Report

Formal candidate: `M7-014-D0001/formal_candidates/0001`.

| Check | Result | Facts |
| --- | --- | --- |
| Source/catalog/privacy/retention audit | PASS | exact fixture hash; only open_field READY/executable; nine NOT_READY; no aggregate export or old-smoke import |
| Atomic manifest/run-plan matrix | PASS | exact READY case executable; status/scenario/seed mismatches fail closed |
| Focused suites, twice | PASS | 93 passed each |
| Accepted M7 suites, twice | PASS | 591 passed each |
| Relevant M3-M6, twice | PASS | 203 passed and one authorized baseline-hash deselection each |
| M2 static | PASS | 10 passed |
| Accepted learned runtime CUDA | PASS | bounded regression |
| Accepted M7-011 environment CUDA | PASS | bounded regression |
| Accepted M7-012 collector CUDA | PASS | bounded regression; not scenario evidence |
| Dedicated open-field CUDA, process 1 | PASS | exact 2/8/16/160/160/160/320/339,840 coverage; zero reset/done/actor/training |
| Dedicated open-field CUDA, process 2 | PASS | same independent gates and exact qualification SHA |
| Complete comparator | PASS | full decoded equality; 11/11 mutation classes rejected |
| Frozen/source/history/diff checks | PASS | locked hashes unchanged; functional history additive and clean |

Formal pytest total: 1,784 passed, zero skipped, two disclosed deselections. The deselected M6 test asserts an earlier Order-base whole-file hash and is superseded only for the two Order-authorized catalog/test changes; the remaining M6 suite ran.

Metric values are finite and exact under accepted formulas but are not thresholds and are not classified as performance pass/fail.
