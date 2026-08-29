# M7-014 Test Report

Formal rework candidate: `M7-014-D0003/formal_candidates/0001`.

| Check | Result | Facts |
| --- | --- | --- |
| Commander F3 and Executor equivalent | PASS | consistent two-copy receipt schema forgery rejected after B0 record hash repair; no record/input mutation |
| D0001/D0002 probes | PASS | extra geometry and nested geometry/metric cases remain rejected |
| Static schema audit matrix | PASS | 7/7 atomic and 7/7 consistently repaired mutations rejected |
| Source/catalog/privacy/retention audit | PASS | corrected fixture hash; only open_field READY/executable; nine rows byte/value-equivalent to D0002 and NOT_READY |
| Focused suites, twice | PASS | 116 passed each |
| Accepted M7 suites, twice | PASS | 614 passed each |
| Relevant M3-M6, twice | PASS | 203 passed and one authorized baseline-hash deselection each |
| M2 static | PASS | 10 passed |
| Accepted learned runtime CUDA | PASS | bounded regression |
| Accepted M7-011 environment CUDA | PASS | bounded regression |
| Accepted M7-012 collector CUDA | PASS | bounded regression; not scenario evidence |
| Dedicated open-field CUDA, process 1 | PASS | pre-motion full-stage inventory 25, forbidden set 0; exact 2/8/16/160/160/160/320/339,840 coverage |
| Dedicated open-field CUDA, process 2 | PASS | independent inventory 25, forbidden set 0; same gates and exact qualification SHA |
| Complete comparator | PASS | full decoded equality; 11/11 mutation classes rejected |
| Frozen/package/prior-evidence/history/diff checks | PASS | locked hashes and D0001/D0002 evidence unchanged; functional history additive and clean |

Formal pytest total: 1,876 passed, zero skipped, two disclosed deselections. The deselected M6 test asserts an earlier Order-base whole-file hash and is superseded only for the Order-authorized catalog/test changes; the remaining M6 suite ran.

Metric values are finite and exact under accepted formulas but are not thresholds and are not classified as performance pass/fail.
