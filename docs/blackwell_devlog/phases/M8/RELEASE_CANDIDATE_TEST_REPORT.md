# M8 release-candidate test report

## Revision 1 / D0002 correction

Commander review found that D0001 encoded `580.173.02-open` although the exact
frozen driver evidence is `580.173.02`. Revision 1 changes only that runtime
identity and the matching focused expectation. Fresh D0002 results are:

| Gate | Result | D0002 labels |
| --- | --- | --- |
| Exact two-line code/test repair audit | PASS | `005_repair_probe` |
| Focused M8 run 1 | PASS — 30 passed | `006_focused_m8_1` |
| Focused M8 run 2 | PASS — 30 passed | `007_focused_m8_2` |
| Accepted M7 pure run 1 | PASS — 976 passed | `008_accepted_m7_1` |
| Accepted M7 pure run 2 | PASS — 976 passed | `009_accepted_m7_2` |
| Relevant M3-M6 run 1 | PASS — 203 passed, 33 subtests, 1 deselected | `010_m3_m6_1` |
| Relevant M3-M6 run 2 | PASS — 203 passed, 33 subtests, 1 deselected | `011_m3_m6_2` |
| M2 static | PASS — 10 passed | `012_m2_static` |

D0001 results below remain retained historical evidence; they are not
relabeled as D0002 results.

## Results

| Gate | Result | Evidence labels |
| --- | --- | --- |
| Initial focused development attempt | FAIL — 28 passed, 2 test-assertion failures | `003_phase1_focused` |
| Corrected focused development gate | PASS — 30 passed | `005_phase1_retry_focused` |
| Required focused M8 run 1 | PASS — 30 passed | `011_focused_m8_1` |
| Required focused M8 run 2 | PASS — 30 passed | `012_focused_m8_2` |
| Accepted M7 pure run 1 | PASS — 976 passed | `013_accepted_m7_1` |
| Accepted M7 pure run 2 | PASS — 976 passed | `014_accepted_m7_2` |
| Relevant M3-M6 run 1 | PASS — 203 passed, 33 subtests, 1 deselected | `015_m3_m6_1` |
| Relevant M3-M6 run 2 | PASS — 203 passed, 33 subtests, 1 deselected | `016_m3_m6_2` |
| M2 static | PASS — 10 passed | `017_m2_static` |
| Frozen/M7 integrity baseline | PASS | `018_integrity_before_docs` |

The initial failures were test defects: one assertion prohibited spaces inside
legitimate string values, and another confused the approved actor input phrase
“timestamps and age” with a forbidden timestamp field. The tests were narrowed
to canonical separators and field-name checks. No product contract was relaxed.

## Focused coverage

The 30 pure tests cover exact exports/signatures/field and enum order; exact
accepted chain, approval hashes and runtime identity; frozen/non-subclassable
records; actor and M7 availability truth; gate classifications; false-claim
rejection; canonical ASCII/hash/round trip; mutation isolation; malformed,
duplicate, reordered, alias, nonfinite and resigned semantic negatives; and
recursive import/reverse-import isolation.

No simulator, scenario, training or performance command was run.
