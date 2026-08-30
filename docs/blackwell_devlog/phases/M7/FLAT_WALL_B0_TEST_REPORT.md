# M7-015 Test Report

## Results

| Gate | Result | Exact result |
| --- | --- | --- |
| Syntax/static compile | PASS | Eight changed Python programs compile |
| Focused catalog/open-field/flat-wall suites, run 1 | PASS | 165 passed |
| Focused suites, run 2 | PASS | 165 passed |
| M2 static | PASS | 10 passed |
| Accepted M7 pure suites, run 1 | PASS | 663 passed |
| Accepted M7 pure suites, run 2 | PASS | 663 passed |
| Relevant M3–M6, run 1 | PASS | 203 passed, 1 accepted deselection, 33 subtests passed |
| Relevant M3–M6, run 2 | PASS | 203 passed, 1 accepted deselection, 33 subtests passed |
| Flat-wall CUDA process 1 | PASS | 26 inventory, 0 forbidden, 8/160/160/160/320/339,840 coverage |
| Flat-wall CUDA process 2 | PASS | Exact independent decoded equality with process 1 |
| Flat-wall comparator | PASS | Complete equality; 13 mutation classes rejected |
| Open-field v2 CUDA processes | PASS | Two fresh processes; unchanged 25 inventory and 0 forbidden |
| Open-field comparator | PASS | Complete equality; unchanged 11 mutation classes rejected |
| Learned/M7-011/M7-012 CUDA regressions | PASS | Three bounded processes passed |
| Functional/history/allowlist/locked/frozen audit | PASS | 9 functional paths; direct-child commit; exact locks/identities |
| Training, B1–B8, M8, scale runs | NOT RUN | Forbidden by Order M7-015 |
| Performance, safety or baseline comparison | NOT RUN | Outside qualification claim |

Total counted pytest results are 2,072 passed and two disclosed accepted M6
deselections. The deselected test checks an earlier order-base byte identity
that is intentionally superseded by accepted later milestones; it is the same
bounded deselection used by accepted M7 evidence.

## Live wall evidence

The two independent flat-wall processes produced identical wall-hit counts:

```text
((8625,5921), (8501,5824), (8449,5801), (8559,5894),
 (8622,5939), (8574,5901), (8575,5887), (8594,5914))
```

All sixteen row/environment cells exceed zero. All valid hits classified as
wall or plane; wall normals verified -x at `1e-5`. The live plane topology was
preserved. Zero actor, training, reset, terminated and truncated calls occurred
in either scenario process; each emitted exactly one receipt, B0 record and
qualification record.

## Development-only rejected candidates

Two pre-formal runner candidates were rejected and retained in raw logs: the
first exposed a USD tuple type-inference mismatch; the second proved the
accepted environment rejects a pre-existing RayCaster root. Both were repaired
before the functional commit by exact list authoring and a counted runner-local
one-call root-preservation bridge. No failed candidate was relabelled PASS.
