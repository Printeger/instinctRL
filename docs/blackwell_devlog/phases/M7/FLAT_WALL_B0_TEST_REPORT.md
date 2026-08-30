# M7-015-D0002 Test Report

## Results

| Gate | Result | Exact result |
| --- | --- | --- |
| Syntax/static compile | PASS | Three authorized Python files compile |
| Commander-equivalent exactness probe | PASS | Exact ordered literal and exposed provenance copies pass; hit-count observation remains non-blocking |
| Focused catalog/open-field/flat-wall suites, run 1 | PASS | 166 passed |
| Focused suites, run 2 | PASS | 166 passed |
| M2 static | PASS | 10 passed |
| Accepted M7 pure suites, run 1 | PASS | 664 passed |
| Accepted M7 pure suites, run 2 | PASS | 664 passed |
| Relevant M3–M6, run 1 | PASS | 203 passed, 1 accepted deselection, 33 subtests passed |
| Relevant M3–M6, run 2 | PASS | 203 passed, 1 accepted deselection, 33 subtests passed |
| Flat-wall CUDA processes | PASS | Two fresh processes; exact 26 inventory, 0 forbidden, 8/160/160/160/320/339,840 coverage |
| Flat-wall comparator | PASS | Complete canonical equality; 13 mutation classes rejected |
| Open-field v2 CUDA processes | PASS | Two fresh processes; unchanged 25 inventory and 0 forbidden |
| Open-field comparator | PASS | Complete canonical equality; 11 mutation classes rejected |
| Learned/M7-011/M7-012 CUDA regressions | PASS | Three bounded processes passed |
| Functional allowlist/locked/frozen/D0001 audit | PASS | 3 functional paths; direct-child commit; exact locks and D0001 hashes |
| Training, B1–B8, M8, scale runs | NOT RUN | Forbidden by Order M7-015 Revision 1 |
| Performance, safety or baseline comparison | NOT RUN | Outside qualification claim |

Total counted pytest results are 2,076 passed and two disclosed accepted M6
deselections. The four-result increase from D0001 is the one new literal
regression appearing in each of two focused and two accepted-M7 runs.

## Live wall evidence

The two independent flat-wall processes produced identical runtime-derived
wall-hit counts:

```text
((8625,5921), (8501,5824), (8449,5801), (8559,5894),
 (8622,5939), (8574,5901), (8575,5887), (8594,5914))
```

All sixteen row/environment cells exceed zero. All valid hits classified as
wall or plane; acquired wall normals verified -x at `1e-5`. Geometry,
visibility, call counts, coverage, privacy, metrics and claims are exactly
D0001-equivalent apart from the ordered identity and its dependent hashes.

## Coordination evidence retries

One nonexistent package-probe path was recorded with exit 2, then replaced by
the accepted read-only `pip list --format=json` command. Two early audit runs
recorded exit 1 because the coordination-only audit first assumed an Isaac app
import context and then treated canonical JSON strings as decoded mappings.
The audit was corrected without a product change and subsequently passed
twice. No failed command was relabelled PASS.
