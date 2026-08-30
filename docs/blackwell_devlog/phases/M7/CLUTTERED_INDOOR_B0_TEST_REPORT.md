# M7-018 cluttered-indoor B0 test report

## Results

| Gate | Result | Evidence |
| --- | --- | --- |
| Syntax for all 15 functional paths | PASS | `006_syntax` |
| Focused six-file pure suite, run 1 | PASS, 331 | `007_focused_pure_1` |
| Focused six-file pure suite, run 2 | PASS, 331 | `030_focused_pure_2` |
| Clutter CUDA processes | PASS, 2/2 | `014_clutter_formal_1`, `015_clutter_formal_2` |
| Clutter complete comparator | PASS, exact equality, 20 mutations | `017_clutter_comparator_retry1` |
| Prior scenario CUDA processes | PASS, 8/8 | `018`--`025` |
| Corridor/corner/flat/open comparators | PASS, 16/16/13/11 mutations | `026`--`029` |
| Accepted M7 pure suite, twice | PASS, 829 each | `031_accepted_m7_1`, `033_accepted_m7_2` |
| M3--M6 relevant suite, twice | PASS, 203 + 33 subtests each; 1 expected deselection | `032_m3_m6_1`, `034_m3_m6_2` |
| M2 static | PASS, 10 | `035_m2_static` |
| Learned/M7-011/M7-012 CUDA regressions | PASS, 3/3 | `036`--`038` |
| Package, frozen source and historical evidence | PASS | `039`--`042` |

Both clutter processes recorded all gates true, zero resets/done, one receipt,
one evaluation record and one qualification record. Each recorded 339,840 rays
and the exact 160/160/160/320 runtime coverage. No unrun check is reported as
PASS.

The failed `011_functional_commit` was an environment-only Git identity error;
`012_functional_commit_retry1` used the accepted per-command identity variables
without changing Git configuration. The failed `016_clutter_comparator` used
an incorrect `.json.pt` input path and did not enter comparison; the exact same
committed comparator passed at `017` against the correct `.pt` artifacts.
