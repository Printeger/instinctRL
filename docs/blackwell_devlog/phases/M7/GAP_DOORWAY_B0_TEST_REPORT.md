# M7 Gap/Doorway B0 Test Report

## Results

| Check | Result | Evidence |
| --- | --- | --- |
| Focused catalog and seven scenario suites, repeated | PASS | commands `036`, `037`: 445 passed each |
| Accepted M7 pure suites, repeated | PASS | commands `038`, `039`: 943 passed each |
| M3-M6 regression, repeated | PASS | commands `040`, `041`: 203 passed, 1 deselected, 33 subtests each |
| M2 static | PASS | command `042`: 10 passed |
| Doorway fresh CUDA pair | PASS | commands `015`, `016` |
| Doorway exact comparator | PASS | command `017`: exact equality and 21/21 mutation rejection |
| Six prior scenario fresh pairs | PASS | commands `018`-`035`: 12 processes and 6 comparators |
| Learned/M7-011/M7-012 CUDA regressions | PASS | commands `043`-`045` |
| Frozen/package/source probe | PASS | command `046` |
| M7-019 D0001/D0002/D0003 evidence identities | PASS | commands `052`-`056` |

The doorway pair records the exact 26-entry inventory, zero forbidden geometry,
all 16 cells with nonzero negative-jamb, positive-jamb and lintel counts,
exclusive valid-hit membership, exact `-x` normals, unchanged B0 body commands,
positive normalized world-x direction, strict aperture intersections, zero
actor/training/reset calls and one bound receipt, evaluation record and
qualification record per process.

The initial pure run reported two test-expectation defects, and two early CUDA
diagnostics exposed persisted float-rounding and catalog-ID defects. All are
retained in commands `003`, `005`-`007`; corrective runs `004`, `008` and all
formal runs pass. No gate was weakened.
