# M7-016-D0002 test report

## Results

| Gate | Result | Exact result |
| --- | --- | --- |
| Commander-equivalent single-mesh probe | PASS | all three locked source hashes and first-descendant behavior confirmed |
| Syntax | PASS | 10 ordered Python files compiled |
| Focused pure suites, twice | PASS | 219 passed per run |
| Accepted M7 suites, twice | PASS | 717 passed per run |
| Relevant M3-M6 suites, twice | PASS | 203 passed, 1 accepted deselection and 33 subtests per run |
| M2 static | PASS | 10 passed |
| Fresh corner processes | PASS | 2/2, exact-equal wall counts and records |
| Fresh flat-wall v2 processes | PASS | 2/2 |
| Fresh open-field v3 processes | PASS | 2/2 |
| Corner comparator | PASS | complete equality; 16 mutation classes rejected |
| Unchanged flat comparator | PASS | complete equality; 13 mutation classes rejected |
| Unchanged open comparator | PASS | complete equality; 11 mutation classes rejected |
| Learned/M7-011/M7-012 CUDA regressions | PASS | 3/3 |
| Locked/runtime integrity | PASS | accepted hashes, versions and frozen identities retained |

Formal repeated pytest execution totals 2,288 passes, two accepted
deselections and 66 subtests. The first development focused run reported 216
passes and three stale test-identity failures; the three tests were corrected
without changing runtime semantics and the same gate then passed 219 tests.

## Corner runtime facts

Both independent formal processes produced the same matrices:

```text
x: ((7611,4638),(7482,4538),(7433,4517),(7543,4624),
    (7610,4670),(7561,4628),(7571,4615),(7597,4640))
y: ((3728,4812),(3658,4735),(3718,4650),(3779,4695),
    (3764,4778),(3764,4755),(3737,4762),(3718,4820))
valid: ((11339,9450),(11140,9273),(11151,9167),(11322,9319),
        (11374,9448),(11325,9383),(11308,9377),(11315,9460))
```

All acquired x/y normals, exclusive membership, 8/12 selected Warp counts,
26-entry inventory, plane topology, coverage, receipt, record, privacy,
finiteness and no-reset gates passed.
