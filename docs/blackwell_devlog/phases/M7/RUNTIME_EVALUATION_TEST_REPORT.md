# Runtime Evaluation Test Report

Formal candidate `0002`: **PASS**, 42/42 commands with expected exits.

| Check | Result |
| --- | --- |
| Runtime pure contract | PASS — 17 tests |
| M7-001 plus M3-M6 pure replay, twice | PASS — 242 tests and 33 subtests each |
| M2 static | PASS |
| Accepted M6 environment smoke | PASS — 2 envs, 32 policy / 640 physics steps |
| Two M7 runtime processes | PASS each — 2 envs, 8 policy / 160 physics steps |
| Semantic inspection | PASS each — 16 samples, 1 selective reset |
| Complete-tree comparison | PASS — 29 tensors, 682 elements, maximum error 0 |
| Eight comparator mutations | PASS — every expected exit was nonzero |
| Accepted-source/static/external integrity | PASS |

Preserved candidate `0001` failed only an extra `.pt` container-byte equality assertion;
its semantic comparison was already zero. Candidate `0002` applies the ordered recursive
semantic comparator required by the Order. Simulator outcome magnitudes were not gates.
