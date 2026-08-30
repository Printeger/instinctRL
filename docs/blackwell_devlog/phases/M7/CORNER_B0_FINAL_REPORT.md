# M7-016-D0002 product report

## Outcome

The exact single-mesh `corner` B0 fixture passed all ordered pure, runtime,
comparison, regression and integrity gates. Functional commit
`09aa3941fd4dbe303c9a828779aca8c536081fa7` has parent
`e77092772762ed6af117095cb240ffd078682629`. This file belongs to the required
second documentation commit, whose identity is resolved from Git history.

The selected Warp target is exactly
`/World/MID360StaticScene/CornerWallsMesh`, with 8 points and 12 indices. Both
acquired wall-normal families are non-vacuously visible in every row and
environment, and every valid hit belongs exclusively to one family. Complete
stage traversal proves the separate accepted PlaneMesh, exact 26-entry
inventory and empty forbidden set.

Catalog READY/executable pairs are exactly `corner@7016`, `flat_wall@7015`,
and `open_field@7014`. Flat-wall v2 and open-field v3 retain their accepted
fixture, geometry, schedule, runtime, privacy, metric and claim semantics.

## Integrity

Locked hashes remain RayCaster
`b6a84cb0e506f4a4724d3262e0f533ccb31d75128e1e46282220fcfa688e286c`,
MID360 `a37a1c423d8818194f23023ba969f55025dc867d2280579633a699807a5aea44`
and environment
`e4099b8085e2fe1cf69aac6d249e86d14f0d06ad6dfedc400f717eb14af161b7`.
Driver 580.173.02, Python 3.11.15, PyTorch 2.7.0+cu128, Isaac Sim 5.1.0,
TorchRL/TensorDict 0.10.0 and the frozen repository heads remain unchanged.
D0001 and M7-015 evidence were read and hashed only; neither was modified.

No actor leakage, runtime patch, package change, gate weakening, hard-coded
hit count, merge, push, history rewrite, next-phase or prohibited work was
performed. Result: `READY_FOR_REVIEW`; Commander approval remains required.
