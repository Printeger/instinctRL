# RayCaster Exact-Normal Test Report

## Results

| Check | Result |
| --- | --- |
| Focused adapter contract | PASS — 51 tests |
| Accepted M7 plus relevant M3-M6 pure replay 1 | PASS — 378 tests, 33 subtests |
| Same pure replay 2 | PASS — 378 tests, 33 subtests |
| M2 static | PASS — 10 tests |
| Frozen source/API/hash and no-leak audit | PASS — 3 source hashes, 23 consumer files, 5 accepted M7 files byte-exact |
| Live process 1 | PASS — 84,960 ray samples |
| Live process 2 | PASS — 84,960 ray samples |
| Complete-tree comparison | PASS — 44 tensors, 1,869,180 elements, maximum numeric error 0 |
| Eleven comparator mutations | PASS — all rejected |

Each live process contains two scans, two environments, and 21,240 rays per
environment. Across both scans, valid/miss counts are 17,362/67,598: initial
8,580/33,900 and after-action 8,782/33,698. Valid-normal norm error, planar XY
absolute value, and `abs(Z)-1` error are all exactly zero. Sensor/extra-cast hit
maximum error is zero. Maximum start-to-hit distance consistency error is
`3.814697265625e-06 m` initially and `7.62939453125e-06 m` after the action,
both within the declared `1e-5 m` tolerance.

Every live `[scan,env]` sample reports rank one, zero `sigma_min`, condition
zero/false, and weak direction zero/false. Both valid and missed beams are
present. Invalid raw normals retain positive infinity and qualified invalid
normals are exact zero.

Mutation coverage is missing, extra, reordered, nonfinite, any-error, altered
normal, mask, hit, distance, rank, and frozen provenance. Each invocation has
an expected nonzero exit and is not represented as a command failure.

## Retained development failures

- Initial focused collection produced 24 passes/27 failures because the test's
  nominal tensors were views and its `inf*0` miss construction contained NaN.
  Strict production validation correctly rejected them. The fixture was made
  independently owned with explicit positive-infinity misses.
- The next two focused runs were 50 passes/1 failure because a dictionary test
  compared multi-element tensors directly and then overconstrained captured
  closure locals. Only the assertion was corrected; the official call gate was
  not changed. The resulting focused suite is 51 passes.

No package was changed, no gate was relaxed, and no failed or unrun check is
reported as PASS.
