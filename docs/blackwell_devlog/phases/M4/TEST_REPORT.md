# M4 Test Report

Order: `M4-001`, revision 1

Passing attempt: `M4-001-D0002`

| Gate | Result | Evidence |
| --- | --- | --- |
| Focused revision-1 semantic/source suite | PASS, 22/22 | `dev_runs/0002` |
| First coherent pure-CUDA process | PASS | `dev_runs/0003` |
| Second independent coherent pure-CUDA process | PASS | `dev_runs/0004` |
| Unchanged complete CUDA comparator | PASS, 3,059,184 elements, max error 0 | `dev_runs/0005` |
| Syntax and both diff-check surfaces | PASS | `dev_runs/0006` |
| Final expanded M4 suite, twice | PASS, 22/22 each | `dev_runs/0007` |
| M3 static regression, twice | PASS, 8/8 each | `dev_runs/0008` |
| Git/allowlist/integrity/prohibited-scope | PASS, recorded in final candidate | `final_candidates/0001` |

Revision 0's failures remain immutable under D0001. D0002 does not relabel or
modify them. D0002 introduces no hidden failed development subrun.

The semantic suite covers exact shape/keys, field and channel order, boundary
ray sentinels, M3 mask preservation, reliability/age, repeated frames,
timestamp rejection, rollover, full/selective reset, gravity validation,
malformed/missing/type/shape/batch/dtype/device/nonfinite cases, alias attempts,
recursive leakage fixtures, source signature/producers, and machine-readable
slice provenance. Negative cases raise fail-closed exceptions.

Revision 1 adds exact invalid/non-40 and near-40 rejection, failed-call
atomicity, mixed valid/invalid 40 m authority, zero/populated frame
classification, partial-frame rejection, populated gravity validation, and a
direct AST proof against accepted M3 constants and elevation-outer,
azimuth-fastest flattening.
