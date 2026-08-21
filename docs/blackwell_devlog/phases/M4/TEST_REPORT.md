# M4 Test Report

Order: `M4-001`  
Attempt: `M4-001-D0001`

| Gate | Result | Evidence |
| --- | --- | --- |
| Syntax for six M4 Python files | PASS | `dev_runs/0005` |
| CPU semantic/negative/source/provenance suite | PASS, 18/18 | `dev_runs/0012` |
| First pure-CUDA process | PASS | `dev_runs/0006` |
| Second independent pure-CUDA process | PASS | `dev_runs/0007` |
| Complete CUDA output comparison | PASS, 3,059,184 elements, max error 0 | `dev_runs/0009` |
| M3 static regression, twice | PASS, 8/8 each | `dev_runs/0010` |
| Final expanded M4 suite, twice | PASS, 18/18 each | `dev_runs/0014` |
| Git/allowlist/integrity/prohibited-scope | PASS, recorded in final candidate | `final_candidates/0001` |

Development failures are not hidden:

- `dev_runs/0002`: implementation reliability reshape plus source-test defects.
- `dev_runs/0003`: one test compared a float32 sentinel to a Python decimal.
- `dev_runs/0008`: comparator invoked with system Python lacking Torch.

Each failure exited nonzero, retained stdout/stderr, and was repaired without
relaxing the contract or changing the frozen environment.

The semantic suite covers exact shape/keys, field and channel order, boundary
ray sentinels, M3 mask preservation, reliability/age, repeated frames,
timestamp rejection, rollover, full/selective reset, gravity validation,
malformed/missing/type/shape/batch/dtype/device/nonfinite cases, alias attempts,
recursive leakage fixtures, source signature/producers, and machine-readable
slice provenance. Negative cases raise fail-closed exceptions.
