# M3 Test Report

## Results

| Result | Test | Evidence |
| --- | --- | --- |
| PASS | Syntax check for all five M3 Python files. | `dev_runs/0006` |
| PASS | Static/analytic suite twice: 6 tests each; stdout and stderr byte-identical. | `dev_runs/0006` |
| PASS | Pattern hash repeated identically: `4c715f5e396f62fabea7b947c42a9734205ab5dad94cba6eb9549328d4b48b8b`. | `dev_runs/0006` |
| PASS | Independent CUDA smoke process 1. | `dev_runs/0007` |
| PASS | Independent CUDA smoke process 2. | `dev_runs/0008` |
| PASS | Full comparison: 21,240 ranges and 21,240 mask values; maximum numeric error `0.0` at `1e-5`. | `dev_runs/0008/comparison.json` |
| PASS | Runtime produced 4,290 valid and 16,950 invalid rays; shape `[1,21240]`; finite serialized output. | `dev_runs/0007/smoke.json` |
| PASS | M2 static suite and bounded M2 CUDA smoke regression. | `dev_runs/0012` |
| PASS | Formal/frozen identities, driver/packages, and M1B/M2 unchanged-path check. | `dev_runs/0011` |
| FAIL | First static attempt imported Kit-dependent `carb` before an App existed. | `dev_runs/0001` |
| FAIL | Second static attempt used exact Python-float equality for a float32 `0.1` representation. | `dev_runs/0002` |

The two failed development subruns are preserved and were corrected without changing the sensor contract or runtime.

## Acceptance mapping

- Geometry/order/axes/unit norm/device/dtype/contiguity: PASS.
- Mount quaternion and positive-pitch sign: PASS.
- Closed range edges and NaN/Inf/no-hit/below/above negative fixtures: PASS.
- Public-API CUDA initialization with valid and invalid returns: PASS.
- Full-vector cross-process determinism: PASS.
- No control, actor, M4, PPO, or training surface: PASS.

## Revision 1 results — M3-001-D0002

| Result | Test | Evidence |
| --- | --- | --- |
| PASS | Nontrivial parent/mount transform plus malformed shape, batch, dtype, and device fixtures. | `D0002/dev_runs/0010` |
| PASS | Complete 8-test M3 static suite twice with byte-identical streams and unchanged pattern hash. | `D0002/dev_runs/0010` |
| PASS | Syntax checks with bytecode redirected outside the product tree. | `D0002/dev_runs/0010` |
| PASS | Corrected CUDA smoke process 1 and full 21,240-ray plane oracle. | `D0002/dev_runs/0008` |
| PASS | Corrected CUDA smoke process 2. | `D0002/dev_runs/0009` |
| PASS | Unchanged comparator: all 21,240 ranges and 21,240 mask bits; cross-process maximum error `0.0`. | `D0002/dev_runs/0009/comparison.json` |
| PASS | Mounted translation separation error `4.842877388e-8 m`; quaternion composition error `2.030979829e-9`. | corrected smoke JSON |
| PASS | Plane mask mismatch count `0`; maximum valid-range analytic error `7.62939453125e-6 m`. | corrected smoke JSON |
| PASS | Wrong parent origin rejected: 18 mask mismatches and maximum range discrepancy `0.9721717834 m`. | corrected smoke JSON |
| PASS | Complete M2 static and CUDA runtime regression. | `D0002/dev_runs/0011` |
| FAIL | Initial corrected-origin smoke exposed the default huge-plane numerical error. | `D0002/dev_runs/0002`–`0003` |
| FAIL | Finite cuboid and initial tiled-plane/oracle variants remained above `1e-5`; preserved while the geometric/numerical cause was isolated. | `D0002/dev_runs/0004`–`0007` |

No threshold was relaxed. The final oracle uses sequential public quaternion
applications, matching the documented float32 composition order without reading
private RayCaster buffers, and computes Euclidean distance using the actual
world-direction norm.
