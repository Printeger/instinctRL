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
