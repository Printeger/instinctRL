# M2 Test Report

## Development results

| Gate | Result | Evidence |
| --- | --- | --- |
| Static suite run 1 | PASS — 10/10 | `dev_runs/0001/static_1.*` |
| Static suite run 2 | PASS — 10/10, byte-identical streams | `dev_runs/0001/static_2.*` |
| Python syntax | PASS | `dev_runs/0001/py_compile.*` |
| CUDA smoke process 1 | PASS | `dev_runs/0002/smoke.json` |
| CUDA smoke process 2 | PASS | `dev_runs/0003/smoke.json` |
| Cross-process comparator | PASS — 442 numerics, max error 0 | `dev_runs/0003/comparison.json` |

## Runtime gates

| Measurement | Observed | Gate | Result |
| --- | ---: | ---: | --- |
| Bodies | 5, ordered base + rotor 0–3 | exact | PASS |
| Rotor joints | 4, ordered rotor 0–3 | exact | PASS |
| Device/dtype | `cuda:0`, `torch.float32` | CUDA/finite | PASS |
| In-process replay max state error | 0 | `<=1e-5` | PASS |
| Cross-process max numeric error | 0 | `<=1e-5` | PASS |
| Quaternion norm error | `5.9604645e-8` | `<=1e-5` | PASS |
| Horizontal displacement | `1.3015629e-10 m` | `<=1e-4 m` | PASS |
| Attitude-angle drift | `6.9053397e-4 rad` | `<=1e-3 rad` | PASS |
| Vertical drop after 32 steps | `0.129491925 m` | `[0.05,0.30] m` | PASS |
| Physics steps/process | 64 | exact | PASS |
| Applied control calls | 0 | exact | PASS |

## Effective properties

PhysX reports base mass `1.1583074331 kg` and total articulated mass
`1.1815179586 kg`. These are deliberately distinct and were observed without
retuning. Full centers of mass and inertia tensors are retained in both smoke
JSON files.

## Not run

Hover, thrust, controller, sensor, MID360, actor, reward, PPO, training,
evaluation, and M3/later tests were **NOT RUN** because they are prohibited.
