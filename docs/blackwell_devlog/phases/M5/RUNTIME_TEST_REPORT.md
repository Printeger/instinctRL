# M5 TASLAB Runtime-Control Test Report

Order: `M5-002`, revision 1

Attempt: `M5-002-D0002`

| Check | Result | Detail |
| --- | --- | --- |
| Revised M5-001 actor-clean suite | PASS | 20/20; exact five-module scope, unchanged tokens, discriminating Lee negative |
| M5-002 analytic/negative/source suite | PASS | 18/18; CPU/CUDA device rejection, math, topology, atomicity, comparator negatives |
| Headless CUDA process 1 | PASS | one environment, four scenarios, 256 steps |
| Headless CUDA process 2 | PASS | independent process, same fixed contract |
| Complete persisted comparator | PASS | 96 tensors, 4,778,624 elements, maximum absolute error 0 |
| M4 complete regression | PASS | 22/22 |
| M3 complete regression | PASS | 8/8; pattern hash unchanged |
| M2 static regression | PASS | 10/10 |
| Diff/allowlist/accepted implementation integrity | PASS | revision-0 paths plus exact revision-1 test only |
| Actor/reward/PPO/training/M6 | NOT RUN | prohibited and not implemented |

The repeated suites and two fresh runtime processes are rerun by the final
fail-closed evidence ledger; its streams, exits and hashes are referenced by
`RUNTIME_EVIDENCE_INDEX.md`.

## Runtime gates and actual diagnostics

| Measurement | Actual | Classification |
| --- | ---: | --- |
| Total physics steps per process | 256 | fixed contract |
| Maximum quaternion norm error | `1.1920928955078125e-7` | PASS (`<=1e-5`) |
| Zero-command vertical drop | `0.032498717308044434 m` | PASS, not free-fall |
| Free-fall reference drop | `0.502272 m` | reference |
| Positive-X world displacement | `0.020645353943109512 m` | PASS, correct sign |
| 90-degree-yaw world-Y displacement | `0.020645368844270706 m` | PASS, correct sign |
| Reset/replay maximum error | `0.0` | PASS |
| Maximum per-rotor thrust | `3.0850894451141357 N` | PASS |
| Theoretical per-rotor maximum | `7.92378755 N` | bound |
| Independent-process maximum error | `0.0` | PASS |

Hover and tracking quality are non-gating diagnostics. No parameter, gain,
physics, timestep, scenario, tolerance, or gate was retuned.

Headless GLFW/no-display warnings and the Isaac Lab future-deprecation warning
for `set_external_force_and_torque` were observed. The public Isaac Lab 2.3.2
API executed successfully; these warnings did not alter a correctness result.
