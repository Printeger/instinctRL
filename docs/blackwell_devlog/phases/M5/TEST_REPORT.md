# M5 Test Report

Order: `M5-001`

| Check | Result | Detail |
| --- | --- | --- |
| Pure-tensor M5 semantic/negative/source suite | PASS | 20/20 cases |
| Governor endpoints/B0/smoothing/norm direction/domain rejection | PASS | analytic tensor oracles |
| Anchor lifecycle/selective reset/precedence/Huber/atomicity | PASS | dense cache remains private |
| Mounted M3 ray count/order/norm/sentinels/AST provenance | PASS | 21,240 body-frame unit rays |
| ICS empty/active/emergency/monotonic/filter/latency/stale cases | PASS | beta bounded in `[0,1]` |
| Adapter rotations/norm/malformed and quaternion isolation | PASS | identity, yaw ±90, roll/pitch |
| Pipeline schema/B0/fixed action/reset/failed-call atomicity | PASS | exact public/internal keys |
| Independent CUDA processes and complete comparator | PASS | 69 tensors, 1,019,658 elements, max error 0 |
| M4 complete regression twice | PASS | 22/22 each run |
| M3 complete regression twice | PASS | 8/8 each run |
| Syntax and both base-to-HEAD diff checks | PASS | empty diagnostics |
| Simulator/controller/reward/PPO/training | NOT RUN | prohibited by Order and not required |

The final raw command ledger, streams, hashes, runtime integrity checks, and
self-audit are indexed under the D0001 evidence path named in `EVIDENCE_INDEX.md`.
