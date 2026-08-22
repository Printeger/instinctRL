# M6 Pure-Tensor Task Test Report

Order: `M6-001`, revision 1

Attempt: `M6-001-D0002`

| Check | Result | Detail |
| --- | --- | --- |
| M6 analytic/negative suite | PASS | 42/42 development cases before final qualification |
| Direct accepted M5 metrics | PASS | five named tensors from one `ControlCoreResult`; no casts/synthetic labels |
| Floating label rejection | PASS | exact 0/1 only; fraction/range/bool/int/shape/dtype/device/nonfinite negatives |
| Anchor usable gate | PASS | below/at/above `0.1`, including active + loss + zero usable |
| Reward components and exact total | PASS | active/inactive, fixed keys, `[N,1]`, no scale |
| Primary actual tracking discrimination | PASS | actual and `v_final_b` deliberately disagree |
| ICS/null/anchor/clearance/collision gates | PASS | independent analytic fixtures |
| Actor-produced gradients | PASS | finite nonzero gradients through governor/final/actual inputs |
| Termination truth table | PASS | boundaries, simultaneous masks and reason precedence |
| Critic schema/perturbation | PASS | exact `[N,18]` order and named-field changes |
| Actor/M5 one-way isolation | PASS | actor and control tensor equality under privilege changes |
| Comparator negative behavior | PASS | rejects missing/extra/reorder/nonfinite/any float error |
| Two pure CUDA processes | PASS | fixed seed/batch and complete persisted trees |
| Complete comparator | PASS | development: 110 tensors, 1,020,440 elements, max error 0 |
| M5-001 twice | PASS | 20/20 each |
| M5-002 twice | PASS | 18/18 each |
| M4 twice | PASS | 22/22 each |
| M3 twice | PASS | 8/8 each; pattern hash unchanged |
| M2 static | PASS | 10/10 |
| Simulator/environment/PPO/training | NOT RUN | prohibited and absent |

The final fail-closed D0002 evidence candidate reruns all required suites and
CUDA processes after the additive correction commit. Its exact commands,
streams, exits and hashes resolve through the stable D0002 root named by
`EVIDENCE_INDEX.md`; development output is not substituted for final evidence.
