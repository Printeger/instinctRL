# M6 Pure-Tensor Task Test Report

Order: `M6-001`, revision 0

Attempt: `M6-001-D0001`

| Check | Result | Detail |
| --- | --- | --- |
| M6 analytic/negative suite | PASS | 27/27 development cases |
| Reward components and exact total | PASS | active/inactive, fixed keys, `[N,1]`, no scale |
| Primary actual tracking discrimination | PASS | actual and `v_final_b` deliberately disagree |
| ICS/null/anchor/clearance/collision gates | PASS | independent analytic fixtures |
| Actor-produced gradients | PASS | finite nonzero gradients through governor/final/actual inputs |
| Termination truth table | PASS | boundaries, simultaneous masks and reason precedence |
| Critic schema/perturbation | PASS | exact `[N,18]` order and named-field changes |
| Actor/M5 one-way isolation | PASS | actor and control tensor equality under privilege changes |
| Comparator negative behavior | PASS | rejects missing/extra/reorder/nonfinite/any float error |
| Two pure CUDA processes | PASS | fixed seed/batch and complete persisted trees |
| Complete comparator | PASS | 108 tensors, 1,020,432 elements, max error 0 |
| M5-001 twice | PASS | 20/20 each |
| M5-002 twice | PASS | 18/18 each |
| M4 twice | PASS | 22/22 each |
| M3 twice | PASS | 8/8 each; pattern hash unchanged |
| M2 static | PASS | 10/10 |
| Simulator/environment/PPO/training | NOT RUN | prohibited and absent |

The final fail-closed evidence candidate reruns the required suites and CUDA
processes after both commits. Its exact commands, streams, exits and hashes are
indexed by `EVIDENCE_INDEX.md`; development output is not substituted for final
evidence.
