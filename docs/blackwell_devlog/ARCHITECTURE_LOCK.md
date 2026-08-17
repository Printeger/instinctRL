# Blackwell migration architecture lock

Status: locked at bootstrap

Applies to: `blackwell-migration`

Source baseline: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`

## 1. Repository topology

The formal source checkout `/home/mint/rl_dev/instinctRL` remains a clean, read-only witness of the reviewed baseline. Migration commits are made only in `/home/mint/rl_dev/worktrees/instinctRL-blackwell`. Coordination state lives in the independent repository `/home/mint/rl_dev/instinctRL_coordination` and is never imported into product history.

Legacy runtime and evidence remain in place under `isaac-training/`, `ros1/`, `ros2/`, existing root tooling, and `docs/instinctRL_devlog/`. They are comparison inputs, not a patch target. The future Isaac Lab runtime has exactly one reserved root, `isaaclab-training/`, and may first be created only by an accepted M1B order. Blackwell-specific evidence lives under `docs/blackwell_devlog/`.

No compatibility shortcut may make the legacy runtime masquerade as the new runtime, import new implementation back into the frozen tree, or mutate a third-party baseline in place. Cross-boundary reuse must be an explicit, provenance-recorded copy or rewrite authorized by an order, with license and API compatibility resolved first.

## 2. Ordered architecture increments

| Milestone | Locked outcome | Entry gate |
| --- | --- | --- |
| M1 | TASLAB provenance/design (M1A), then separately ordered platform asset/config migration (M1B). | Bootstrap governance committed; M1A accepted before M1B. |
| M2 | Minimal modern Isaac Lab runtime boots TASLAB deterministically with no learning stack. | M1 platform migration accepted. |
| M3 | MID360 geometry, frames, ray ordering, validity and timing are migrated and statically/runtime qualified. | M2 platform smoke accepted. |
| M4 | Actor observation/history schema is implemented and proves the information boundary. | M3 sensor contract accepted. |
| M5 | Velocity-control adapter and governor/anchor/ICS components integrate without actor leakage. | M4 schema/audit accepted. |
| M6 | Rewards, critic-only privilege, PPO plumbing, and only explicitly bounded smoke training integrate. | M5 control path accepted. |
| M7 | Evaluation, metrics, ablations, reproducibility and actor-input audits close. | M6 bounded pipeline accepted. |
| M8 | End-to-end Blackwell qualification and release evidence close; scale training remains a separate explicit authorization. | M7 report accepted. |

One milestone report must be accepted before its successor is ordered. An acceptance suggestion inside an Executor report is not approval.

## 3. TASLAB platform contract

TASLAB_UAV is a project platform derived from a third-party AechPro Isaac Sim 5.1 fork plus local feasibility commits. Modern OmniDrones lacks this class. Migration therefore requires an explicit port; it is not an upstream feature toggle.

Before any asset is copied or converted, M1A must identify the legacy class and registry path, YAML dynamics, binary USD, Lee gains, `RotorGroup`, `MultirotorBase`, `VelController`, all runtime consumers, and the corresponding modern APIs. The record must distinguish authored data from runtime-effective data and state units, frames, prim paths, rotor count/order/direction, joints, body/rotor topology, transforms, and controller command frame. Unknowns remain unknown until demonstrated.

USD/YAML conflicts, stale README claims, parameter comments, binary-crate compatibility, and license/ownership uncertainty are hard migration risks. None may be resolved by assumption.

## 4. Actor and controller boundary

The learned deployed actor may receive only the order-approved MID360 measurement/history representation, validity/reliability information, time/age fields, approved onboard IMU-derived fields, and command input. It must not receive pose, odometry, explicit velocity estimates, velocity reconstructed from privileged state, maps, SLAM state, privileged goal vectors, obstacle ground truth, simulator state, or future information.

Reward and critic may consume explicitly declared privileged simulation state. Evaluation/logging may observe it without feeding it back to the actor. Clearly labelled upper-bound baselines must be isolated from deployable policies. A conventional low-level controller may use attitude, angular rate, or other stabilization state internally after the learned velocity command; this does not authorize those fields in the actor observation.

Every interface must state tensor shape/dtype/device, units, coordinate frame, ordering, normalization, latency/timestamp semantics, validity behavior, saturation, and ownership. Frame conversions occur at named boundaries and require tests.

## 5. Correctness gates

Every order starts from an exact accepted commit and a clean branch, enumerates writable paths and permitted commands, defines the exact commit count, and names stop conditions. Every report must prove:

1. provenance and licensing are traceable to exact repo/commit/path/blob/hash evidence;
2. the diff is confined to the allowlist and the worktree is clean;
3. static/API checks and, only when authorized, runtime checks map to acceptance criteria;
4. USD prim and rotor topology, parameters, units, frames, and controller semantics are explicit;
5. actor inputs pass both positive schema assertions and negative forbidden-field assertions;
6. deterministic/reproducible commands, stdout, stderr, exit codes, environment identity, and raw-log index are retained;
7. failures, conflicting authorities, and limitations are reported without hiding or guessing.

The Commander reviews the base-to-HEAD diff and evidence before acceptance. A green test does not waive a provenance, actor-contract, path-scope, or commit-shape failure.

## 6. Training and mutation lock

Bootstrap and M1A authorize no simulator execution, asset conversion, controller/MID360 execution, PPO/reward/governor/anchor/ICS work, frozen-runtime changes, or training. Later milestones may run only the bounded checks named in their order. A 128k, 1M, or other scale run is forbidden through M8 qualification and remains forbidden afterward until separately ordered. Merge and push are outside the Commander/Executor loop unless the user separately authorizes them.

Changing this architecture lock requires a dedicated Commander governance order and commit. An implementation order cannot amend it implicitly.
