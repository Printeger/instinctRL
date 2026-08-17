# Blackwell Migration Governance

These instructions apply to the entire repository.

## Authority and roles

- The Commander is the sole order writer and approver. The Commander may create governance and coordination records, issue one bounded order at a time, accept or reject an Executor report, and authorize the next milestone only after the current one passes.
- The Executor is the sole implementation writer on `blackwell-migration`. It may change only paths explicitly allowed by the current order and must produce exactly the commits required by that order.
- No agent may self-approve, expand an order, combine milestones, merge, push, or modify Git configuration.
- `/home/mint/rl_dev/instinctRL` is the formal source checkout and is read-only for this migration. Work only in `/home/mint/rl_dev/worktrees/instinctRL-blackwell`.
- `/home/mint/rl_dev/instinctRL_coordination` is a separate coordination repository. It is not product source and must never be copied into this repository.

## Immutable inputs and directory boundary

- Treat the source baseline and all qualified OmniDrones, NavRL, Isaac Lab, Isaac Sim, CUDA, driver, asset, and checkpoint baselines as read-only evidence. Never patch them in place.
- `isaac-training/`, `ros1/`, `ros2/`, existing root scripts, and existing `docs/instinctRL_devlog/` are legacy evidence/runtime. They may be inspected but not changed during Blackwell migration unless a later order explicitly authorizes a narrowly enumerated exception.
- `isaaclab-training/` is the reserved new-runtime root. It must not be created before an approved M1B order. New implementation belongs there; legacy implementation must not be silently repurposed as the new runtime.
- Blackwell records belong under `docs/blackwell_devlog/`. An order must enumerate every writable file. Unlisted paths are forbidden.

## Ordered milestones

Execute exactly one accepted milestone at a time, in this order:

1. M1 — platform provenance and migration design: M1A records TASLAB_UAV provenance only; M1B may migrate the platform only after M1A acceptance.
2. M2 — minimal Isaac Lab runtime and deterministic platform smoke.
3. M3 — MID360 sensor migration and geometry/ray-contract qualification.
4. M4 — actor observation/history schema and hard information-contract audit.
5. M5 — controller/governor/anchor/ICS integration under the actor boundary.
6. M6 — reward, PPO, and bounded training-pipeline integration.
7. M7 — evaluation, logging, ablation, and input/audit qualification.
8. M8 — Blackwell acceptance, reproducibility closeout, and only then separately authorized scale training.

Subphases do not permit skipping a milestone. A failed or unaccepted report leaves the next milestone unauthorized.

## Hard actor contract

The deployed actor may consume only explicitly approved onboard measurement history, validity/reliability information, timestamps/age, IMU-derived fields approved by an order, and the commanded velocity. Pose, odometry, explicit or reconstructed velocity estimates, maps, SLAM state, goal-relative privileged state, obstacle ground truth, and privileged simulator state are forbidden actor inputs. Privileged state may be used only by reward, critic, evaluation, logging, or clearly labelled upper-bound baselines. Conventional low-level flight-control stabilization may use vehicle state internally, but that state must not enter the learned actor.

## Order and correctness gates

Before writing, verify the current order ID, `BASE_COMMIT`, branch, clean status, allowed paths, allowed commands, required commit count, and stop conditions. Stop on any mismatch.

Every Executor delivery must provide:

- the resulting commit SHA and parent/base relationship;
- a base-to-HEAD name-status and stat diff, with an allowed-path ledger;
- the exact commands run with stdout, stderr, exit code, and a raw-log index;
- tests/checks mapped to acceptance criteria;
- known limitations, unresolved provenance, and an explicit statement that prohibited work was not performed.

Correctness requires provenance integrity, API/runtime compatibility, deterministic/static checks appropriate to the milestone, actor-contract compliance, explicit units/frames/prim topology, and a clean worktree. Missing evidence is a failure, not permission to infer.

## Standing prohibitions

Unless a later accepted order explicitly narrows and lifts one of these restrictions, do not run simulation, controllers, MID360, PPO, reward integration, governor/anchor/ICS changes, training, evaluation, asset conversion, or frozen-runtime modification. Do not start 128k, 1M, or any scale-training run. Never merge or push. M1A is documentation-only and may use only read-only Git/hash/file/stat/search/static-diff commands.
