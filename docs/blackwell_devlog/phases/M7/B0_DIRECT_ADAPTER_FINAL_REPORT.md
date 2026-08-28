# M7-007 B0 Direct Adapter Final Report

## Outcome

The functional commit `bf92c054378954cb14635c3cec2af93c3e62364c`
adds one pure, stateless B0 adapter. Its public function accepts exactly body-
frame commanded velocity, a separately named controller-private scalar-first
quaternion, and immutable velocity-limit configuration. It returns distinct
owned detached body/world commands with no learned action, correction,
smoothing, history, anchor, ICS, hidden state, alias, or fallback.

## Source and capability binding

| Capability | Provider | Exact source SHA-256 |
| --- | --- | --- |
| `m5.b0_direct_velocity` | `governor.b0_governor` | `cd5cf6c73578bc69dc5f696ac6d294d9d4c8db0baaf12265e6715646554b7197` |
| `m5.b0_body_to_world` | `command_adapter.rotate_body_to_world` | `a79ea8d88fb0391762a9efcebbfa783d5287ee1393d3b6e2757812032c32991d` |
| `m7.b0_direct_adapter` | `baseline_b0.execute_b0_direct_velocity` | `5fa2b8d746eb231a1bb12f331fe1a76e8f988005e9a315e1dbf622e0bf1ed3d1` |

B0 READY/EXECUTED requires all three bindings in this order. Ten persisted
old/missing/extra/reordered/duplicate/wrong/B1/alias cases reject. B1-B8
registry source and readiness remain unchanged.

## Correctness gates

- Exact command-only actor surface and controller-private quaternion: **PASS**.
- Accepted norm/body and scalar-first world rotation parity: **PASS**.
- Float32/64 numeric/rotation/batch/direction oracles: **PASS**.
- Strict type/shape/dtype/device/finite/gradient/unit/config rejection:
  **PASS**.
- Input nonmutation, distinct storage, detached ownership, deterministic
  repeat, two-process byte equality, stateless failure atomicity: **PASS**.
- Exact three-capability binding and no fallback; B1-B8 preserved: **PASS**.
- Import/signature/source hashes, frozen packages/checkouts, functional
  allowlist, diff hygiene and clean state: **PASS**.
- Focused/M7/M2/relevant M3-M6 pure tests: **PASS** (1,356 pytest passes,
  zero skips, 66 additional subtests).

## Scope boundary

This Order adds no environment or low-level-controller wiring and executes no
baseline or scenario. No actor, observation, MID360, IMU, anchor, ICS,
pipeline, task, reward, critic, PPO, rollout, checkpoint, evaluation runtime,
metrics, simulator, or frozen implementation changed.

## Stop declaration

No environment/runtime integration, baseline execution, scenario, CUDA,
Isaac Lab/Isaac Sim launch, training, M8, 128k/1M/scale work, package/frozen-
tree/Git-configuration mutation, history rewrite, merge, or push occurred.

`NO NEXT-PHASE WORK WAS STARTED.`

`AWAITING COMMANDER REVIEW.`
