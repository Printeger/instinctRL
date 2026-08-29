# M7-009 B0 TASLAB Controller Handoff Plan

## Goal

Add one exact stateless evaluation bridge from a canonical B0 READY manifest
and command-only actor surface to the accepted B0 execution boundary and then
the conventional TASLAB Lee controller. Qualify source-to-wrench wiring with
two independent, bounded, one-articulation headless CUDA processes.

## Source locks

| Source | SHA-256 | Use |
| --- | --- | --- |
| `evaluation/baseline_execution.py` | `db7b73ee7e9c4b39689814530363bafb6a1ef48c2513a0208c0ff29c7073c108` | exact manifest-gated B0 execution |
| `control/baseline_b0.py` | `5fa2b8d746eb231a1bb12f331fe1a76e8f988005e9a315e1dbf622e0bf1ed3d1` | exact config/result contract |
| `control/lee_velocity_controller.py` | `5fbd9ea46141a8c9e15a0d67ed04c95ebcbe4a28a7760789f8bdce56f63c54e2` | exact conventional controller |
| `evaluation/baseline_controller_execution.py` | `72d104850f8c8fede59998f14661b5b7ea7cd2c113ec11218fe232337732ab52` | fifth exact B0 capability |

## Implementation

1. Validate exact manifest/config/tensor types and controller-private root
   shape, batch, dtype/device, ownership, gradient, finite, and quaternion
   gates before accepted calls.
2. Clone command/root/quaternion; call `execute_ready_b0` once, then pass its
   exact world tensor to `taslab_velocity_control` once.
3. Return one frozen envelope; keep root state out of actor schema, command,
   manifest, and B0 adapter inputs except the private quaternion snapshot.
4. Bind the exact bridge bytes as B0 capability five and preserve B1-B8.
5. Run four constant-command/reset fixtures for 64 steps each in two fresh
   headless CUDA processes and compare complete normalized records exactly.

## Boundaries and stop conditions

Only the 16 M7-009 paths are writable. Stop on actor leakage, need for accepted
control/environment/runtime changes, package/frozen mutation, GUI/cameras,
prohibited object creation, more than one articulation/app or 256 steps per
process, training/M8/scale work, merge, push, or history rewrite.
