# M7-009 B0 TASLAB Controller Handoff Contract

## Public API

```text
execute_ready_b0_controller(
    manifest: BaselineRunManifest,
    commanded_body_velocity_m_s: Tensor[N,3],
    controller_private_root_state_w: Tensor[N,13],
    config: B0DirectConfig,
) -> B0ControllerExecutionEnvelope
```

There are no defaults, aliases, coercions, provider overrides, or fallbacks.
Command is the sole actor surface. Root position, scalar-first quaternion,
world velocity, and world angular velocity are conventional-controller private.

## Validation and ownership

Inputs must be exact types, nonempty matching rank-two batches, float32 or
float64 on one device, finite, detached/no-gradient, and mutually nonaliasing.
The root quaternion slice `[3:7]` must be unit length within `1e-5`. All bridge-
owned gates finish before either accepted function. Owned command, root, and
quaternion snapshots isolate the caller.

## Exact call chain

```text
B0 = execute_ready_b0(manifest, command_snapshot, quaternion_snapshot, config)
Lee = taslab_velocity_control(root_snapshot, B0.result.issued_world_velocity_m_s)
```

Each function is called exactly once and in order on success. B0 rejection
prevents Lee. The exact returned world tensor object is the Lee target; there
is no recomputation, learned action, correction, smoothing, anchor, ICS, or
fallback. The frozen result contains the exact accepted B0 and Lee records.

## Capability five

| Name | Provider | Source SHA-256 |
| --- | --- | --- |
| `m7.b0_taslab_controller_handoff` | `instinctlab.evaluation.baseline_controller_execution.execute_ready_b0_controller` | `72d104850f8c8fede59998f14661b5b7ea7cd2c113ec11218fe232337732ab52` |

The preceding four accepted bindings remain ordered and exact. B1-B8 remain
unchanged and non-executable.

## Bounded smoke

Each independent runner launches `AppLauncher(headless=True,
enable_cameras=False)`, one TASLAB articulation on `cuda:0`, four fixed reset/
constant-command fixtures, 64 physics steps per fixture, and 256 total steps.
Only the handoff Lee rotor coordinates enter the accepted wrench adapter.
Complete root/command/B0/Lee/rotor/wrench/manifest/reset/step records must be
finite, topology-exact, domain-bounded, reset-replay exact, and equal across
processes. No environment, MID360, actor, pipeline, anchor, ICS, task, reward,
collector, PPO, or training object is created.
