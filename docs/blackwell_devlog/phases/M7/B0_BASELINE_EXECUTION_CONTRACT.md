# M7-008 B0 Manifest-Gated Execution Contract

## Public API

```text
execute_ready_b0(
    manifest: BaselineRunManifest,
    commanded_body_velocity_m_s: Tensor[N,3],
    controller_private_quaternion_wxyz: Tensor[N,4],
    config: B0DirectConfig,
) -> B0ExecutionEnvelope
```

There are no defaults, aliases, coercions, provider overrides, or fallbacks.
The command is the sole actor input. The quaternion is controller-private and
is used only inside the accepted direct adapter after the actor boundary.

## Fail-closed manifest gate

The input must be the exact manifest type and must reparse from its own
canonical bytes. It must identify exact B0/direct-velocity, be deployable and
actor-clean with command-only schema, have all component toggles false, contain
the four exact ordered accepted capability bindings, record actor audit PASS,
be READY, and have no skip reason. Every manifest-level failure occurs before
adapter invocation.

## Execution and successor

After the complete manifest gate and exact tensor/config type checks,
`execute_b0_direct_velocity` is invoked exactly once. Its result is returned
without recomputation or postprocessing. The accepted builder constructs an
exact canonical EXECUTED successor preserving product commit, baseline/config,
capabilities, scenario, seed, audit, schemas, toggles, and skip reason. Only
`execution_status` changes in the payload and the canonical hash is recomputed.

`B0ExecutionEnvelope` is frozen and contains the unchanged READY manifest, the
independent EXECUTED manifest, and the adapter-owned detached nonaliasing
result. Tensor payloads are never serialized into either manifest.

## Four-capability chain

| Name | Provider | Source SHA-256 |
| --- | --- | --- |
| `m5.b0_direct_velocity` | `instinctlab.control.governor.b0_governor` | `cd5cf6c73578bc69dc5f696ac6d294d9d4c8db0baaf12265e6715646554b7197` |
| `m5.b0_body_to_world` | `instinctlab.control.command_adapter.rotate_body_to_world` | `a79ea8d88fb0391762a9efcebbfa783d5287ee1393d3b6e2757812032c32991d` |
| `m7.b0_direct_adapter` | `instinctlab.control.baseline_b0.execute_b0_direct_velocity` | `5fa2b8d746eb231a1bb12f331fe1a76e8f988005e9a315e1dbf622e0bf1ed3d1` |
| `m7.b0_manifest_execution` | `instinctlab.evaluation.baseline_execution.execute_ready_b0` | `19d910c0f7bf16994ffcc2c0a15a9efc0cbe2e2148075a1056681044e8140767` |

## Boundary

This is a pure evaluation-side contract. It does not select a baseline, wire
an environment/controller/runtime/scenario, step simulation, aggregate
metrics, or make a baseline-performance claim. B1-B8 remain non-executable.
