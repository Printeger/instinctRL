# M7-007 B0 Direct Adapter Contract

## Public API

```text
execute_b0_direct_velocity(
    commanded_body_velocity_m_s: Tensor[N,3],
    controller_private_quaternion_wxyz: Tensor[N,4],
    config: B0DirectConfig,
) -> B0DirectResult
```

There are no defaults and no other public function inputs. The first tensor is
the sole actor-approved surface. It is body-frame commanded linear velocity in
metres per second. The scalar-first unit quaternion is controller-private,
maps body vectors to world vectors, and is never part of an actor schema or
learned mapping. `B0DirectConfig.velocity_limit_m_s` is an immutable positive
finite scalar.

## Input contract

Both inputs are exact `torch.Tensor` objects, detached and gradient-free,
nonempty rank-2 batches with shapes `[N,3]` and `[N,4]`. They share one device
and exact float32 or float64 dtype and contain only finite values. Quaternion
norm error may not exceed the accepted M5 tolerance `1e-5`. Invalid input is
rejected before output/state mutation; the adapter owns no state or cache.

## Output contract

```text
issued_body_velocity_m_s = clip_vector_norm(
    commanded_body_velocity_m_s, velocity_limit_m_s
)
issued_world_velocity_m_s = rotate_body_to_world(
    issued_body_velocity_m_s, controller_private_quaternion_wxyz
)
```

The body result is exact Euclidean norm clipping: zero, below-limit, and exact-
limit commands are value-unchanged; over-limit direction is preserved. There
is no learned action, correction, smoothing, attenuation, history, anchor,
ICS, default selection, alias, or fallback. Both returned tensors preserve
input dtype/device, are finite and detached, own distinct storage, and share
no storage with either input or each other.

## Capability chain

B0 is READY only with these exact ordered requirements:

| Name | Provider | Source SHA-256 |
| --- | --- | --- |
| `m5.b0_direct_velocity` | `instinctlab.control.governor.b0_governor` | `cd5cf6c73578bc69dc5f696ac6d294d9d4c8db0baaf12265e6715646554b7197` |
| `m5.b0_body_to_world` | `instinctlab.control.command_adapter.rotate_body_to_world` | `a79ea8d88fb0391762a9efcebbfa783d5287ee1393d3b6e2757812032c32991d` |
| `m7.b0_direct_adapter` | `instinctlab.control.baseline_b0.execute_b0_direct_velocity` | `5fa2b8d746eb231a1bb12f331fe1a76e8f988005e9a315e1dbf622e0bf1ed3d1` |

Old one-binding, missing, extra, reordered, duplicated, or wrong provider/hash
claims reject without fallback. B1-B8 values/readiness remain unchanged.

## Boundary

This module is pure PyTorch. It is not wired into `TaslabDirectEnv`, the M7
runtime collector, a low-level controller, any scenario, Isaac Lab/Isaac Sim,
or training. It makes no performance claim and executes no baseline run.
