# M7-007 B0 Direct Adapter Plan

## Goal

Add one pure command-only B0 adapter and bind B0 readiness to its exact source,
the accepted M5 norm source, and the accepted M5 scalar-first rotation source.
This is an execution primitive contract, not environment/controller runtime
integration or baseline execution.

## Source locks

| Source | Accepted SHA-256 | Use |
| --- | --- | --- |
| `instinctlab/control/governor.py` | `cd5cf6c73578bc69dc5f696ac6d294d9d4c8db0baaf12265e6715646554b7197` | `clip_vector_norm` semantics and B0 parity oracle |
| `instinctlab/control/command_adapter.py` | `a79ea8d88fb0391762a9efcebbfa783d5287ee1393d3b6e2757812032c32991d` | scalar-first `rotate_body_to_world` boundary |
| M7-006 base | `9c0144272913357ab1cc1e319b23ae58562922b4` | canonical B0-B8 registry/manifest contract |

## Writable files

Only the 12 product paths enumerated by Order M7-007 are writable. The first
commit contains the adapter, exports, B0 capability update, focused tests,
README boundary, plan, and contract. The second contains only the four result/
evidence/limitations/final records.

## Implementation

1. Validate exact tensor/config types, shapes, batch, dtype/device, finiteness,
   no-gradient status, quaternion norm, and positive finite limit before math.
2. Call the accepted norm helper on only `commanded_body_velocity_m_s`; clone
   and detach the owned body output.
3. Call the accepted scalar-first rotation with only that body output and the
   separately named controller-private quaternion; clone and detach the owned
   world output.
4. Bind B0 to three exact ordered source requirements and update canonical
   successor fixtures without touching M7-006 historical evidence.
5. Prove numeric/rotation parity, strict negatives, ownership, statelessness,
   import/signature boundaries, two-process equality, B1-B8 preservation, and
   accepted M2-M7 pure regressions.

## Stop conditions

Stop if implementation requires changing accepted M5 helpers, an unlisted
path, actor/research semantics, packages/frozen inputs, runtime integration,
CUDA/simulation, baseline/scenario execution, training, M8, scale, merge,
push, or history rewrite.
