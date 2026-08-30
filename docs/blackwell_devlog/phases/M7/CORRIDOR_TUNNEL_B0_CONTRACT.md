# Corridor/Tunnel B0 Contract

## Identity

| Field | Exact value |
| --- | --- |
| Scenario / baseline / seed | `corridor_tunnel` / `B0` / `7017` |
| Fixture | `m7-017-corridor-tunnel-b0-two-env-eight-step-v1` |
| Provider | `instinctlab.evaluation.scenario_corridor_tunnel.CorridorTunnelScenarioFixture` |
| Qualification schema | `m7.corridor-tunnel-b0-qualification.v1` |
| Environments / policy rows | `2 / 8` |
| Physics / lifecycle / wrench calls | `160 / 160 / 160` |
| Environment / ray samples | `320 / 339840` |

## Geometry

The RayCaster root remains exactly `('/World/MID360StaticScene',)`. Its first
descendant Mesh is the collision Mesh
`/World/MID360StaticScene/CorridorWallsMesh`, with the exact ordered eight
points, four triangle counts, twelve indices and subdivision `none` specified
by M7-017. The y=-6 m wall has authored/acquired inward normal `(0,1,0)` and
the y=6 m wall has `(0,-1,0)`, checked to `1e-5`.

The accepted z=0 PlaneMesh remains a separate USD stage topology witness with
1,849 points, 3,528 triangles and 10,584 indices. It is not claimed as part of
the wall-only Warp hit target. Pre-motion stage inventory is exactly 26 entries
and the forbidden geometry set is empty.

## Visibility and ownership

Every policy-row/environment cell must contain at least one bounded hit from
each wall family. Every valid hit must classify exactly once using its point
and acquired normal. Both 8x2 wall-family matrices and the valid-hit matrix are
runtime-derived and must compare exactly across two independent processes.

The bound run plan and READY manifest exist before construction. Only the
prescribed body-frame velocity command is forwarded. Geometry, pose, velocity,
ray hits, normals and evaluation state cannot select or modify actor commands.

## Claim boundary

The canonical qualification proves exact migration wiring and scenario
identity only. It makes no performance, comparison, safety, clearance,
ranking, training, campaign or M8 claim.
