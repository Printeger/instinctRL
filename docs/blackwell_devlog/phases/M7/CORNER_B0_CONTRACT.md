# Exact single-mesh corner B0 contract

## Identity

| Field | Exact value |
| --- | --- |
| Scenario / baseline / seed | `corner` / `B0` / `7016` |
| Fixture | `m7-016-corner-b0-two-env-eight-step-v1` |
| Provider | `instinctlab.evaluation.scenario_corner.CornerScenarioFixture` |
| Qualification schema | `m7.corner-b0-qualification.v1` |
| Environments / policy rows / decimation | `2 / 8 / 20` |
| Physics / lifecycle / wrench calls | `160 / 160 / 160` |
| Environment samples / rays | `320 / 339840` |

## Geometry

`/World/MID360StaticScene/CornerWallsMesh` is the first selected descendant
Mesh. Its eight ordered points, four triangle counts and twelve indices are
exactly those in Order Revision 1. Faces 0-1 have acquired normal `(-1,0,0)`;
faces 2-3 have acquired normal `(0,-1,0)`. The seam duplicates point indices.
CollisionAPI is present and subdivision is `none`.

The unchanged `/World/MID360StaticScene/PlaneMesh` remains a separate exact
1,849-point, 3,528-triangle, 10,584-index z=0 grid. It is verified by complete
stage traversal, not claimed as a hit in the wall-only Warp scan. The complete
inventory is exactly the accepted open-field 25 entries plus
`CornerWallsMesh`, hence 26 entries and no forbidden geometry.

## Visibility and information boundary

Every valid selected-mesh hit classifies exactly once using both its acquired
point and acquired normal. Each of eight rows and both environments must have
at least one x-wall and one y-wall hit within `1e-5`. Counts are live-derived
and never embedded as source constants.

The catalog READY/executable pairs are exactly `corner@7016`,
`flat_wall@7015`, and `open_field@7014`. Scenario generation forwards only
`commanded_body_velocity_m_s`; pose, velocity, obstacle state, mesh, ray hits,
surface normals, odometry, map and ground truth remain evaluation-only and do
not enter the actor.
