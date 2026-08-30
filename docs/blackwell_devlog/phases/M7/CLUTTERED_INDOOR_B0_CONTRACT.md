# M7-018 cluttered-indoor B0 contract

## Identity

- schema: `m7.cluttered-indoor-b0-qualification.v1`
- fixture: `m7-018-cluttered-indoor-b0-two-env-eight-step-v1`
- provider: `instinctlab.evaluation.scenario_cluttered_indoor.ClutteredIndoorScenarioFixture`
- seed/baseline: `7018` / `B0`
- fixture source SHA256: `0e77c42e0fb38b1f1f9531c5c7e3c9f8e9432dede13e579ebfcaffb290db83a4`

The canonical READY/executable order is `(corner,7016)`,
`(flat_wall,7015)`, `(corridor_tunnel,7017)`, `(open_field,7014)`,
`(cluttered_indoor,7018)`.

## Geometry and observation

The selected Warp mesh is exactly
`/World/MID360StaticScene/ClutterPanelsMesh`: one collision Mesh, 16 ordered
points, eight triangular faces and 24 indices. Its four disconnected panels
are bounded at x=+6, x=-6, y=+6 and y=-6 m, with acquired inward normals -x,
+x, -y and +y respectively. Every one of the 16 policy-row/environment cells
must contain at least one hit from every panel family. Each valid clutter hit
must classify into exactly one family using its acquired point and normal.

The unchanged z=0 PlaneMesh is verified separately. The complete inventory is
exactly 26 entries, with zero forbidden geometry. Neither the plane nor the
four disconnected panels imply a same-scan plane hit or a closed room.

## Runtime and privacy

The run is exactly two environments, eight policy rows, 160 physics steps,
160 lifecycle calls, 160 wrench applications, 320 environment samples and
339,840 ray samples. Commands are the prescribed body-frame schedule. Actor
and training calls remain zero; pose, velocity, ray hits, normals, geometry and
evaluation state remain evaluation-only. Qualification claims migration
wiring and scenario identity only, never performance or comparison.
