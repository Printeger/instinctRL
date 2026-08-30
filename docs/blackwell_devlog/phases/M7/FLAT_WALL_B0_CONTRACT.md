# Exact Flat-Wall B0 Contract

## Fixture and catalog

- Schema: `m7.flat-wall-b0-qualification.v1`
- Fixture: `m7-015-flat-wall-b0-two-env-eight-step-v1`
- Provider: `instinctlab.evaluation.scenario_flat_wall.FlatWallScenarioFixture`
- Fixture source SHA-256:
  `ed170f337abf9cd2aff288b54473b9404824be7885b4c3fa61e75484de88bdcf`
- Exact READY/executable pairs in canonical order: `flat_wall@7015`,
  `open_field@7014`; both are B0/PASS/READY with the accepted eight capability
  bindings. The remaining eight rows are NOT_READY and `dynamic_obstacle` is
  the sole optional row.
- Catalog SHA-256:
  `581e894f108315306a7ed818323de3eb30fbb9d6b53a4555d229056eb492036e`

The actor-facing field remains exactly `commanded_body_velocity_m_s`. Pose,
velocity, mesh, ray hits, normals and other scenario/evaluation facts never
select or modify actor commands.

## Exact geometry

The accepted z=0 plane and RayCaster path
`('/World/MID360StaticScene',)` remain unchanged. The sole added scenario prim
is `/World/MID360StaticScene/FlatWallMesh`, an unsubdivided CollisionAPI Mesh:

```text
points m: ((6,-42,0), (6,-42,8), (6,42,8), (6,42,0))
faceVertexCounts: (3,3)
faceVertexIndices: (0,1,2,0,2,3)
authored normal: (-1,0,0)
classification tolerance: 1e-5 m
```

The dedicated runner authors this mesh before B0 environment and RayCaster
construction. Because the accepted environment subsequently requests the same
root, the runner permits that exact Xform request once, counts it, and restores
the original factory immediately. No production runtime source is patched.

## Runtime and evidence

Each process uses two environments, eight prescribed rows, decimation 20,
160 physics/lifecycle/wrench calls, 320 environment samples, 21,240 rays per
environment per row and 339,840 total ray samples. The pre-motion inventory is
the accepted 25 entries plus exactly `FlatWallMesh`, for 26 entries and an
empty forbidden set.

Every valid same-state static hit is classified exactly once as bounded plane
or wall. Every row/environment must contain at least one wall hit. Acquired wall
normals equal -x within `1e-5`; acquired plane normals, when present, equal +z.
Per-row/environment wall, plane and valid-static counts are canonical
qualification evidence and must compare exactly across independent processes.

## Claim boundary

The qualification is a strict wiring, geometry visibility and scenario
identity record. It contains no threshold, ranking, safety, clearance,
performance, baseline-comparison, training or policy-quality claim.
