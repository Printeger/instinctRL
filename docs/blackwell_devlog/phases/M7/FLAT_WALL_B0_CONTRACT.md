# Exact Flat-Wall B0 Contract

## Fixture and catalog

- Schema: `m7.flat-wall-b0-qualification.v1`
- Sole fixture ID: `m7-015-flat-wall-b0-two-env-eight-step-v1`
- Provider: `instinctlab.evaluation.scenario_flat_wall.FlatWallScenarioFixture`
- Fixture source SHA-256:
  `900a9147cd74cd1750d831afbba5319acef62c30f748e9599053266d83c87e6e`
- Catalog SHA-256:
  `0ff2a806ffa5452c784b60dd7a65d7f9ba4b23a2d669af8f990274cc37612769`
- Exact READY/executable pairs in canonical order: `flat_wall@7015` and
  `open_field@7014`. The remaining eight rows are NOT_READY.

The fixture evaluation provenance, canonical qualification fixture section,
and embedded B0 evaluation-record provenance all bind the same ordered
M7-015 literal. A pure regression compares each surface against an
independently written literal rather than importing the product constant as
its expected oracle.

The actor-facing field remains exactly `commanded_body_velocity_m_s`. Pose,
velocity, mesh, ray hits, normals and other scenario/evaluation facts never
select or modify actor commands.

## Preserved geometry and runtime semantics

The accepted z=0 plane and RayCaster path
`('/World/MID360StaticScene',)` remain unchanged. The sole scenario prim is
`/World/MID360StaticScene/FlatWallMesh`, an unsubdivided CollisionAPI Mesh:

```text
points m: ((6,-42,0), (6,-42,8), (6,42,8), (6,42,0))
faceVertexCounts: (3,3)
faceVertexIndices: (0,1,2,0,2,3)
authored normal: (-1,0,0)
classification tolerance: 1e-5 m
```

Both reviewed D0001 runners and comparators are byte-identical. No production
runtime source changed. Each process uses two environments, eight prescribed
rows, decimation 20, 160 physics/lifecycle/wrench calls, 320 environment
samples, and 339,840 total ray samples. The flat-wall inventory remains exactly
26 entries with an empty forbidden set; open field remains 25 and empty.

Every valid same-state static hit is classified exactly once as bounded plane
or wall. Every row/environment must contain at least one wall hit. Acquired wall
normals equal -x within `1e-5`; acquired plane normals, when present, equal +z.
Wall, plane and valid-static hit counts remain runtime-derived canonical
evidence. They compare exactly across fresh processes but are not hard-coded as
an offline trajectory oracle.

## Claim boundary

The qualification is a strict wiring, geometry visibility and scenario
identity record. It contains no threshold, ranking, safety, clearance,
performance, baseline-comparison, training or policy-quality claim.
