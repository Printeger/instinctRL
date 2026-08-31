# M7 Gap/Doorway B0 Contract

## Identity and coverage

- Schema: `m7.gap-doorway-b0-qualification.v1`
- Fixture: `m7-020-gap-doorway-b0-two-env-eight-step-v1`
- Generator: `instinctlab.evaluation.scenario_gap_doorway`
- Provider: `instinctlab.evaluation.scenario_gap_doorway.GapDoorwayScenarioFixture`
- Scenario/baseline/seed: `gap_doorway` / `B0` / `7020`
- Coverage: 2 environments, 8 policy steps, 160 physics/lifecycle/wrench
  applications, 320 environment samples and 339,840 ray samples.

## Geometry and visibility

`/World/MID360StaticScene/DoorwayFrameMesh` is the first selected descendant
Mesh and has CollisionAPI, subdivision `none`, exact inward normal `(-1,0,0)`,
12 ordered points, six triangular faces and the ordered 18-index topology from
the Order. The strict aperture is the x=6 plane with `-2 < y < 2` and
`0 < z < 3`.

Every valid selected-frame hit belongs to exactly one deterministic family:
negative jamb (`y <= -2`), positive jamb (`y >= 2`) or lintel
(`-2 < y < 2`, `z >= 3`). Every one of the 16 cells must contain a nonzero
count from all three families and exact acquired `-x` normals. The separate
PlaneMesh remains topology evidence and is not claimed as a same-scan hit.

## Direction and privacy

The prescribed body-x magnitudes are `0.04, 0.08, 0.12, 0.16, 0.20, 0.18,
0.14, 0.10` m/s. For each cell, the runner independently verifies the live
quaternion norm within `1e-5`, normalizes it, applies `quat_apply`, proves
bit-exact equality to the issued world command, requires positive world x and
positive intersection parameter, and proves four strictly positive aperture
margins. These complete matrices are retained in both JSON summaries and PT
bundles and compare exactly across processes.

Only `commanded_body_velocity_m_s` crosses the actor boundary. Geometry,
pose, hits, normals and projection evidence remain runner/evaluation privilege.
