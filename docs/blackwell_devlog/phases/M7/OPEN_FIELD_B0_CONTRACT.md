# Exact Open-Field B0 Contract

`instinctlab.evaluation.scenario_open_field` is direct-import-only and not re-exported by the aggregate evaluation package. Its exact fixture identity is `open_field` / `B0` / seed 7014 / two environments / eight policy rows / decimation 20. The provider is `instinctlab.evaluation.scenario_open_field.OpenFieldScenarioFixture`; its committed source SHA-256 is `5ae2a457ec7de4fcba5c32a014e8bed640c25b4cfd49d3965aedf66cacdd61a3`.

Only the exact READY B0 manifest with PASS actor audit and matching open-field selection/seed produces `executable=true`. EXECUTED, NOT_READY, mismatched seed/scenario and other identities are non-executable or rejected. No run plan exposes a callback.

The fixture owns and reparses the manifest, selection and plan and creates the accepted `B0EnvironmentConfig` and `B0RuntimeEvaluationConfig`. Its sole schedule is the eight rows recorded in `OPEN_FIELD_B0_PLAN.md`; there is no random/default/clipping alternative. Actor forwarding remains exactly `commanded_body_velocity_m_s`; geometry, pose, velocity, rays and all evaluation state remain outside the actor boundary.

Live geometry must contain the global ground at `/World/ground`, the `Xform` `/World/MID360StaticScene` and its `Mesh` `/World/MID360StaticScene/PlaneMesh`. The mesh is z=0, has x/y coordinates -42 through +42 m at 2 m spacing, 1,849 points, 3,528 triangle faces and 10,584 indices. RayCaster mesh paths are exactly `('/World/MID360StaticScene',)`.

Finalization requires the exact executable plan, live geometry evidence, finalized environment receipt and finalized `B0RuntimeEvaluationResult`. Canonical schema `m7.open-field-b0-qualification.v1` binds catalog, fixture source, manifest, plan, geometry, commands, coverage, unchanged metric facts, termination/reset counts, receipt, B0 record and derived correctness gates. No caller supplies metric values or performance claims.
