# M7-019 unsafe-command-toward-wall B0 plan

## Goal

Qualify only the exact `unsafe_command_toward_wall` / B0 / seed 7019 scenario
as a command-direction and migration-wiring witness, while preserving the five
previously accepted scenario facts and the six-pair READY/executable catalog.

## Sources of truth

- Commander Order M7-019 Revision 2
- original order base `4d12c7ed2d7e5a2519bc61af002025cd2fe9fdb7`
- retained D0001/D0002 product history and blocked reviews
- corrective D0003 product commit
  `566ec74bb35a9dbcdfb72b1fa1be8cfacf9454af`
- fresh D0003 raw evidence and frozen M2--M7 contracts

## Implementation and qualification

1. Retain the exact two-environment/eight-row positive body-x schedule and the
   bounded x=6 m wall Mesh plus separate PlaneMesh.
2. Use the twentieth of 20 exact controller-private snapshots per row,
   explicitly normalize its float32 quaternion, and independently apply
   Isaac Lab `quat_apply` to the staged body command.
3. Require bit-exact staged body-command and normalized world-command
   identity, positive world-x approach, and positive signed wall distance in
   all 16 cells.
4. Require the exact wall topology, 26-entry inventory, zero forbidden
   geometry, nonzero wall hits and acquired -x normals in all 16 cells.
5. Compare two fresh complete bundles exactly and reject all 17 mutations;
   use a one-row cyclic command rotation because the schedule is palindromic.
6. Requalify two fresh processes and unchanged comparators for clutter v2,
   corridor v3, corner v4, flat-wall v5 and open-field v6, then run the ordered
   pure, historical, CUDA and integrity regressions.

## Non-goals and stop conditions

This is not an avoidance, braking, safety-effectiveness, ranking or performance
claim. No governor/ICS response, B1--B8, doorway, dynamic obstacle, real-robot,
training, M8 or scale work is included. Stop on any command, direction, wall,
scenario, locked-source, package, frozen-state or history-shape drift.
