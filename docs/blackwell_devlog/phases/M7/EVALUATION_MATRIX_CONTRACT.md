# M7 Evaluation Availability Matrix Contract

## Identity and shape

The schema is `m7.evaluation-availability-matrix.v1`. A caller supplies one
exact lowercase 40-hex product commit. The root binds the accepted baseline and
scenario schema identities and canonical source hashes, ordered IDs, ordered
statuses, counts, 90 immutable cells and its payload SHA-256. JSON encoding is
finite canonical ASCII with no timestamp, moving ref, coordination path,
metric, threshold or default.

Baseline order is B0 through B8. Scenario order is `corner`, `flat_wall`,
`corridor_tunnel`, `open_field`, `cluttered_indoor`,
`unsafe_command_toward_wall`, `gap_doorway`, `dynamic_obstacle`,
`real_robot_observable_holding`, then
`real_robot_unsafe_command_approach`. Status order is `READY`, `NOT_READY`,
`SKIPPED`. Matrix order is baseline-major.

The accepted source identities at the Order base are:

- baseline projection SHA-256:
  `f15db3ce373656640d130434c00cb2f78322dad96f202437eb28e7fed8814af4`
- scenario projection SHA-256:
  `78dc30087cf8770985d41201ca4c7827601a07f0b138cc1dca5153e4a6c57724`
- base-product matrix SHA-256:
  `a0d1c1cdb8cedf18705a7ab4144a682b75ba259ef5e83809d85947546f540b65`

## Exact blocker map

Baseline capability blockers, in registry order, are:

| Baseline | Class | Blocker |
| --- | --- | --- |
| B0 | none | none |
| B1 | required | `m7.b1_state_velocity_upper_bound_adapter` |
| B2 | required | `m7.b2_no_history_adapter` |
| B3 | required | `m7.b3_no_anchor_adapter` |
| B4 | required | `m7.b4_no_imu_adapter` |
| B5 | required | `m7.b5_no_ics_adapter` |
| B6 | required | `m7.b6_distance_only_filter` |
| B7 | required | `m7.b7_braking_heuristic` |
| B8 | optional | `m7.b8_simplified_cbf_vo_filter` |

Scenario blockers are:

| Scenario | Class | Exact blocker |
| --- | --- | --- |
| first seven simulator rows | none | none |
| `dynamic_obstacle` | optional | `optional dynamic-obstacle scenario has no accepted fixture or evidence` |
| `real_robot_observable_holding` | required | `no accepted real-robot observable-holding access, fixture, or evidence` |
| `real_robot_unsafe_command_approach` | required | `no accepted real-robot unsafe-command-approach access, fixture, or evidence` |

Every cell concatenates its baseline blocker and scenario blocker in that order
within their respective required/optional tuples. A required blocker always
takes precedence over optional blockers. This yields B0's seven `READY`, one
dynamic `SKIPPED` and two real-robot `NOT_READY` cells; B1-B7 are all
`NOT_READY`; B8 has eight `SKIPPED` and two real-robot `NOT_READY` cells. Totals
are exactly 7 `READY`, 74 `NOT_READY` and 9 `SKIPPED`.

## Safety boundary

B1 stays nondeployable and privileged. Every other deployable row stays
actor-clean. All cells state `PASS_BEFORE_EXECUTION` as a future audit
requirement and `execution_performed=false`; no audit pass, execution, safety or
performance result is asserted.
