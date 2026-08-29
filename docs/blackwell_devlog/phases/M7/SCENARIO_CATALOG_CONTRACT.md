# M7-013 Canonical Evaluation Scenario Catalog Contract

## Public direct-module API

```text
scenario_registry() -> tuple[ScenarioSpec, ...]
resolve_scenario(scenario_id) -> ScenarioSpec
select_scenario(*, scenario_id, seed) -> ScenarioSelection
build_scenario_catalog() -> ScenarioCatalog
bind_scenario_run_plan(*, selection, manifest) -> ScenarioRunPlan
```

The module is not exported by aggregate `instinctlab.evaluation`. It imports
only standard-library record/enum/JSON/hash support and the accepted public
baseline manifest/platform/sensor types. It imports no metric implementation,
Torch, simulator, environment, actor, collector, generator or training module.

## Canonical rows

| Order | ID | Domain | Requirement | Current readiness |
| ---: | --- | --- | --- | --- |
| 1 | `corner` | simulator | required | NOT_READY |
| 2 | `flat_wall` | simulator | required | NOT_READY |
| 3 | `corridor_tunnel` | simulator | required | NOT_READY |
| 4 | `open_field` | simulator | required | NOT_READY |
| 5 | `cluttered_indoor` | simulator | required | NOT_READY |
| 6 | `unsafe_command_toward_wall` | simulator | required | NOT_READY |
| 7 | `gap_doorway` | simulator | required | NOT_READY |
| 8 | `dynamic_obstacle` | simulator | optional when available | NOT_READY |
| 9 | `real_robot_observable_holding` | real robot | required | NOT_READY |
| 10 | `real_robot_unsafe_command_approach` | real robot | required | NOT_READY |

Every row has absent provider/evidence SHA-256 and one exact nonempty reason.
There is no registry construction parameter or function that accepts readiness
or evidence, and no row is executable. A standalone future READY evidence
record cannot modify the fixed private registry or canonical catalog.

## Privacy, log and metric obligations

Privileged scenario generation may use pose, velocity, obstacle state, mesh,
ray hits, surface normals, odometry, map and ground truth. Those fields remain
scenario-generation/evaluation-only. The sole forwarded actor field is
`commanded_body_velocity_m_s`.

Required identity logs preserve product commit, `TASLAB_UAV`, `Livox MID360`,
baseline ID, scenario ID, seed, actor audit status, scenario readiness/provider/
evidence/reason, execution domain and requirement class. Every row names the
ten accepted M7 metric obligations without formulas, values, masks, thresholds
or rankings. The two real-robot rows additionally name the later-unavailable
`real_robot_repeatability` obligation.

## Selection and manifest binding

A selection requires an exact canonical ID and exact nonnegative integer seed;
booleans, coercion, aliases, case changes, whitespace and unknown IDs reject.
Binding requires exact `ScenarioSelection` and `BaselineRunManifest` types,
reparses the manifest's canonical JSON, owns the reparsed result, and requires
exact scenario ID/seed plus platform/sensor equality. It preserves the
manifest's product/baseline/audit/execution identities and never changes its
bytes or creates READY, EXECUTED or SKIPPED state.

The resulting `m7.scenario-run-plan.v1` JSON reports the catalog and manifest
SHA-256, exact identity/readiness/obligation metadata and `executable=false`.
It has no callback or execution method.

## Canonical serialization

`m7.scenario-catalog.v1` and `m7.scenario-run-plan.v1` use fixed key and row
order, exact ASCII canonical JSON, `allow_nan=False`, duplicate-key rejection,
exact round trips and SHA-256 over the canonical bytes. Missing, extra,
reordered, changed or private fields reject. Independent processes must produce
identical decoded content and identical catalog/run-plan hashes.

No scenario fixture, command schedule, seed campaign, reset behavior, simulator,
real-robot run, baseline comparison, performance threshold, training, M8 or
scale work is included.
