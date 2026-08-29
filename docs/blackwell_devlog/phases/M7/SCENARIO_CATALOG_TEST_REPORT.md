# M7-013 Canonical Evaluation Scenario Catalog Test Report

## Result

`PASS` — accepted formal candidate `M7-013-D0001/formal_candidates/0001`,
executed from clean functional commit
`784589e1dfd820b782ac7284b320e0d23ce7cc5a` without CUDA or simulator launch.

## Pure and static checks

| Check | Runs | Result |
| --- | ---: | --- |
| Focused scenario catalog contract | 2 × 69 | PASS |
| Accepted M7 pure suites including M7-012/M7-013 | 2 × 567 | PASS |
| Relevant M3-M6 behavior regressions | 2 × 203 | PASS |
| M2 static suite | 10 | PASS |
| Source/API/import/privacy/locked-hash audit | 1 | PASS |
| Independent catalog/run-plan audit | 1 | PASS |
| Atomic negative matrix | 24 cases | PASS |

The M3-M6 command used only the standing Order-authorized deselection
`test_m6_environment_contract.py::test_accepted_files_are_unmodified_from_order_base`;
it was deselected once in each run. Formal pytest total was 1,688 passed, two
authorized deselections and zero skipped.

## Catalog facts

| Fact | Exact result |
| --- | --- |
| ordered rows | 10 |
| simulator / real-robot | 8 / 2 |
| required / optional | 9 / 1 |
| sole optional ID | `dynamic_obstacle` |
| NOT_READY / READY | 10 / 0 |
| executable | 0 |
| provider or evidence claims | 0 |
| actor-forwarded fields | `commanded_body_velocity_m_s` only |
| simulator / real-robot metric obligations | 10 / 11 names |

Every row retains an exact nonempty unavailability reason. The real-robot rows
add only the later-unavailable `real_robot_repeatability` obligation.

## Serialization and mutation checks

Two fresh isolated Python processes emitted byte-identical bundles. Both
produced catalog SHA-256
`3d226fa9974ba56123f5b266d9204890754973ea32ca9e24318fc986c48cecc1`,
run-plan SHA-256
`4257d16aefc76074960da123de57475974dc7dfe1366da90c56f4d0b0b69ddca`
and bound manifest SHA-256
`c8fe4f8e75db767344a0adcb8a61735f20e3d2a3533f5833301d87225d88b1b0`.

The complete comparator checked 28 mappings, 53 lists, 442 list elements and
593 scalar leaves with exact type, order and value equality. Missing, extra,
reordered, changed-readiness and private-field bundles all returned nonzero.

## Integrity

The Handbook and four locked evaluation sources matched the Order hashes.
B0-B8 identity/readiness/capabilities, M7-012 collector behavior and aggregate
exports remained unchanged. Functional history, five-path allowlist, full diff,
`git diff --check`, clean start/end, frozen checkouts and Python/package
identities passed. No CUDA, simulator, scenario, real-robot, comparison
campaign, training or M8 work ran.
