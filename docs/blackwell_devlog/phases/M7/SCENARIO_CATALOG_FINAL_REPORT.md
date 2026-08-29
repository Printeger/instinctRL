# M7-013 Canonical Evaluation Scenario Catalog Final Report

## Scope completed

`instinctlab.evaluation.scenarios` now provides one pure-Python,
direct-import-only canonical identity catalog for the exact ten Handbook
scenario groups. Exact enums and frozen non-subclassable records enforce
canonical lookup, immutable selection, evidence invariants, fixed registry
truth, canonical catalog bytes and manifest-bound run-plan identity.

The first eight rows are simulator-domain and the last two are real-robot.
`dynamic_obstacle` is the sole optional-when-available row. All ten rows are
truthfully `NOT_READY`, non-executable, have no provider/evidence SHA-256 and
retain explicit unavailability reasons. No existing B0 smoke was relabelled.

## Identity and privacy boundary

The catalog records exact `TASLAB_UAV` / `Livox MID360` identity, baseline,
scenario, seed, actor audit, scenario evidence/readiness, domain and requirement
log fields. Privileged generator state remains evaluation/generation-only;
only `commanded_body_velocity_m_s` is forwarded toward the actor.

Selection requires an exact ID and exact nonnegative integer seed. Run-plan
binding reparses and owns the canonical `BaselineRunManifest`, requires exact
scenario/seed/platform/sensor equality, preserves its bytes and execution
identity, and emits `executable=false` with no callback or execution method.

## Correctness gates

| Gate | Result |
| --- | --- |
| exact ten-row order and IDs | PASS |
| eight simulator / two real-robot domains | PASS |
| nine required / sole optional dynamic row | PASS |
| ten NOT_READY / zero READY / zero executable | PASS |
| absent providers/evidence and explicit reasons | PASS |
| command-only actor forwarding and privilege isolation | PASS |
| exact log and metric-name obligations | PASS |
| exact selection and manifest ownership/binding | PASS |
| canonical catalog/run-plan JSON and SHA-256 | PASS |
| two isolated processes and complete comparison | PASS |
| five comparator mutations rejected | PASS |
| locked B0-B8/M7 runtime/package/frozen integrity | PASS |
| no CUDA/simulator/scenario/training/M8 work | PASS |

Formal evidence contains 1,688 passed tests, two standing authorized
deselections and zero skips. The two process bundles are byte-identical; the
complete comparator checked 28 mappings, 53 lists, 442 list elements and 593
scalar values.

## Source identity

New `evaluation/scenarios.py` SHA-256:
`a5d6f302a9ec48a319db83d32f8754bd5886ac73eeac79e9163e7ecc9b0262de`.

Canonical catalog SHA-256:
`3d226fa9974ba56123f5b266d9204890754973ea32ca9e24318fc986c48cecc1`.

Canonical run-plan SHA-256:
`4257d16aefc76074960da123de57475974dc7dfe1366da90c56f4d0b0b69ddca`.

## Recommendation and stop declaration

`READY_FOR_REVIEW`

`NO NEXT-PHASE WORK WAS STARTED.`

`AWAITING COMMANDER REVIEW.`
