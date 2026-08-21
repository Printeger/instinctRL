# M3 Plan

## Understanding of goal

Record and qualify the MID360 static ray geometry, TASLAB mount, validity/serialization behavior, and deterministic minimal runtime smoke required by order M3-001.

## Source-of-truth files

- Commander order M3-001 and approved M2 baseline
- Formal read-only legacy MID360 implementation and TASLAB configuration
- Frozen Isaac Lab 2.3.2 public APIs

## Files expected to change

The M3-001 allowlisted sensor module, tests, README, and M3 phase records only.

## Files guaranteed not to change

Existing assets, M1/M2 implementation and records, legacy runtime, frozen external repositories, actor/controller/reward/PPO/training code, and package/system configuration.

## Implementation steps

Implement the deterministic pattern and contract; independently test it; execute a bounded public-API runtime smoke twice; compare every range and validity element; record results and limitations.

## Tests

Static/unit checks precede runtime smoke and deterministic comparison. M2 static and smoke checks are rerun as regressions.

## Evidence

Coordination raw logs are indexed by the repository M3 evidence record; large/full outputs remain outside Git.

## Risks

The frozen runtime may require tightly bounded initialization/update operations. No behavior outside the order will be added to work around failures.

## Stop conditions

Stop on governance mismatch, forbidden write need, runtime/package modification need, private API need, correctness failure, or exhausted bounded repair attempts.

## Revision 1 correction plan

Attempt `M3-001-D0002` preserves D0001 and adds exactly one corrective commit.
It will compose the public parent pose with the configured mount pose in one
validated helper, use only the mounted origin for ranges, and independently
check all 21,240 static-plane rays. Static transform/malformed fixtures, two
fresh CUDA processes, the unchanged comparator, and complete M2 regressions
must pass before fresh evidence is frozen.
