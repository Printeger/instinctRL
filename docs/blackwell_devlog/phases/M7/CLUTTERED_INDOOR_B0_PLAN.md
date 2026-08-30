# M7-018 cluttered-indoor B0 plan

## Goal

Qualify only the exact Handbook `cluttered_indoor` B0 scenario at seed 7018,
then preserve the four previously accepted scenario runtime facts while the
catalog advances to five READY/executable identities.

## Sources of truth

- Commander order `M7-018`
- product base `ae4786eb343698d2ec940201fb2a8e48eca66c0c`
- accepted M7-017 records and raw evidence
- locked M2--M7 runtime, MID360, RayCaster and actor contracts

## Implementation

1. Add one direct-import-only fixture, pure contract, CUDA runner and exact
   comparator for `cluttered_indoor`.
2. Author one selected collision Mesh at
   `/World/MID360StaticScene/ClutterPanelsMesh` with four disconnected bounded
   panels and preserve the separate PlaneMesh.
3. Require all four acquired inward-normal panel families in all 16
   policy-row/environment cells and exclusive valid-hit membership.
4. Advance the catalog to the five ordered exact B0 pairs and evolve only the
   qualification identities of the four earlier fixtures.
5. Run two fresh CUDA processes per scenario, all five comparators, repeated
   pure/regression suites and frozen-state checks.

## Non-goals and stop conditions

No closed-room claim, threshold, ranking, learned-policy claim, B1--B8 work,
training, campaign, M8 work or actor-private input is permitted. Stop on any
missing panel family, selected-mesh ambiguity, runtime drift, locked-path
change, package drift or evidence-shape failure.
