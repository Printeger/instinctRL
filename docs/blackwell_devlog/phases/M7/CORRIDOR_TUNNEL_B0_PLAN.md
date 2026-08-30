# Corridor/Tunnel B0 Qualification Plan

Order: `M7-017`, revision 0, attempt `M7-017-D0001`.

## Goal and boundary

Qualify only `corridor_tunnel` / B0 / seed 7017 with two environments and the
ordered eight-row command schedule. Add the exact two-wall single Mesh before
environment construction, preserve the accepted PlaneMesh as independent USD
topology, and evolve the catalog to exactly four READY pairs. No ceiling,
tunnel enclosure, unsafe approach, threshold, comparison, training or M8 work
is included.

## Sources of truth

- Commander Order `M7-017` and M7-016 approval.
- Accepted B0, environment, RayCaster, MID360 and scenario sources at
  `0e5c3c7f7034b7c4918c0bf3fe2d5c8ec6e07831`.
- Exact geometry, command, coverage, privacy and evidence requirements in the
  Order.

## Implementation

1. Add a self-contained, direct-import-only immutable corridor fixture.
2. Bind its committed source in the catalog and preserve the other six
   NOT_READY rows plus the sole optional dynamic-obstacle row.
3. Add a dedicated runner that validates the selected Warp Mesh and complete
   USD stage independently, then classifies every valid hit by point and
   acquired normal.
4. Add a complete exact comparator and pure fail-closed contract tests.
5. Run two fresh processes per READY scenario, four comparators, repeated pure
   suites and bounded CUDA regressions.
6. Deliver exactly one functional and one documentation commit.

## Stop conditions

Stop for any missing wall family in a row/environment, selected Mesh mismatch,
plane or accepted runtime drift, actor leakage, unlisted required change,
package/frozen drift, or inability to preserve the two-commit shape.
