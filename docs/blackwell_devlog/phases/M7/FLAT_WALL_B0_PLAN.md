# M7-015 Exact Flat-Wall B0 Plan

Order: `M7-015`, revision 0, attempt `M7-015-D0001`
Base: `eb4f265f181169fcd75ac16fb4bc4a49be441a87`
Branch: `blackwell-migration`

## Goal and boundary

Qualify exactly `flat_wall@B0/7015` and retain `open_field@B0/7014` as the
only other READY/executable pair. The work is migration-wiring and scenario
identity qualification only. It does not qualify approach safety, ICS effect,
clearance, performance, comparison or policy quality.

## Implementation

1. Add a direct-import-only immutable fixture, wall geometry/visibility
   records, canonical qualification and strict nested fail-closed validation.
2. Bind exactly two READY catalog rows and evolve open-field qualification to
   v2 without changing its schedule, geometry, runtime or metric semantics.
3. Add a dedicated runner that authors only `FlatWallMesh` before environment
   construction, preserves the pre-authored RayCaster root exactly once, and
   restores the accepted prim factory immediately after construction.
4. Traverse and classify the full stage before motion; independently inspect
   plane/wall USD topology and classify accepted live hits and acquired normals.
5. Run two flat-wall and two open-field processes, complete comparators,
   repeated pure/static suites and the accepted CUDA regressions.
6. Bind evidence and deliver exactly two nominal commits.

## Stop conditions

Stop on baseline/branch/status drift, an unlisted-path requirement, inability
to author before RayCaster construction without a locked-source edit,
unclassifiable wall/plane hits, open-field drift, frozen/package drift, or loss
of the exact commit/evidence shape. No M8, training, B1–B8 or other Handbook
scenario work is permitted.
