# M1B Final Report

## Scope completed

The three authorized TASLAB artifacts were copied from exact base Git objects
without byte changes. Provenance and bounded user authorization are recorded.
A standard-library-only declarative contract preserves all active vehicle,
rotor, and Lee-gain values, their order and units, the proven USD rotor prim
paths, and the runtime semantics that remain unresolved.

Ten persisted static checks pass twice. The persisted read-only USD inspector
opens both a temporary materialization of the exact source object and the
target twice. Source and target normalized inventories are identical and both
runs produce identical semantic JSON. The stage has default prim
`/taslab_uav`, Z up-axis, one meter per unit, 28 prims, five rigid bodies, four
rotor candidates, and four base-link-to-rotor revolute joints. The ordered
four-rotor transform mapping is proven using unique centroid-relative XY
matches cross-checked by prim index.

## Correctness gates

- Byte-preserved authorized artifacts and complete provenance: PASS
- Exact active YAML/config values without retuning or inference: PASS
- Independent new-tree contract and negative boundary scans: PASS
- Deterministic read-only USD topology and four-rotor mapping: PASS
- Exactly two prescribed commits, allowed paths, clean final tree: PASS
- Frozen runtime/source integrity and prohibited-work audit: PASS

Exact commit identities, command exits, raw evidence hashes, diffs, and
integrity records are maintained in coordination report `M1B-001.md` and
attempt `M1B-001-D0001`.

## Stop declaration

`NO NEXT-PHASE WORK WAS STARTED.`

`AWAITING COMMANDER REVIEW.`

