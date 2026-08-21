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

## Attempt and Git-hygiene result

Revision-0 attempt `M1B-001-D0001` remains an immutable failed attempt: its
unqualified canonical `git diff --check` exited 2 because it combined authored
EOF findings with byte-preserved source-YAML whitespace.

Revision-1 attempt `M1B-001-D0002` removes only the authorized authored EOF
findings and applies the Commander-defined composite gate. The unfiltered
base-to-HEAD output remains visible and contains only the eight permitted
`taslab_uav.yaml` source-owned diagnostics. A second check excluding only the
three byte-frozen copied artifacts exits 0 with no diagnostics. No copied byte,
M1B semantic, validator, or prior commit is changed.

## Correctness gates

- Byte-preserved authorized artifacts and complete provenance: PASS
- Exact active YAML/config values without retuning or inference: PASS
- Independent new-tree contract and negative boundary scans: PASS
- Deterministic read-only USD topology and four-rotor mapping: PASS
- Revision-1 composite Git-hygiene gate: PASS
- Exactly three additive M1B commits, allowed paths, clean final tree: PASS
- Frozen runtime/source integrity and prohibited-work audit: PASS

Exact final commit identity, command exits, evidence hashes, diffs, and
integrity records are maintained in coordination report `M1B-001.md` and
attempt `M1B-001-D0002`.

## Stop declaration

`NO NEXT-PHASE WORK WAS STARTED.`

`AWAITING COMMANDER REVIEW.`
