# M7-019 unsafe-command-toward-wall B0 test report

## Results

| Gate | Result | Evidence |
| --- | --- | --- |
| Comparator/pure-test syntax | PASS | `003_syntax_repair` |
| Unsafe pure contract before commit | PASS, 106 | `004_unsafe_pure_repair` |
| Focused seven-file suite, twice | PASS, 386 each | `009_focused_pure_1`, `010_focused_pure_2` |
| Unsafe CUDA processes | PASS, 2/2 | `011_unsafe_1`, `012_unsafe_2` |
| Unsafe complete comparator | PASS, exact equality, 17/17 mutations | `013_unsafe_compare` |
| Other five scenario CUDA processes | PASS, 10/10 | `014`--`027` excluding comparator labels |
| Clutter/corridor/corner/flat/open comparators | PASS, 20/16/16/13/11 mutations | `016`, `019`, `022`, `025`, `028` |
| Accepted M7 pure suite, twice | PASS, 884 each | `029_accepted_m7_1`, `031_accepted_m7_2` |
| Relevant M3--M6 suite, twice | PASS, 203 + 33 subtests each; 1 expected deselection | `030_m3_m6_1`, `032_m3_m6_2` |
| M2 static | PASS, 10 | `033_m2_static` |
| Learned/M7-011/M7-012 CUDA regressions | PASS, 3/3 | `034`--`036` |
| Locked/package/frozen integrity | PASS | `037_integrity_probe`, `041_accepted_kit_identity` |
| D0001/D0002/M7-018 evidence identity | PASS | `038`--`040` |

Both fresh unsafe processes recorded every gate true: exact normalized
direction, 160 controller snapshots, exact command identity, positive approach
and distance, wall topology/inventory/hits/normals, deterministic source steps,
metric identity, exact coverage, bound receipt/record, zero reset/done, actor
privacy and finite values.

## Exact matrices

Raw quaternion norms were identical across processes:

```text
[[1.0, 1.0],
 [0.9999999403953552, 0.9999999403953552],
 [1.0, 1.0],
 [0.9999999403953552, 0.9999999403953552],
 [1.0, 1.0],
 [1.0, 1.0],
 [1.0, 0.9999999403953552],
 [1.0, 1.0000001192092896]]
```

Approach component (m/s), signed wall distance (m), and wall/valid hit counts
were respectively:

```text
approach = [[0.049993306398391724, 0.049993306398391724],
 [0.09991040080785751, 0.09991040080785751],
 [0.1497538685798645, 0.1497538685798645],
 [0.1996142715215683, 0.1996142715215683],
 [0.19966226816177368, 0.1996622532606125],
 [0.14985232055187225, 0.14985232055187225],
 [0.0999692901968956, 0.0999692901968956],
 [0.04999930411577225, 0.04999930411577225]]
distance = [[3.999838352203369, 7.9998393058776855],
 [3.9978604316711426, 7.997861385345459],
 [3.9916677474975586, 7.991669178009033],
 [3.9798362255096436, 7.979838848114014],
 [3.9619405269622803, 7.961942672729492],
 [3.938434362411499, 7.938436031341553],
 [3.9106783866882324, 7.910680294036865],
 [3.880570650100708, 7.88057279586792]]
hits = valid_hits = [[8619, 5908], [8504, 5841], [8456, 5798],
 [8432, 5773], [8448, 5798], [8507, 5835], [8593, 5892],
 [8668, 5950]]
```

The exact 8x2x3 world-command matrix is retained in both fresh JSON summaries
and compared bit-for-bit in the complete PT bundles. No unrun check is
reported as PASS.

## Prior blocked attempts

D0001 failed because its independent oracle omitted mandatory tolerance-level
quaternion normalization. D0002 corrected that oracle and proved two-process
equality, then stopped because the comparator reversed a palindromic schedule
and produced a no-op mutation. D0003 uses the ordered one-row rotation; it
changes the schedule candidate and all 17 mutations are rejected.
