# Corridor/Tunnel B0 Test Report

Order `M7-017`; functional commit
`53a98eed3d041d61eb6ddf00ccde12eeaa3e13b3`.

## Results

| Gate | Result | Evidence |
| --- | --- | --- |
| Syntax compilation | PASS | command `006_syntax` |
| Focused catalog plus four fixture suites, twice | PASS | `273 passed` per accepted run; commands `009`, `029` |
| Accepted M7 pure suites, twice | PASS | `771 passed` per run; commands `030`, `032` |
| Relevant M3-M6 suites, twice | PASS | `203 passed, 1 deselected, 33 subtests passed` per run; commands `031`, `033` |
| M2 static | PASS | `10 passed`; command `034` |
| Eight formal scenario CUDA processes | PASS | commands `017` through `024`; stderr empty |
| Four exact comparators | PASS | commands `025` through `028`; 16/16/13/11 mutation classes rejected |
| Learned, M7-011 and M7-012 CUDA regressions | PASS | commands `035` through `037` |

Formal repeated pytest evidence totals 2,504 passes, two accepted deselections
and 66 subtests. Eleven formal CUDA processes passed.

## Corridor runtime facts

- Selected Mesh: `/World/MID360StaticScene/CorridorWallsMesh`.
- Selected topology: 8 points, 12 indices, exact collision and subdivision.
- Complete USD inventory: 26 entries; forbidden set empty.
- Separate plane: exact 1,849-point / 3,528-triangle / 10,584-index topology.
- Both wall families: nonzero in all 16 cells; acquired +y and -y normals exact
  within `1e-5`; exclusive and total membership.
- Two formal processes: complete bundle equality and identical 8x2 matrices.

The development probe `008` returned PASS against its then-current oracle but
used the wrong mechanically derived prim name `CorridorTunnelWallsMesh`; it is
rejected as qualification evidence. The corrected exact prim identity was
retested by `010` and both formal processes. The first Git commit invocation
`014` failed before history changed because author identity was absent; `015`
used command-local identity variables without modifying Git configuration.

## Not run

B1-B8, training, scale runs, thresholds, comparisons, safety/clearance gates
and M8 were not run because they are outside this Order.
