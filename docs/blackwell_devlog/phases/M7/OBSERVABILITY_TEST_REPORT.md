# Exact Observability Test Report

## Accepted results

| Layer | Result |
| --- | --- |
| Focused exact-observability contract | PASS — 42 tests |
| Full M7 pure plus relevant M3-M6 pure replay, pass 1 | PASS — 327 tests and 33 subtests |
| Full M7 pure plus relevant M3-M6 pure replay, pass 2 | PASS — 327 tests and 33 subtests |
| M2 static checks | PASS — 10 tests |
| Static source/import/name audit | PASS — 23 consumer modules, no reverse dependency or surface-normal field |
| Existing accepted M7 integrity | PASS — metrics, audit, runtime, and two accepted contract tests byte-identical to the base |
| Exact semantic inspector, two processes | PASS — byte-identical summaries |
| Real `HistoryActor` isolation/discrimination probe | PASS |

The focused suite covers float32 and float64; open field, wall, corridor,
isotropic corner, anisotropic full rank, and batched mixed validity; all result
fields, shapes, dtypes, device, detachment, finiteness, and ownership; rank
tolerance; repeated singular values; canonical sign ties; sign/ray invariance;
configuration boundaries; malformed type/order/shape/dtype/device/value;
non-unit normals; invalid nonzero placeholders; views/shared storage; missing
and extra arguments; output mutation isolation; exports; recursive imports;
actor isolation; and M7-002 `UNAVAILABLE` preservation.

Hand-computed expectations are rank 0 for open field, rank 1 with no unique
null direction for a wall, rank 2 with the tunnel axis for a corridor, rank 3
and condition one with no unique weakest direction for an isotropic corner,
and singular values `[1,sqrt(2),sqrt(3)]`, condition `sqrt(3)`, and the positive
x weak direction for the anisotropic fixture.

## Retained development results

- The default shell Python lacked pytest, so collection was NOT RUN (exit 1);
  no package was changed. All accepted runs use the frozen environment.
- An initial canonical-sign tie fixture exposed floating SVD tie sensitivity:
  39 passed and 1 failed. Tie selection was corrected to use the prescribed
  numerical tolerance, after which the focused suite passed.

No failed or unrun command is represented as PASS. No simulator or CUDA
runtime was required or run for this pure bounded core.
