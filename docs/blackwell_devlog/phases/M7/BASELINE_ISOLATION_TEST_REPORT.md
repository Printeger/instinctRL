# M7-006 Baseline Isolation Test Report

D0001 accepted evidence remains immutable at
`M7-006-D0001/formal_candidates/0002`. The D0002 correction evidence is
`M7-006-D0002/formal_candidates/0001`; its 55-row ledger records every command,
stream, exit, and hash.

## D0002 results

| Gate | Result | Count | Evidence command |
| --- | --- | ---: | --- |
| Exact unchanged READY/EXECUTED/NOT_READY/SKIPPED fixtures | PASS | 4 | ledger 005-009 |
| Persisted manifest negative matrix | PASS | 27 expected rejects | ledger 010-037, each actual exit 2 |
| Standard-library and reverse-import/source-hash isolation | PASS | 1 audit | ledger 038 |
| Commander exact-boolean reproduction probe | PASS | 5 malformed integer forms rejected | ledger 039, exit 0 |
| Commander registry-drift reproduction probe | PASS | 6 formerly accepted drifts rejected | ledger 040, exit 0 |
| Correction allowlist/D0001/frozen repos/critical packages | PASS | 1 audit | ledger 041 |
| Focused baseline isolation suite, pass 1 | PASS | 72 | ledger 042 |
| Focused baseline isolation suite, pass 2 | PASS | 72 | ledger 043 |
| M2 static suite | PASS | 10 | ledger 044 |
| Accepted M7 pure suites, pass 1 | PASS | 292 | ledger 045 |
| Relevant M3-M6 pure regressions, pass 1 | PASS | 204 + 33 subtests | ledger 046 |
| Accepted M7 pure suites, pass 2 | PASS | 292 | ledger 047 |
| Relevant M3-M6 pure regressions, pass 2 | PASS | 204 + 33 subtests | ledger 048 |
| Full history/correction and base allowlists/stat/diff/diff-check/clean | PASS | 7 commands | ledger 049-055 |

The formal summary records 1,146 pytest passes and zero skips; 66 regression
subtests are reported separately by pytest and are not added to that count.

## D0002 requirement mapping

- Exact JSON boolean typing: integer `1` at `deployable`, integer `0` at all
  four toggle positions, and representative float, string, null, array, and
  object substitutions are persisted in the negative matrix and covered by
  focused parametrized tests. Four exact-boolean status fixtures still pass.
- Exact canonical registry: focused tests mutate each of the 13
  `BaselineSpec` fields individually and require rejection. A separately
  reconstructed value-identical tuple passes. The Commander six-field drift
  probe also exits 0 with an empty accepted-case list.
- D0001 preservation: field order, canonical bytes, and the four valid fixture
  hashes are unchanged; D0001 summary and ledger hashes are recomputed by the
  integrity and final audits.
- Existing round-trip, mutation-isolation, duplicate/missing/extra/reordered,
  nonfinite, capability, audit, skip, alias, and no-fallback negatives remain
  active in the focused suite and persisted comparator matrix.
- Standard-library-only imports, reverse-import isolation, accepted actor/B0/
  metrics/runtime source hashes, exact six-path correction allowlist, frozen
  checkout/package identity, clean state, and diff hygiene are audited.

## Test boundaries

No CUDA device, Isaac Lab/Isaac Sim runtime, scenario, baseline adapter,
ablation, rollout, training, checkpoint, policy-quality metric, or M8 gate was
executed. No D0001 evidence file was changed.
