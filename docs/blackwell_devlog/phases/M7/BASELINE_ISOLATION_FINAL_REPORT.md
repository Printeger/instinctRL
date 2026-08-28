# M7-006 Baseline Isolation Final Report

## Scope completed

D0002 adds one bounded correction to the accepted D0001 baseline-isolation
layer. Manifest parsing now rejects non-boolean JSON values in all five locked
boolean positions before equality comparison. Registry validation now rejects
drift in every `BaselineSpec` field while accepting an independently
constructed, value-identical registry. No canonical registry or valid manifest
value changed.

The preserved D0001 functional commit is
`93602e05238cf207bfefea235cbaadef35c957f6`; the rework starts at
`4e2575bfc70bf2dff9d844b01eb27f92b9118583`. D0002 contributes exactly one
additive commit with subject `m7: enforce exact baseline manifest types` and no
repair commit.

## Finding closure

| Finding | Correction | Result |
| --- | --- | --- |
| F1 numeric values accepted as booleans | Exact-type checks for `deployable`, `history`, `imu`, `anchor`, and `ics` occur before canonical-value comparison | PASS; Commander probe exits 0 |
| F2 noncanonical row fields accepted | Each field of each candidate row is compared with its canonical row | PASS; Commander probe exits 0 |

Tests cover JSON integer `0`/`1` at all five positions plus float, string,
null, array, and object substitutions. They also cover single-field drift in
all 13 registry fields and the positive independently reconstructed registry.

## Changed files

| File | Change | Reason | Semantic effect |
| --- | --- | --- | --- |
| `instinctlab/evaluation/baselines.py` | Exact JSON boolean and canonical row-field validators | Close F1/F2 | Malformed manifests and drifted registries fail closed |
| `tests/test_m7_baseline_isolation_contract.py` | Boolean-type, every-field drift, and reconstruction coverage | Prove correction | Test only |
| `BASELINE_ISOLATION_CONTRACT.md` | State exact type/value locks | Contract accuracy | Documentation only |
| `BASELINE_ISOLATION_TEST_REPORT.md` | D0002 matrix/results | Required evidence | Documentation only |
| `BASELINE_ISOLATION_EVIDENCE_INDEX.md` | D0002/D0001 evidence locations | Required evidence | Documentation only |
| `BASELINE_ISOLATION_FINAL_REPORT.md` | Finding closure and gates | Required report | Documentation only |

## Correctness gates

- Exact booleans pass; numeric and other JSON types fail: **PASS**.
- All registry fields are locked; exact independent reconstruction passes:
  **PASS**.
- D0001 valid field order, bytes, four hashes, commits, summary, and ledger are
  unchanged: **PASS**.
- Existing privilege, actor schema, capability, status, skip, mutation,
  canonical byte/hash, no-alias, and no-fallback behavior: **PASS**.
- Focused tests twice, accepted M7 twice, M3-M6 regressions twice, and M2
  static: **PASS** (1,146 pytest passes, zero skips; 66 additional subtests).
- Standard-library/import/source, exact six-path correction allowlist, one
  additive commit, clean state, diff hygiene, frozen checkout/package identity:
  **PASS**.

## Provenance and limitations

D0001 product code, commits, fixtures, raw evidence, accepted actor/runtime
sources, Handbook, paper, and frozen runtime/checkouts remain read-only inputs.
This correction establishes static/canonical contract correctness only; it
does not make B1-B8 executable or claim policy/baseline performance.

## Stop declaration

No baseline behavior or scenario was executed. No actor/control/environment/
task/reward/training/runtime/observability/source-manifest semantics changed.
No CUDA, simulator, hardware proxy, real robot, PPO/training, M8,
128k/1M/scale work, package mutation, frozen-tree mutation, Git configuration
change, history rewrite, merge, or push was performed.

`NO NEXT-PHASE WORK WAS STARTED.`

`AWAITING COMMANDER REVIEW.`
