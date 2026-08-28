# M7-006 Baseline Isolation Test Report

Accepted evidence candidate: `M7-006-D0001/formal_candidates/0002`

## Results

| Gate | Result | Count | Evidence command |
| --- | --- | ---: | --- |
| Exact READY/EXECUTED/NOT_READY/SKIPPED manifest fixtures | PASS | 4 | ledger 005-009 |
| Persisted alias/schema/capability/audit/hash/nonfinite negative matrix | PASS | 17 expected rejects | ledger 010-027, each actual exit 2 |
| Standard-library and reverse-import/source-hash isolation | PASS | 1 audit | ledger 028 |
| Functional commit/allowlist/frozen repos/critical packages | PASS | 1 audit | ledger 029 |
| Focused baseline isolation suite, pass 1 | PASS | 47 | ledger 030 |
| Focused baseline isolation suite, pass 2 | PASS | 47 | ledger 031 |
| M2 static suite | PASS | 10 | ledger 032 |
| Accepted M7 pure suites, pass 1 | PASS | 267 | ledger 033 |
| Relevant M3-M6 pure regressions, pass 1 | PASS | 204 + 33 subtests | ledger 034 |
| Accepted M7 pure suites, pass 2 | PASS | 267 | ledger 035 |
| Relevant M3-M6 pure regressions, pass 2 | PASS | 204 + 33 subtests | ledger 036 |
| Functional history/name-status/stat/full diff/diff-check/clean | PASS | 6 commands | ledger 037-042 |

The formal summary records 1,046 pytest passes and zero skips; repeated subtests
are reported separately by pytest and are not added to that count.

## Requirement mapping

- Exact B0-B8 identity, order, descriptions, families, schemas, toggles,
  capabilities, optionality, privilege, and readiness: focused tests
  `test_exact_canonical_rows_and_every_field` and
  `test_exact_policy_schemas_and_b0_evidence_binding`.
- B1 nondeployability and deployable actor cleanliness:
  `test_b1_privilege_and_all_deployable_rows_are_actor_clean`.
- Alias, case, whitespace, duplicate, missing/extra/reordered, unsupported,
  wrong-hash, false-readiness, and fallback rejection: focused parametrized
  negatives plus persisted 17-file comparator matrix.
- B8 explicit optional skip: `test_ready_executed_not_ready_and_b8_skipped_hand_fixtures`
  and `skipped.json`; unavailable B8 `NOT_READY` is rejected.
- Canonical bytes/order/hash/round trip/ownership/commit/platform/sensor/
  schema/toggle/scenario/seed/audit/status binding: four hand fixtures and
  focused serialization/mutation tests.
- Nonfinite/control/duplicate-key/hash tampering: focused negatives and
  negative files 15-17.
- Standard-library-only and reverse-import isolation: focused static test and
  `audit_import_isolation.py`.
- Accepted source identity: actor, B0 governor, metrics source manifest, and
  runtime source-manifest file hashes match the base.

## Test boundaries

No CUDA device, Isaac Lab/Isaac Sim runtime, scenario, baseline adapter,
ablation, rollout, training, checkpoint, policy-quality metric, or M8 gate was
executed. `candidate 0001` is not acceptance evidence: it stopped before tests
because the formal runner was given a mistyped caller-supplied functional SHA.
It remains immutable. Fresh `candidate 0002` used the exact Git object and
passed all gates.
