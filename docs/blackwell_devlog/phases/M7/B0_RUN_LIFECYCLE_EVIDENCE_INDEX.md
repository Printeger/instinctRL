# M7-010 Exact B0 Run Lifecycle Evidence Index

## Accepted candidate

Coordination root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-010/M7-010-D0001`

Accepted formal candidate: `formal_candidates/0002`.

| Evidence | Path | SHA-256 / fact |
| --- | --- | --- |
| formal summary | `formal_candidates/0002/SUMMARY.json` | status PASS; 1636 passed |
| command ledger | `formal_candidates/0002/COMMAND_LEDGER.json` | 35 exact command records |
| requirement map | `formal_candidates/0002/REQUIREMENT_MAP.json` | four correctness groups |
| manifest fixtures | `formal_candidates/0002/MANIFEST_FIXTURES.json` | `2991f94839653318e4c41bed65db4a34e435e41e3b3eda30e1c8d1c6ec0b2d88` |
| receipt process 1 | `formal_candidates/0002/process_receipts/process_1.json` | `c3402e869da30748c871a58f0a79451f460b1bbaf6b64df4c0de393803c68296` |
| receipt process 2 | `formal_candidates/0002/process_receipts/process_2.json` | same exact bytes |
| receipt comparator | `formal_candidates/0002/process_receipts/exact_comparison.json` | PASS |
| raw stdout/stderr | `formal_candidates/0002/raw_logs/` | indexed by command ledger |

## Persisted programs

The accepted summary binds the SHA-256 of `independent_hash_oracle.py`,
`process_receipt.py`, `compare_process_receipts.py`, `negative_matrix.py`,
`audit_call_state.py`, `audit_source.py`, `audit_integrity.py`,
`generate_manifest_fixtures.py`, and `run_formal_validation.py`. The final
audit program and its ledger are indexed by the Executor report after the
delivery commit exists.

## State and privacy evidence

- Call/state audit: `raw_logs/008_call_state_audit.stdout`; three accepted
  calls, transitions `ACTIVE/0` through `ACTIVE/3` to `FINALIZED/3`, retained
  input count zero.
- Negative matrix: `raw_logs/009_negative_matrix.stdout`; nine rejecting cases,
  all atomic.
- Independent hash oracle: `raw_logs/010_independent_hash_oracle.stdout`; 27
  framed tensors and exact digest equality.
- Source/import/allowlist audit: `raw_logs/006_source_audit.stdout`; exact source
  SHA `5210566e987b737657d30ba44075a4dc835e379eae438726449dbc76002ab847`.
- Frozen integrity audit: `raw_logs/007_integrity_audit.stdout`; formal source,
  Isaac Lab, OmniDrones, NavRL, Python and Torch identities PASS.

Candidate `0001` is immutable and superseded only for evidence-strengthening;
its files are not acceptance evidence.
