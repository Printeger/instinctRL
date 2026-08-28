# Runtime Evaluation Evidence Index

Revision 1 attempt root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-002/M7-002-D0002/`

- `audit_source.py`: dependency, structured-metadata, exact actor-call, and
  prohibited-work audit.
- `assert_integrity.py`: exact formal/frozen heads and status bytes, Isaac Lab
  kit SHA256, R580 driver, locked core versions, and frozen `pip freeze` byte
  assertions.
- `inspect_runtime.py`: exact 23-entry metadata, terminal row, post-reset policy,
  next audited policy, station-reference, coverage, finiteness, and storage
  ownership inspection.
- `generate_mutations.py`: eight persisted comparator-negative artifacts.
- `extract_manifest.py`: actor/trajectory/timeout-policy hashes and source
  manifest digest.
- `run_formal_validation.py`: persisted 51-command formal runner.
- `final_audit.py`: exact parent chain, three commits, seven-path rework
  allowlist, base allowlist, diff hygiene, clean state, and `STOP` assertion.
- `formal_candidates/0001/VALIDATION_SUMMARY.json`: preserved **FAIL**, 50/51,
  caused by the validator `PYTHONPATH`/package-collection context mismatch.
- `formal_candidates/0002/COMMAND_LEDGER.json`: argv, cwd, environment, UTC
  times, expected/actual exits, streams, and program hashes.
- `formal_candidates/0002/VALIDATION_SUMMARY.json`: all passing semantic,
  history, scope, frozen, driver, and package assertions.
- `formal_candidates/0002/OUTPUT_MANIFEST.json`: derived artifact sizes and
  SHA256 values.
- `formal_candidates/0002/derived/runtime_1.pt`, `runtime_2.pt`: complete
  deterministic runtime trees.
- `formal_candidates/0002/derived/inspection_1.json`, `inspection_2.json`:
  structured metadata and terminal/post-reset lifecycle checks.
- `formal_candidates/0002/derived/comparison.json`: zero-error recursive
  complete-tree comparison.
- `formal_candidates/0002/derived/runtime_manifest.json`: actor, policy,
  concentration, action, timeout-policy, and metadata hashes.
- `formal_candidates/0002/derived/mutations/`: eight negative artifacts.
- `formal_candidates/0002/raw_logs/`: exact stdout, stderr, and exit records.

Immutable passing D0002 `formal_candidates/0002` hashes:

- `COMMAND_LEDGER.json`: `16c4622ff694f8d67ce21531323c8bcb50abd5531348a4ad8d2ce198a954b628`.
- `VALIDATION_SUMMARY.json`: `2e98e9803f910fda96999836eacd7132b2aa15f94924495ccbf3a4a1ee61b453`.
- `OUTPUT_MANIFEST.json`: `813187c4abaf64de4b256cef3de9448cf155688dceda4ce83325f82b335544ad`.

D0001 remains available read-only at its original attempt root and is referenced
without copying or changing any historical byte.
