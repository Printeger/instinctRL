# M7-013 Canonical Evaluation Scenario Catalog Evidence Index

## Accepted candidate

Coordination root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-013/M7-013-D0001/formal_candidates/0001`

| Evidence | SHA-256 |
| --- | --- |
| `SUMMARY.json` | `66a9b97d929bc13839681bd7af0a1ee5351bcbe843cae9eecab6a1df762a1635` |
| `COMMAND_LEDGER.json` | `fe30b039fc57c2a33189e8da9e84a0c421532ced34f8d819caba281b7ec5dfcc` |
| `REQUIREMENT_MAP.json` | `4c6034dffe6d44be3c682e7ead77c8f879016c67b801df168b6eeb24302079b5` |
| `OUTPUT_VALIDATION.json` | `4b797a9951677c62a031740d2d994d8028343cf533ca6de3ac5830f622815f18` |
| `COMPARATOR_MUTATIONS.json` | `2e4eab856326f833c8a184c49a43a7d353161cdbaba4a73a83376962274a15f1` |
| `bundles/exact_comparison.json` | `f859e1582ec9e7716db58b5c8f7aee084cf1b78478baf7fe3ef347eb9fe95695` |
| `bundles/process_1.json` | `21f316ba8b1aa702a538ec1d8fe55f6a838af41d48ebd4ff0e280c3f65131208` |
| `bundles/process_2.json` | `21f316ba8b1aa702a538ec1d8fe55f6a838af41d48ebd4ff0e280c3f65131208` |

The formal ledger contains 35 exact argv/cwd/environment/time/exit/stdout/
stderr records. `SUMMARY.json` hashes every formal program and candidate
artifact; `REQUIREMENT_MAP.json` maps each correctness gate to exact commands.

## Persisted auditors

Fresh attempt-root programs include `audit_source.py`, `audit_catalog.py`,
`audit_integrity.py`, `negative_matrix.py`, `emit_canonical_bundle.py`,
`compare_canonical_bundles.py`, `mutate_comparator.py`, `validate_outputs.py`
and `run_formal_validation.py`. Formal raw logs are under `raw_logs/`, exact
identity bundles under `bundles/`, and rejected mutation inputs under
`comparator_mutations/`.

No binary, image, video, checkpoint, simulator log or large runtime artifact is
committed to the product repository.
