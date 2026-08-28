# M7-007 B0 Direct Adapter Test Report

Accepted functional evidence candidate: `M7-007-D0001/formal_candidates/0003`

## Results

| Gate | Result | Count | Evidence |
| --- | --- | ---: | --- |
| Independent-process B0 records and comparator | PASS | 2 records, byte-identical | ledger 005-007 |
| READY/EXECUTED/NOT_READY/SKIPPED manifests | PASS | 4 | ledger 008-012 |
| Capability/alias/fallback negative manifests | PASS | 10 expected rejects | ledger 013-023 |
| Exact source/signature/import/B1-B8 audit | PASS | 1 | ledger 024 |
| Functional allowlist/frozen/package integrity | PASS | 1 | ledger 025 |
| Focused B0 adapter, pass 1/pass 2 | PASS | 53 / 53 | ledger 026-027 |
| Updated baseline isolation, pass 1/pass 2 | PASS | 72 / 72 | ledger 028-029 |
| M2 static | PASS | 10 | ledger 030 |
| Accepted M7 pure suites, pass 1/pass 2 | PASS | 345 / 345 | ledger 031, 033 |
| Relevant M3-M6 pure regressions, pass 1/pass 2 | PASS | 203 / 203 + 33 / 33 subtests | ledger 032, 034 |
| Old M6 whole-control-directory immutability assertion | NOT RUN | 1 deselected per pass | ledger 032, 034 |
| Functional history/allowlist/stat/diff/diff-check/clean | PASS | 6 | ledger 035-040 |

Aggregate pytest result is 1,356 passed and zero skipped, plus 66 separately
reported subtests. The deselected M6 assertion requires the entire control
directory to remain byte-identical to an M6 base and is structurally
incompatible with this Order's explicit authorization to add
`control/baseline_b0.py` and update `control/__init__.py`. It is replaced by
the stricter M7 exact eight-path allowlist and accepted-source hash audits; no
M3-M6 behavior test is deselected.

## Requirement mapping

- Zero/below/exact/over-limit, direction, float32/64, batch, identity, yaw
  ±90°, and nontrivial quaternion oracles: focused adapter tests.
- Accepted `b0_governor` alpha-one/zero-correction/body parity and accepted
  `rotate_body_to_world` parity: focused adapter tests.
- Type/shape/batch/dtype/device/finite/requires-grad/quaternion/config
  negatives: focused parametrized tests.
- No mutation, distinct owned storage, detached output, caller mutation
  isolation, repeated-call determinism, stateless failure atomicity, frozen
  records, and no default object: focused ownership/signature tests.
- Exact command-only actor surface, controller-private quaternion, import/name
  exclusion, and no actor/environment/training/evaluation reverse import:
  focused AST test plus persisted source audit.
- Exact three-binding order/providers/hashes, old/missing/extra/reordered/
  duplicate/wrong/B1/alias rejection, B1-B8 source preservation, and canonical
  manifests: isolation tests and persisted fixtures/negative matrix.

## Boundaries

No CUDA, Isaac Lab, Isaac Sim, environment/controller runtime integration,
baseline/scenario execution, metric aggregation, training, M8, or scale work
was run.

Candidates `0001` and `0002` are retained but excluded from acceptance. The
first was executed before the final-audit validator was persisted. The second
passed every command but its summary counter also parsed a pytest-count phrase
inside `git diff` documentation. Candidate `0003` persisted both corrected
programs before execution and reports the exact 1,356-test aggregate.
