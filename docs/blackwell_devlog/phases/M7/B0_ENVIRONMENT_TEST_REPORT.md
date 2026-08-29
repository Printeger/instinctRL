# M7-011 Exact B0 TASLAB Environment Test Report

## Accepted result

Formal candidate `M7-011-D0003/formal_candidates/0001` is `PASS`. It ran 41
directly invoked, timestamped commands in the frozen Python 3.11.15 / Torch
2.7.0+cu128 / Isaac Lab 2.3.2 environment with third-party pytest plugin
auto-loading disabled. The bounded headless CUDA smokes used the RTX 5070 and
the accepted TASLAB direct environment; no collector, scenario matrix,
training, M8, or scale work ran.

| Check | Repetitions | Result | Count |
| --- | ---: | --- | ---: |
| environment + lifecycle + execution focused suite | 2 | PASS | 147 each |
| accepted M7 pure suite | 2 | PASS | 450 each |
| M3-M6 behavior regression | 2 | PASS | 203 each, 1 deselected |
| M2 static | 1 | PASS | 10 |
| M6 environment CUDA regression | 1 | PASS | 32 policy / 640 physics steps |
| M7-009 controller CUDA regression | 1 | PASS | 256 controller steps |
| independent B0 environment process | 2 plus comparator | PASS | 8 policy / 160 physics steps each |
| comparator mutation matrix | 1 | PASS | 7 rejecting mutations |
| source, integrity, call/state/privacy, negative audits | 1 each | PASS | 4 auditors |

Total formal pytest result: **1610 passed, 2 deselected, 0 skipped**. The two
deselections are the Order-authorized obsolete assertion
`test_accepted_files_are_unmodified_from_order_base`, once in each M3-M6 run.

## Requirement coverage

- Exact B0 command-only environment construction, headless mode, two
  environments, `cuda:0`, decimation 20, and disabled action noise: PASS.
- Exact factory-owned configuration and post-construction revalidation of the
  realized runtime configuration: PASS.
- Exact `[N, 3]` finite command validation, copied ownership, no actor/private
  input path, and failure before simulator mutation: PASS.
- Exactly one lifecycle step and one wrench application per physics step,
  with no duplicate controller pipeline: PASS.
- Exact disabled control-core diagnostics (`alpha=1`, correction zero, anchor
  inactive, ICS beta 1 and emergency false): PASS.
- Exact policy/physics/lifecycle/wrench counts and zero reset, collision or
  done event during the bounded smoke: PASS.
- Body/world command transforms and all five accepted Lee outputs have zero
  mismatch count and zero maximum error: normalized rotor commands, held
  position, held yaw, desired rotation, and collective thrust: PASS.
- Exact horizon finalization, one READY and one EXECUTED manifest, and cleared
  command/control/Lee/wrench caches: PASS.
- Two separate process bundles: PASS; 66 tensors and 45,496 elements compare
  exactly, with trajectory SHA-256
  `97ba25dd50bfdaf89e6f5da1dfb58f23972f98dee7cfa9e7c146bfb0bfc65e96`.
- Missing, extra, reordered, bit-mutated, count-mutated, receipt-mutated and
  private-injected bundles are all rejected: PASS.
- Seven ordered B0 capability bindings, stale-six rejection, direct-import
  isolation, unchanged B1-B8, M2-M7 regressions and frozen integrity: PASS.

## Superseded attempts

`M7-011-D0001/formal_candidates/0001` is immutable and superseded because its
coordination-only comparator mutation driver had an indexing defect. D0001
candidate `0002` supported the initial delivery but was superseded by the
Commander-ordered exactness closure.

`M7-011-D0002/formal_candidates/0001` is immutable failed evidence: all
functional exactness checks passed, but the smoke runner emitted one redundant
root-level `oracle_diagnostics` bundle key. D0003 deleted exactly that one line
in repair commit `4eb7e067dcfe92ff1318679e8704c98728c035be` and reran the
complete formal validation from fresh evidence.
