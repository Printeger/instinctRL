# M7-011 Exact B0 TASLAB Environment Test Report

## Accepted result

Formal candidate `M7-011-D0001/formal_candidates/0002` is `PASS`. It ran 37
directly invoked, timestamped commands in the frozen Python 3.11.15 / Torch
2.7.0+cu128 / Isaac Lab 2.3.2 environment with third-party pytest plugin
auto-loading disabled. The bounded headless CUDA smokes used the RTX 5070 and
the accepted TASLAB direct environment; no collector, scenario matrix,
training, M8, or scale work ran.

| Check | Repetitions | Result | Count |
| --- | ---: | --- | ---: |
| environment + lifecycle + execution focused suite | 2 | PASS | 142 each |
| accepted M7 pure suite | 2 | PASS | 445 each |
| M3-M6 behavior regression | 2 | PASS | 203 each, 1 deselected |
| M2 static | 1 | PASS | 10 |
| M6 environment CUDA regression | 1 | PASS | 32 policy / 640 physics steps |
| M7-009 controller CUDA regression | 1 | PASS | 256 controller steps |
| independent B0 environment process | 2 plus comparator | PASS | 8 policy / 160 physics steps each |
| comparator mutation matrix | 1 | PASS | 7 rejecting mutations |
| source, integrity, call/state/privacy, negative audits | 1 each | PASS | 4 auditors |

Total formal pytest result: **1590 passed, 2 deselected, 0 skipped**. The two
deselections are the Order-authorized obsolete assertion
`test_accepted_files_are_unmodified_from_order_base`, once in each M3-M6 run.

## Requirement coverage

- Exact B0 command-only environment construction, headless mode, two
  environments, `cuda:0`, decimation 20, and disabled action noise: PASS.
- Exact `[N, 3]` finite command validation, copied ownership, no actor/private
  input path, and failure before simulator mutation: PASS.
- Exactly one lifecycle step and one wrench application per physics step,
  with no duplicate controller pipeline: PASS.
- Exact disabled control-core diagnostics (`alpha=1`, correction zero, anchor
  inactive, ICS beta 1 and emergency false): PASS.
- Exact policy/physics/lifecycle/wrench counts and zero reset, collision or
  done event during the bounded smoke: PASS.
- Finite state/control/Lee/wrench records, body-command identity, exact
  Lee-to-wrench identity, and accepted rotor/throttle/thrust/state bounds:
  PASS.
- Exact horizon finalization, one READY and one EXECUTED manifest, and cleared
  command/control/Lee/wrench caches: PASS.
- Two separate process bundles: PASS; 43 tensors and 35,256 elements compare
  exactly, with trajectory SHA-256
  `15205de4ff92eb1a12ae9b6c32c43721397bab7375f0d5549244fc6db7657f5b`.
- Missing, extra, reordered, bit-mutated, count-mutated, receipt-mutated and
  private-injected bundles are all rejected: PASS.
- Seven ordered B0 capability bindings, stale-six rejection, direct-import
  isolation, unchanged B1-B8, M2-M7 regressions and frozen integrity: PASS.

## Non-accepted development invocations

During development, the first CUDA smoke stopped before a physics step because
the accepted TASLAB environment's `[N, 4]` action guard was still selected;
the environment mode was corrected to invoke the inherited Isaac Lab step
while retaining the TASLAB task callbacks. One pure-suite run then exposed an
existing reverse-import audit and the exact B0 config import was routed through
the accepted lifecycle module. Both issues were fixed before the functional
commit and all development suites then passed.

Formal candidate `0001` passed product tests and both CUDA environment smokes,
but its coordination-only comparator mutation driver indexed a three-rank root
tensor with four indices. It stopped at command 22 and is immutable and
superseded. Candidate `0002` contains the corrected coordination program and is
the sole acceptance candidate.
