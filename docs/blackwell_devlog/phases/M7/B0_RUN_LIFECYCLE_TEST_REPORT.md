# M7-010 Exact B0 Run Lifecycle Test Report

## Accepted result

Formal candidate `M7-010-D0001/formal_candidates/0002` is `PASS`. It ran
35 directly invoked, timestamped commands in the frozen Python 3.11.15 / Torch
2.7.0+cu128 environment with third-party pytest plugin auto-loading disabled.
No CUDA, Isaac Sim, environment, collector, training, M8, or scale work ran.

| Check | Repetitions | Result | Count |
| --- | ---: | --- | ---: |
| lifecycle + manifest + direct-import + isolation focused suite | 2 | PASS | 180 each |
| accepted M7 pure suite | 2 | PASS | 430 each |
| M3-M6 behavior regression | 2 | PASS | 203 each, 1 deselected |
| M2 static | 1 | PASS | 10 |
| independent hash oracle | 1 | PASS | 27 framed tensors / 3 steps |
| independent process receipt | 2 plus comparator | PASS | exact equality |
| source, call/state, negative, frozen integrity audits | 1 each | PASS | 4 auditors |

Total formal pytest result: **1636 passed, 2 deselected, 0 skipped**. The two
deselections are the Order-authorized obsolete assertion
`test_accepted_files_are_unmodified_from_order_base`, once in each M3-M6 run.

## Requirement coverage

- Exact one-step/multi-step direct parity, counters, state, finalization,
  successor consistency, frozen ownership, float32/float64 and batch variation:
  PASS.
- Zero/one call behavior, malformed input, handoff exception, successor drift,
  framing failure, later-clean recovery, early/repeated finalize and overrun:
  PASS.
- Independent framing oracle and order/bit/field/dtype/shape/manifest/count
  sensitivity: PASS.
- No tensor retention, input mutation/alias, or private value in receipt:
  PASS.
- Exact API/import/actor audit, six ordered B0 bindings, stale-five rejection,
  and unchanged non-executable B1-B8: PASS.
- Two separate process receipts: PASS; exact public JSON and trajectory digest
  equality, `a7503ca6dc664f3cbe8746fa83436c4e5d7d500c0c37a4a96d7ff57a6a23a36d`.

## Non-accepted development invocations

One system-Python invocation stopped during collection because Torch was not
installed there. A second frozen-Python invocation stopped during pytest plugin
discovery because host ROS exposed an unrelated plugin lacking `lark`. Neither
executed product tests or changed state. Formal candidates used the isolated
environment above. Candidate `0001` passed all behavior checks but was
superseded by `0002` because its runner logged, but did not hard-assert, the
known frozen Isaac Lab kit-file state.
