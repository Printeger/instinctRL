# M7 Evaluation Availability Matrix Test Report

## Results

| Check | Result | Evidence |
| --- | --- | --- |
| Matrix contract precommit | PASS — 32 then 33 tests | commands `001`, `005` |
| Matrix/baseline/scenario focused pass 1 | PASS — 197 tests | command `011` |
| Matrix/baseline/scenario focused pass 2 | PASS — 197 tests | command `012` |
| Accepted M7 pure pass 1 | PASS — 976 tests | command `013` |
| Accepted M7 pure pass 2 | PASS — 976 tests | command `014` |
| Relevant M3-M6 pass 1 | PASS — 203 tests, 33 subtests; one ordered stale-base check deselected | command `015` |
| Relevant M3-M6 pass 2 | PASS — 203 tests, 33 subtests; one ordered stale-base check deselected | command `016` |
| M2 static | PASS — 10 tests | command `017` |

The matrix suite exhaustively verifies public signatures and field order; all
90 ordered cells and the exact 7/74/9 map; blocker precedence; B1 privilege;
B8, dynamic and real-robot behavior; seed/provider/evidence binding; canonical
ASCII and hash; frozen records; mutation isolation; and malformed, aliased,
reordered, duplicate, nonfinite, false-ready, false-executed and forged-audit
rejection. It also recursively audits imports and reverse dependencies.

## Execution classification

No CUDA, simulator, controller, scenario runner, B1-B8 adapter, dynamic fixture,
real robot, training or evaluation campaign was run. These are pure/static
contract tests only.

The first informal invocation used the host Python and failed collection because
that interpreter lacks frozen `torch`. It made no product change. All formal
tests used the frozen `navrl_blackwell_test` Python and passed; the failed Git
commit attempt caused by missing configured identity is separately retained,
then retried with the existing one-command Executor identity without modifying
Git configuration.
