# M6-003 PPO Tensor-Core Test Report

## Functional candidate 0002

| Check | Result | Measured evidence |
| --- | --- | --- |
| Compile and AST isolation audit | PASS | actor args `self,policy`; no forbidden imports/recurrent/cost tokens |
| Regression round 1 | PASS | 173 tests + 33 subtests |
| Regression round 2 | PASS | 173 tests + 33 subtests |
| M2 static | PASS | 10 tests |
| CUDA PPO process 1 | PASS | batch 4; exactly 2 updates; 83,504,128 peak bytes |
| CUDA PPO process 2 | PASS | batch 4; exactly 2 updates; 83,504,128 peak bytes |
| Semantic inspectors | PASS | 13/13 assertions per process |
| Complete-tree comparator | PASS | 33 tensors; 1,020,176 elements; max error 0.0 |
| Comparator negatives | PASS | missing/extra/reordered/nonfinite/any-error rejected |
| Checkpoint negatives | PASS | missing/extra/corrupt/nonfinite/shape/dtype/config rejected |
| Accepted M6-002 environment regression | PASS | 2 envs; 32 policy / 640 physics steps |
| Accepted-file/frozen/runtime/scope integrity | PASS | expected exits and pinned heads |

The two independent smoke artifacts have identical SHA256
`cee1883680360fd1a1028b74813225202933556148143af4e75866d0a68b0ca9`.
Synthetic losses and parameter changes are numerical migration evidence only;
they are not policy-quality or convergence metrics.

## Retained failed candidate

Functional candidate 0001 is FAIL because its evidence-only scope regex
matched the legitimate phrase `log-probability evaluation`. Commands 1-29 had
passed. Candidate 0002 narrowed the regex to concrete evaluation entry points
and replayed all 41 commands. No product correction or relaxed gate resulted.

## Not run

Environment collection, simulator training, learning horizon, evaluation,
curriculum, randomization, M7, 128k, 1M, scale performance, convergence, merge,
and push were NOT RUN.
