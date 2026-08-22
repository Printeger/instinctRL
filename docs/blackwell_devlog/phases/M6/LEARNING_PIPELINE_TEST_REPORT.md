# M6-004 Bounded Learning-Pipeline Test Report

## Result

Formal implementation candidate `0001` is PASS: all 41 commands exited exactly
as declared and all eight aggregate assertions passed.

| Check | Result | Measured evidence |
| --- | --- | --- |
| Compile and AST information-flow audit | PASS | actor args `policy`; critic args `built`, `privileged`, `next_privileged`; no forbidden imports/tokens |
| Adapter focused tests | PASS | included in each 204-test regression round; 29 focused tests |
| Accepted regression round 1 | PASS | 204 tests + 33 subtests |
| Accepted regression round 2 | PASS | 204 tests + 33 subtests |
| M2 static | PASS | 10 tests |
| Accepted M6-003 tensor CUDA smoke | PASS | four samples, two updates, 83,504,128 peak bytes |
| Accepted M6-002 environment runtime | PASS | 32 policy / 640 physics steps |
| Pipeline CUDA process 1 | PASS | 2 env, 8 policy / 160 physics steps, 16 samples, 2 updates, 1 reload |
| Pipeline CUDA process 2 | PASS | same exact dimensions |
| Independent semantic inspectors | PASS | 23/23 assertions each; one pure timeout each |
| Complete-tree comparison | PASS | 136 tensors, 9,931,586 elements, maximum error 0.0 |
| Comparator negatives | PASS | missing, extra, reordered, nonfinite, and any-error trees rejected |
| Accepted-byte/frozen/runtime/package/scope integrity | PASS | exact exits and pinned heads |

Each new process used 143,420,928 peak CUDA allocation bytes. The forced
timeout occurred on segment 1, step 1, env0 through the public step path. The
same segment contained three finite post-reset steps and segment two contained
four more. Both segment batches contained exactly eight samples and one update.

## Retained development evidence

Initial debug trees `smoke_1.pt` and `smoke_2.pt` individually passed semantic
inspection but did not compare exactly after the first PPO backward. Persisted
diagnosis proved segment-one environment inputs, actions, reward, masks,
bootstrap, and GAE were exact; the first difference was the gradient-norm
reduction. The smoke then added an explicit simulator-to-PPO CUDA synchronization
boundary and deterministic cuDNN/TF32 settings without changing PPO or relaxing
the comparator. Fresh debug trees 3/4 and formal trees 1/2 compare at error 0.0.

## Not run

Production collection, restarted-simulator resume, general training, policy-
quality/convergence/throughput gates, evaluation, curriculum, randomization,
M7, 128k, 1M, scale training, merge, and push were NOT RUN.
