# M6-004 Bounded Learning-Pipeline Final Report

## Delivered boundary

M6-004 connects the accepted two-environment task and PPO core through one
narrow `training.rollout` adapter. Exact M4 policy observations reach only the
actor; exact M6 privilege reaches only the critic. The sampled normalized Beta
action is passed unchanged to the public environment step. Each transition is
owned by detached clones, assembled time-major, and flattened losslessly.

For pure timeout, the adapter verifies the detached pre-reset/task cause and
done masks, reconstructs `[N,18]` only through accepted `build_critic_state`,
and supplies that critic value to accepted GAE. Post-reset next value cannot
affect the timeout row. Termination, including simultaneous timeout, retains
zero-bootstrap precedence and all done boundaries cut recursion.

## Bounded runtime result

Each of two independent headless CUDA processes used exactly two environments,
two four-step segments, eight policy steps / 160 physics steps, 16 transitions,
two PPO updates, and one checkpoint reload. Env0 crossed a public pure-timeout
edge on the first step and then completed seven finite post-reset steps. Strict
reload reproduced actor/critic outputs, both optimizer states, update counter,
and CPU/all-CUDA RNG exactly before segment two.

Both 23-assertion semantic inspections passed. Complete trees compared across
136 tensors and 9,931,586 elements at maximum absolute error 0.0. Two accepted
regression rounds each passed 204 tests plus 33 subtests; M2 static, accepted
M6-003 tensor smoke, accepted M6-002 runtime, all five comparator negatives,
accepted byte integrity, source/scope audits, and frozen/runtime/package checks
passed.

## Provenance identities

- accepted environment SHA256:
  `4fba15d459db403e751a0741364f567a350359d97fd3e3457b85ef9ba697e2b1`
- accepted critic builder SHA256:
  `f438cafe94a4f109da433a900aea0469b44591b0e95362dffc334678444f6341`
- accepted networks SHA256:
  `a0790d5f80c8a9f2f06d05fc80bcdcfe8fd3e1a5c71a482443f1b3f5769dc129`
- accepted PPO SHA256:
  `796f8bcb2309cea09fd229784f6b8c6b154b01be69ec5605b793952db89ec189`
- accepted checkpoint SHA256:
  `d04f77d3e443ed8466db5dc6f0c05d633fd012b799eedf4d6047d4e395e9b37b`
- accepted actor observation SHA256:
  `dfc339233e146e585493bfcc1893b5ee87c46c827963e36d6f9ec79e4b81ce44`

## Stop declaration

NO ENVIRONMENT/SIMULATOR STATE RESTART SUPPORT WAS CLAIMED OR IMPLEMENTED.
NO PRODUCTION TRAINER, EVALUATION, CURRICULUM, RANDOMIZATION, M7, 128K, 1M,
OR SCALE WORK WAS STARTED. NO MERGE, PUSH, PACKAGE/RUNTIME/SYSTEM CHANGE, OR
HISTORY REWRITE WAS PERFORMED.
