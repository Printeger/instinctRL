# M6-002 Environment Contract — Revision 1

## Preserved contract

The D0001 public API provenance, `cuda:0`, `0.005 s` physics interval, 20-step
decimation, `0.1 s` policy interval, normalized float32 `[N,4]` non-B0 action,
exact M4 policy pair, separate M6 `[N,18]` critic, frames, units, RayCaster
producer, direct M5 metrics, reward formulas, and termination semantics remain
unchanged. Frozen Isaac Lab authority remains
`37ddf626871758333d6ed89cf64ad702aef127d0`.

Collision is the boolean result of any TASLAB ContactSensor public
`net_forces_w` full three-dimensional vector norm exceeding `1.0 N`. No
component is removed before the norm. The label remains privilege-only.

## Terminal and post-reset provenance

DirectRLEnv calls done, reward, selected reset, and observation in that order.
After reward and before reset, the environment copies a detached diagnostic
`extras["pre_reset"]` tree containing reset mask, elapsed step, root/joints,
control, Lee, wrench/rotor, reward, and task state. It is never read by policy
or action/control code.

Returned reward, terminated/truncated, reason, and pre-reset extras describe
the same terminal post-physics state. Returned policy and critic are built only
after selected reset. Reward is returned as an independent clone so clearing a
selected reward-cache slice cannot mutate the terminal return value.

## Selective cache contract

Control, Lee, wrench, reward, and task caches remain batch-shaped. A subset
reset recursively zeroes only tensors whose leading dimension is the
environment batch and only at selected indices. Unaffected slices retain exact
tensor values, dtype, device, shape, key order, and object-level semantics.
Only a full reset may replace these whole-batch cache objects with `None`.

## Complete reset matrix

| Subsystem | Selected reset | Unaffected environment |
| --- | --- | --- |
| articulation root pose/velocity | public defaults plus env origin | exact equality |
| articulation joints | public default position/velocity | exact equality |
| articulation/sensor reset | selected public reset invoked | exact equality |
| RayCaster | timestamp reset, outdated before read, zero configured drift | exact equality |
| ContactSensor | force/history/timestamp reset, outdated before read | exact equality |
| rotor/pending wrench | throttle, force, torque zero | exact equality |
| M4 history/time | lidar/state/time/has-time cleared | exact equality |
| M5 pipeline/anchor/ICS | selected instance replaced with empty caches | exact equality |
| elapsed/action/commands | selected values zero | exact equality |
| reset flags | selected flags staged true | exact equality |
| policy cache | selected lidar/state zero before refresh | exact equality |
| control/Lee/wrench/reward/task caches | selected tensor slices zero | exact equality |

The test fixture seeds both environments with distinct non-default values
before this matrix is checked. Full reset separately proves both environments
return to defaults and all whole-batch caches become unavailable.

## Automatic timeout fixture

The product default remains 1,000 policy steps. At rollout step 16 only the
test fixture stages env0 elapsed count to `max_episode_length - 1`; the public
step increment reaches the already accepted timeout edge. Env0 returns
`truncated=True`, reason code 1, terminal reward/extras, then resets to elapsed
zero. Env1 returns no done and its cached state remains exact. The returned env0
M4 history has three zero older frames and one post-reset frame, preventing
terminal history leakage. Sixteen further finite public steps are completed.

## Privilege boundary

Terminal/pre-reset trees, root state, actual velocity, clearance, contact,
critic, reward, reason, controller, wrench, and rotor state are diagnostics or
critic/task-only. The policy branch remains recursively restricted to exact M4
`lidar_grid` and `state_vec`; `_pre_physics_step` reads neither extras nor
critic/task privilege.
