# M6-004 Bounded Learning-Pipeline Contract

## Information flow

```text
returned observation["policy"] exact ordered M4 pair
  lidar_grid float32 [N,12,59,360] ─┐
  state_vec  float32 [N,52]         ├─> actor.sample(policy)
                                    └─> sampled action float32 [N,4] in [0,1]
                                         └─> environment.step(the same action object)

returned observation["critic"] float32 [N,18] ─> critic(current/ordinary next value)

pure-timeout extras["pre_reset"] root/task fields
  └─> accepted build_critic_state ─> terminal [N,18] ─> critic(timeout value)
```

The adapter has no environment implementation import. Actor sampling receives
one argument named `policy`; privilege, environment, reward, masks, root/task
state, extras, and terminal data never enter that call. Critic calls receive
only explicit `built`, `privileged`, or `next_privileged` tensors.

## Configuration and transition ownership

`RolloutConfig(horizon, batch_size)` is frozen and accepts positive integers
only. Each transition independently clones and detaches:

- exact pre-action M4 policy mapping and current `[N,18]` privilege;
- sampled `[N,4]` action, old log probability `[N,1]`, and current value;
- reward and independent terminated/truncated cause masks;
- reset boundary equal to the exact cause-mask union;
- returned next privilege/value;
- pure-timeout terminal privilege/value, zero outside pure-timeout rows.

The environment receives the sampled action itself; there is no mean, fixed,
B0, privileged, or clamped fallback. Collection occurs under no-gradient
semantics. Returned observation tensors may not alias their pre-action inputs,
and stored tensors may not alias either live surface.

## Terminal provenance and GAE

The accepted environment returns post-reset observation on a done row. That
returned critic value is evaluated and stored but GAE may consume it only for
ordinary continuation. For `truncated & ~terminated`, the adapter requires
`extras["pre_reset"]`, verifies returned/task masks and their exact done union,
verifies the duplicated root snapshot, and calls the accepted
`build_critic_state` with only:

- `pre_reset["root_state_w"]`;
- `pre_reset["task"]["actual_velocity_b"]`;
- `pre_reset["task"]["exact_clearance_m"]`;
- `pre_reset["task"]["collision"]`.

Termination, including simultaneous termination+timeout, uses zero bootstrap.
Pure timeout uses the reconstructed terminal value. Every done/reset boundary
cuts recursion. No post-reset or benign default value substitutes for missing
terminal data.

## Temporal order and update boundary

Transitions stack exactly as `[T,N,...]` oldest step first. Every field then
reshapes in identical time-major order to `T*N`; no sample is dropped, padded,
replayed, or reordered. `collect_rollout` performs no optimizer mutation. The
caller invokes exactly one accepted `PPOTrainer.update` after a complete
four-step segment.

After update one, the accepted strict checkpoint stores actor, critic, both
optimizers, update counter, and CPU/all-CUDA RNG. A fresh trainer loads it and
must reproduce current-observation actor/critic outputs and every saved state
exactly before collecting segment two. The live environment continues; its
state is neither saved nor recreated.

## Fail-closed behavior

Wrong/missing/extra/reordered observations, invalid batch/shape/device/dtype/
finiteness, action mutation, returned alias, malformed reward/masks, missing or
inconsistent pre-reset/task fields, invalid terminal state, incomplete horizon,
and inconsistent reset union raise before any PPO optimizer mutation. The
adapter contains no `nan_to_num`, hidden action clamp, exception swallowing,
collector, replay, recurrent, cost, or Lagrangian branch.
