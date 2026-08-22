# M6-002 Environment Test Report — D0002

## Commander findings

| Finding | Result | Behavioral closure |
| --- | --- | --- |
| F1 automatic done/reset absent | PASS | env0 public-step timeout at step 16; terminal and post-reset trees persisted; 16 later steps finite |
| F2 subset reset clears batch caches | PASS | selected slices zero; every unaffected control/Lee/wrench/reward/task slice exact |
| F3 reset evidence incomplete | PASS | discriminating runtime matrix covers every ordered subsystem plus full reset |
| F4 contact record inaccurate | PASS | record now states full `net_forces_w` vector norm at 1 N |

## Development validation

| Gate | Result |
| --- | --- |
| strengthened environment suite | PASS — 8 tests |
| M3/M4/M5-001/M5-002/M6-001/new suite pass 1 | PASS — 118 tests + 33 subtests |
| same suite pass 2 | PASS — 118 tests + 33 subtests |
| M2 static | PASS — 10 tests |
| accepted M5 runtime | PASS — 256 physics steps |
| two independent environment processes | PASS — each 2 envs, 32 policy/640 rollout physics steps plus reset qualification |
| unchanged complete comparator | PASS — 4,423 tensors, 20,925,086 elements, maximum error `0.0` |

## Automatic-reset timeline

1. Before step 16, fixture stages only env0 elapsed counter to 999.
2. Public `env.step()` holds the accepted action for 20 physics iterations.
3. Base counter increment makes env0 elapsed 1,000 and env1 elapsed 16.
4. Accepted termination returns env0 timeout/reason 1; reward and the detached
   pre-reset tree describe this terminal state.
5. DirectRLEnv automatically calls selected reset for env0.
6. Returned env0 episode counter is zero and policy/critic are post-reset;
   env1 is not reset and all cache slices remain exact.
7. Steps 17 and 18 pass explicit continuation gates; execution continues
   through step 32 with finite policy, critic, reward, and root state.

## Reset evidence

Before selective reset, root velocity, joint state, RayCaster drift/times,
contact force/history/times, rotor/pending wrench, M4 history/times, elapsed
steps, actions, command buffers, reset flags, policy cache, and all task/control
caches are populated with discriminating values. Runtime asserts exact selected
defaults/zeros and recursively compares the entire unaffected environment tree.
The same tree is persisted in both rollout outputs and compared unchanged.

NOT RUN: policy/critic networks, PPO/GAE, optimization, learning, checkpoint,
evaluation, curriculum, randomization, M7, 128k, 1M, or scale training.
