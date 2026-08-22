# M6-003 Actor, Critic, GAE, PPO, and Checkpoint Contract

## Information flow

```text
exact ordered M4 policy mapping
  lidar_grid float32 [N,12,59,360] ┐
  state_vec  float32 [N,52]        ├─> HistoryActor -> alpha,beta [N,4]
                                     -> normalized action [N,4] in [0,1]

exact M6 privileged float32 [N,18] ──> PrivilegedCritic -> value [N,1]
```

`HistoryActor.forward(self, policy)` has no critic/extras/environment argument.
It rejects missing, extra, or reordered keys, empty/mismatched batches,
non-float32, cross-device, and nonfinite values. `PrivilegedCritic.forward`
accepts one explicit `[N,18]` tensor. Parameters and Adam optimizers are
disjoint; critic-only mutation checks preserve every actor tensor exactly.

## Network manifest

The actor is feedforward. Its LiDAR encoder is:

| Layer | Weight shape | Output spatial shape |
| --- | --- | --- |
| Conv2d, kernel 5, stride (2,4), pad 2 | `[8,12,5,5]` | `[N,8,30,90]` |
| Conv2d, kernel 3, stride (2,3), pad 1 | `[16,8,3,3]` | `[N,16,15,30]` |
| Conv2d, kernel 3, stride (3,3), pad 1 | `[16,16,3,3]` | `[N,16,5,10]` |

The flattened LiDAR feature has 800 elements. The state branch is
`Linear(52,32) -> ELU -> LayerNorm(32)`. The merged branch is
`Linear(832,128) -> ELU -> LayerNorm(128) -> Linear(128,8)`. Range channels
alone are divided by the fixed qualified 40 m maximum; validity/reliability
channels retain their values. Actor parameter count is 115,568.

The critic is `Linear(18,64) -> ELU -> LayerNorm(64) -> Linear(64,64) -> ELU
-> LayerNorm(64) -> Linear(64,1)`, with 5,697 parameters. There is no shared
actor feature, recurrent state, cost critic, or Lagrange branch.

## Beta action contract

Eight raw outputs are sigmoid-mapped into four alpha and four beta
concentrations in validated `[1.0,30.0]`. Mean action is `alpha/(alpha+beta)`.
Sampling and mean action stay in the environment's closed `[0,1]` domain.
Log probability and entropy sum over exactly four dimensions to `[N,1]`.
The only clamp is the declared `1e-6` open-boundary epsilon used while
evaluating Beta log probability; it does not change the environment action.

## GAE contract

Defaults are `gamma=0.99`, `lambda=0.95`. For each transition:

```text
delta_t = reward_t + gamma * bootstrap_t - value_t
adv_t = delta_t + gamma * lambda * recurse_t * adv_(t+1)
return_t = adv_t + value_t
```

- ordinary continuation: bootstrap from supplied next-state value and recurse;
- termination: zero bootstrap and no recursion, including when timeout is also
  true;
- pure timeout (`truncated=True, terminated=False`): bootstrap only from an
  explicit terminal-state critic value and do not recurse into the reset
  episode;
- explicit reset boundary: zero bootstrap and no recursion.

The task cause masks remain independent, so terminated and truncated may both
be true. True termination has precedence and the simultaneous row depends on
neither timeout bootstrap nor post-reset `next_values`. An explicit terminal
bootstrap is required only when at least one pure-timeout row exists.

## PPO update contract

Old log probability is detached, cloned for immutability verification, and
never recomputed as old policy. Ratio is `exp(new_logp-old_logp)`. Actor loss is
the negative mean of the minimum of unclipped `ratio*advantage` and the
`[1-0.2,1+0.2]` clipped surrogate. Value loss is half mean squared error for
the single critic. Total loss is actor loss plus `0.5*value_loss` minus
`0.01*entropy`. Metrics include ratio mean, approximate KL, clip fraction,
entropy, actor/value/total loss, and actor/critic pre-clip gradient norms.

Advantages use finite population-variance normalization; a zero-variance batch
returns explicit zeros. Batch divisibility is mandatory. A seeded CPU
permutation covers every sample exactly once per epoch with no drop. Inputs,
losses, gradients, parameters, optimizer states, and metrics are checked for
finiteness. Separate global gradient clipping defaults to `0.5`.

## Checkpoint contract

Version `instinctlab-m6-ppo-core-1` contains exact schema, immutable network and
PPO config, actor/critic model state, both optimizer states, update counter,
and CPU/all-CUDA RNG states. Save uses a fsynced temporary sibling and atomic
`os.replace`. Load uses `torch.load(..., weights_only=True)` and rejects
missing/extra keys, version/schema/config mismatch, parameter and optimizer
shape/dtype mismatch, nonfinite tensors, corruption, invalid counter/RNG, and
partial state. Reload must reproduce deterministic mean action, log
probability, critic value, counter, optimizer, and RNG state exactly.

## Legacy disposition

Retained from formal legacy evidence: Beta bounded-action concept, PPO minimum
clipped surrogate, GAE defaults, entropy/value terms, and minibatch update
lessons. Corrected: timeout/reset bootstrap ambiguity, broad dependency and
container interfaces, shared actor/critic feature optimizer, incomplete finite
audits, and loose checkpoint loading. Excluded: goal/world-frame action
conversion, privileged actor inputs, TensorDict/TorchRL collector coupling,
environment collection, cost critic/Lagrangian, recurrence, and training jobs.
