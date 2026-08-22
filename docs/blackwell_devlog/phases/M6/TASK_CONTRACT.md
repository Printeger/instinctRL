# M6 Reward, Termination, and Critic-Only Tensor Contract

Order: `M6-001`, revision 1

## Information boundary

Actor-visible state remains exactly:

```text
lidar_grid [N,12,59,360]
state_vec [N,52]
```

Accepted M5 produces `v_gov_b` and `v_final_b` from that actor-clean surface.
M6 does not import into or write back to observations or control. Privileged
actual body velocity, root state, exact clearance, collision and world height
enter only named reward, critic or termination functions.

All public floating tensors are required finite float32/float64 tensors with
exact rank/shape, one batch, dtype and device. Accepted M5 state labels are
same-dtype floating `[N,1]` tensors containing exact 0 or 1; collision remains
an exact `[N,1]` boolean privileged label. Elapsed steps are nonnegative
`[N,1]` `torch.long`. Missing values,
NaN/Inf, malformed quaternions, invalid domains and mismatches raise without
partial output. No detach, CPU transfer, `torch.no_grad`, `nan_to_num`, hidden
fallback or broad exception handling exists.

## Reward

Required inputs are:

| Tensor | Shape | Frame/unit | Ownership |
| --- | --- | --- | --- |
| `v_cmd_b` | `[N,3]` | body, m/s | actor-clean command |
| `v_gov_b` | `[N,3]` | body, m/s | accepted M5 |
| `v_final_b` | `[N,3]` | body, m/s | accepted M5 |
| `previous_issued_v_final_b` | `[N,3]` | body, m/s | accepted command history |
| `actual_velocity_b` | `[N,3]` | body, m/s | reward-only privileged |
| `anchor_loss` | `[N,1]` | accepted nonnegative Huber loss | actor-clean metric |
| `anchor_active` | `[N,1]` float exact 0/1 | accepted state | accepted M5 metric |
| `anchor_usable_fraction` | `[N,1]` float | closed `[0,1]` | accepted M5 metric |
| `ics_beta` | `[N,1]` | closed `[0,1]` | accepted M5 metric |
| `ics_emergency` | `[N,1]` float exact 0/1 | emergency requires beta exactly zero | accepted M5 metric |
| `exact_clearance_m` | `[N,1]` | metres, nonnegative | reward-only privileged |
| `collision` | `[N,1]` bool | exact contact label | reward-only privileged |

Let `active = ||v_cmd_b|| > 1e-3`. Ordinary compliance is active only when
`beta >= 0.999` and emergency is false. Under legitimate attenuation or
emergency, primary tracking, proxy and preservation penalties are zero; the
intervention-usage term remains active.

| Component | Formula/default weight |
| --- | --- |
| `reward_actual_tracking` | `-1.0 * ordinary * ||actual_velocity_b-v_cmd_b||` |
| `reward_null_actual_speed` | `-2.0 * (1-active) * ||actual_velocity_b||` |
| `reward_anchor` | `-4.0 * (1-active) * anchor_active * (usable_fraction >= 0.1) * anchor_loss` |
| `reward_clearance` | `-0.5 * relu(1.0-exact_clearance_m)`; `1.0=0.8+0.2 m` |
| `reward_collision` | `-10.0 * collision` |
| `reward_intervention_usage` | `-0.05 * ((1-beta)+||v_gov_b-v_final_b||)` |
| `reward_issued_smoothness` | `-0.05 * ||v_final_b-previous_issued_v_final_b||` |
| `reward_governor_proxy` | `-0.1 * ordinary * ||v_gov_b-v_cmd_b||` |
| `reward_final_proxy` | `-0.25 * ordinary * ||v_final_b-v_cmd_b||` |
| `reward_preservation` | `-0.5 * ordinary * (relu(0.75-rho)+relu(rho-1.05))`, `rho=||v_final_b||/||v_cmd_b||` |

`reward_total` is the exact sum of the ten exposed components. There is no
clip, scale, normalization, hidden bonus or hidden component. The actual
tracking weight must remain positive, preventing a proxy-only configuration.
All actor-produced command inputs remain differentiable.

The anchor minimum usable fraction is an explicit finite configuration value
in `[0,1]`; its default is exactly accepted M5
`AnchorConfig.min_valid_fraction = 0.1`. Below threshold the anchor component is zero even for active plus
nonzero loss; the exact threshold is valid. The five values
`anchor_metrics[active/usable_fraction/huber_loss]` and
`ics_metrics[beta/emergency]` cross the M5-to-M6 boundary directly, with no
cast, synthetic label, fallback, clamp, or adapter. Floating state labels are
converted to masks only after exact validation inside `compute_reward`.

## Termination

Defaults are world-height closed safe interval `[0.5,4.0] m` and maximum 1,000
elapsed steps. Exact boundary heights remain valid. Independent masks are:

```text
collision
below_height = height_w_m < 0.5
above_height = height_w_m > 4.0
timeout = elapsed_steps >= 1000
```

`terminated = collision | below_height | above_height`; `truncated = timeout`;
both may be true simultaneously and `done` is their union. Stable reason codes
are none `0`, timeout `1`, above `2`, below `3`, collision `4`, with precedence
`collision > below > above > timeout > none`. There is no target position,
goal-distance or reach-goal input or termination.

## Critic-only schema

The builder accepts scalar-first `root_state_w [N,13]`, independently named
actual body velocity `[N,3]`, exact clearance `[N,1]`, and collision label. The
root ordering is position world metres, quaternion wxyz, world linear velocity
m/s, and world angular velocity rad/s. Quaternion norm tolerance is `1e-5`.

The output is a new `[N,18]` tensor in this immutable order:

```text
position_w_x/y/z_m,
quaternion_w/x/y/z,
linear_velocity_w_x/y/z_m_s,
angular_velocity_w_x/y/z_rad_s,
actual_linear_velocity_b_x/y/z_m_s,
exact_clearance_m,
collision_label
```

The builder accepts no actor or control container and cannot append to, alias
or mutate M4/M5 tensors. This is a single-critic input contract only; no critic
network or cost critic is implemented.

## Provenance and semantic differences

| Authority | Commit/path | SHA256 |
| --- | --- | --- |
| Active method lock | formal `CONTEXT.md` at `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3` | `dbb16ef0b23d84882d5f2ca30397bf2abeb28a0735562eed46953d7beb63181e` |
| Handbook | `docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex` | `2ea4dd3da74ef8be67ae4b4e7a5906e6b4e13537fe9b86514517fd7f092dbe24` |
| Legacy reward evidence | `isaac-training/training/scripts/instinctRL/rewards.py` | `b118e12d476c5e1741c2e727632d38c26d62bc1d23bcede5484904544b9c6e62` |
| Legacy termination evidence | `isaac-training/training/scripts/instinctRL/task_metrics.py` | `0cee0eb0f2c0c4c210586a42604ea53208cf122b622037f426288e9408433d2a` |
| Accepted M5 anchor interface | `instinctlab/control/anchor.py` at rework start | `2d73af1c2f58fee40e616826771b9e97b07758556f6ce5ab66b398bbcaea4c8d` |
| Accepted M5 ICS interface | `instinctlab/control/ics.py` at rework start | `87dba0e3f4d2e252b74fb8b89954b0673e07fe07748cf8b3a36dea6f7c03b6ac` |
| Accepted M5 pipeline interface | `instinctlab/control/pipeline.py` at rework start | `dd1e81f51f76422d7dfc6673dbf59c3c082b7fd9ceb12ff84a56272c745d1d2b` |

The active CONTEXT overrides the legacy optional proxy-primary behavior:
actual velocity is required and primary. Legacy benign defaults, label clamps,
`nan_to_num`, total clipping/scaling, optional privilege switch, goal branches
and fail-open repairs are excluded. Legacy weight/band/clearance values provide
defaults where consistent with the active method; they are not performance-
tuned in this Order.
