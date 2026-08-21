# M5 Pure-Tensor Control-Core Contract

Order: `M5-001`

## Information boundary and sequence

The actor-clean surface is exactly `lidar_grid [N,12,59,360]`, `state_vec
[N,52]`, normalized `governor_action [N,4]`, finite positive `history_dt_s`,
and two explicit boolean reset masks. The only controller-private value is a
scalar-first unit `controller_quaternion_wxyz [N,4]` accepted by the final
adapter.

The sequence is M4 audit, latest-command extraction, governor decode/smoothing/
norm limit, latest-measurement anchor update, four-frame mounted-ray ICS
attenuation, final body-command norm limit, then body-to-world rotation. The
quaternion cannot affect actor observation, governor, anchor, rays, ICS,
`v_final_b`, or any pre-adapter metric/cache.

## Governor

For `action` in the closed unit hypercube:

```text
alpha = action[:,0:1]
v_corr_b = (2 * action[:,1:4] - 1) * v_corr_limit
candidate = alpha * v_cmd_b + v_corr_b
```

Optional smoothing uses only M4's previous actually issued body command:
`candidate = (1-smoothing)*candidate + smoothing*previous_b`. Default
`smoothing=0`. Euclidean clipping preserves direction and enforces the positive
`velocity_limit`. B0 returns alpha one, zero correction, and only the same norm
limit.

## Measurement anchor

Defaults are `eps_enter=0.05 m/s`, `eps_exit=0.15 m/s`, minimum usable fraction
`0.1`, minimum reliability `0.1`, and Huber delta `0.25 m`. A null command
captures the latest M4 range/validity/reliability tensors only on inactive to
active transition; the dense reference remains internal and immutable while
held. Usability requires current and captured validity plus both reliability
thresholds.

Reset precedence from highest to lowest is episode (`1`), explicit (`2`),
minimum-valid (`4`), command exit (`3`), then none (`0`). Reset is selective by
environment. Metrics are only active, reset code, usable fraction, weighted
mean absolute error, and weighted standard Huber loss.

## Mounted rays and ICS

All 21,240 rays are regenerated in M3 elevation-outer/azimuth-fastest order
from 59 elevations and 360 azimuths, then rotated by M3's scalar-first +30-degree
pitch mount quaternion into TASLAB body frame. M4 history is extracted as four
oldest-to-newest range/validity/reliability frames; all-zero initialization is
excluded by validity and reliability.

Defaults are `d_safe=0.8 m`, emergency clearance `0.25 m`, `a_max=2.0 m/s^2`,
active-horizon margin `0.5 m`, minimum reliability `0.1`, velocity limit
`2.0 m/s`, zero latency and clearance margin, newest-pair finite difference,
range-rate filtering disabled, empty-set beta one, and a zero brake command.
Optional filtering deterministically averages the usable consecutive rates in
the four-frame window and combines closing evidence with line-of-sight command
approach speed.

```text
effective_clearance = range - |v_gov_b| * latency - clearance_margin
v_safe = sqrt(2 * a_max * max(effective_clearance - d_safe, 0))
beta = min_active clamp(v_safe / closing_speed, 0, 1)
v_final_b = norm_limit(beta * v_gov_b)
```

Invalid/stale beams are excluded, an empty active set preserves the command,
and any reliable emergency beam sets beta zero. No pose, translational velocity,
surface normal, height, obstacle truth, map, SLAM state, or simulator state is
accepted.

## Adapter and atomicity

The adapter accepts same-batch/device/dtype finite tensors and normalizes only
quaternion norm drift within `1e-5`; larger drift fails closed. It applies the
public scalar-first tensor formula equivalent to `q*v*q_conjugate` and preserves
command norm. M4 audit, action/reset/history interval, anchor inputs, rays, and
quaternion are preflighted before the only state mutation. Failed calls leave
anchor and last ICS evidence cache unchanged.
