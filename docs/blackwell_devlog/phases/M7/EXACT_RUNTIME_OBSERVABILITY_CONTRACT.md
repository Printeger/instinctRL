# M7-005 Exact Runtime Observability Contract

## Environment source

`extras.evaluation_ray_source` has this exact order:

1. `ray_starts_w`: float32 `[N,R,3]`, world frame, finite, detached exact-storage owner.
2. `ray_directions_w`: float32 `[N,R,3]`, world frame unit directions, finite, detached exact-storage owner.
3. `accepted_ray_hits_w`: float32 `[N,R,3]`, world frame; valid hits finite and misses positive infinity.
4. `accepted_beam_valid`: bool `[N,R]`, the accepted MID360 closed-range validity mask.
5. `mesh_path`: absolute world prim path.
6. `maximum_distance_m`: finite positive SI scalar.
7. `source_step`: int64 `[N]`, positive global transition identity.
8. `reset_mask`: bool `[N]`, the same pre-reset row mask.

The source is captured once during `_refresh_task_snapshot` at the sensor update that supplies clearance and validity. `_get_rewards` adds the now-known reset mask and exposes owned clones before automatic reset. The later returned-observation sensor update cannot overwrite it. The exact M6 `pre_reset` mapping and policy/critic mappings are unchanged.

## Acquisition boundary

Only `instinctlab.evaluation.raycaster_observability` may resolve `RayCaster.meshes`, import Warp raycast, or hold surface normals. It verifies frozen Isaac Lab head `37ddf626871758333d6ed89cf64ad702aef127d0`, version `2.3.2`, the accepted API signature, and the three accepted source SHA-256 values before mesh resolution. Cast validity must exactly equal accepted MID360 validity. A finite cast rejected by the `0.1 m` MID360 minimum-range rule therefore fails closed. Valid hit and distance errors must not exceed `1e-5 m`; raw miss infinities remain inside acquisition and are never serialized.

## Collector boundary

`ExactRuntimeObservability` contains only rank, ascending three-value spectrum, `sigma_min`, condition value/validity, weak world direction/validity, valid/total counts and fraction, hit/distance errors, source step/reset mask, and source/result SHA-256 byte tensors. Every tensor has exact shape/dtype/device, detached exact-storage ownership, mutual non-aliasing, and declared finiteness. The result seal detects post-distillation mutation.

The collector requires source step `k` for transition `k`, exact reset-mask equality, and a source identity not previously consumed. Thus stale, duplicate, skipped, reordered, wrong-step, wrong-mask, post-reset-substituted, aliased, or mutated evidence fails closed.

Canonical mapping is:

- `observability_value = sigma_min`;
- `observability_valid = true`;
- `observability_source = EXACT (2)`.

Exact zero is valid exact evidence, not unavailable and not a proxy.

## Actor isolation

The actor retains exactly `lidar_grid` and `state_vec`. Raw rays, normals, exact results, source identities, mesh/cache state, pose, velocity, simulator state, and all evaluation details are absent from actor and critic inputs. No evaluation value feeds reward, control, task, PPO, or training.
