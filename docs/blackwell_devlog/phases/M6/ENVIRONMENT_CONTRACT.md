# M6-002 Environment Contract

## Public API provenance

All files below were read from frozen Isaac Lab commit
`37ddf626871758333d6ed89cf64ad702aef127d0` and were not copied or modified.

| Frozen file | SHA256 | Use |
| --- | --- | --- |
| `source/isaaclab/isaaclab/envs/direct_rl_env.py` | `3a7573303d83ec8c816d11ca6e8c8997c92f6765ae5d2e0d2a369e65d2cec3c0` | step/reset hook order and public buffers |
| `source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py` | `d99fe6bcd607d2b1a326eea30b8e858e4af04659c085981625bf08c5d8a49ca4` | config and Gym space API |
| `source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env.py` | `819f066c8c42a76bb1d9c6724cee7f1671a05f46efeba76d433695d58e4f6183` | public scene/articulation/sensor composition pattern |
| `source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env_cfg.py` | `27054865401dd0ad4244f454201887f41212d65e539e00106d689d0f4c4efe56` | cloned prim and sensor config pattern |
| `source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py` | `43a28b82050363e3a1c1c5e6e872fe6eada7bdda19b8992a111bb0a9638bcbef` | public ContactSensor configuration |
| `source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py` | `281e5b6d4028b5c8c58305b1ae525eb6328052652f338295a3a08ed0185a7307` | public `net_forces_w` producer |
| `source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_data.py` | `f07add496ff30947c9a8770e540b6c049eefade8378cfafec439dad9cd3c3487` | public pose and `ray_hits_w` producers |

## Configuration and spaces

- device: `cuda:0`;
- physics interval: `0.005 s`;
- decimation: `20` physics steps;
- policy interval: `0.1 s`;
- default vector size: two environments, `4 m` spacing;
- action: float32 `[N,4]`, finite, already on the environment device, closed
  `[0,1]`, copied only after complete validation;
- action decoding: accepted `ControlCoreConfig(use_b0=False)` path;
- policy: exactly `lidar_grid [N,12,59,360]` and `state_vec [N,52]`;
- critic: a separate accepted `[N,18]` tensor in exact `CRITIC_FIELD_ORDER`.

## Frames, units, and producers

`root_state_w` is the public articulation root state: world position in metres,
scalar-first body-to-world quaternion, world linear velocity in metres/second,
and world angular velocity in radians/second. Reward/critic actual body velocity
is derived only after physics by applying the inverse root quaternion to public
world linear velocity.

MID360 uses the accepted 21,240-ray order, body mount translation `(0,0,0.05) m`,
scalar-first +30 degree pitch quaternion, `[0.1,40.0] m` domain, and invalid
`40.0 m` fill. `RayCaster.data.ray_hits_w`, its public tracked parent pose, and
the accepted mounted-pose/range functions produce M4 ranges/validity.
Privileged exact clearance is the minimum finite serialized simulator RayCaster
range in metres. It is unavailable to action/control and is passed only to M6
reward/critic and logging.

Collision is the boolean result of any TASLAB ContactSensor public
`net_forces_w` body-force norm exceeding `1.0 N`. The label is unavailable to
the actor/control path and enters only reward, termination, critic, and extras.

## Step information flow

```text
validate/copy [N,4] action
  -> accepted M4 policy tensors + reset masks -> non-B0 M5 control once
  -> hold v_final_w for 20 iterations
  -> public root state -> Lee controller -> rotor/wrench every physics step
  -> scene update at 0.005 s
  -> one post-step root/RayCaster/ContactSensor snapshot
  -> M6 done, then M6 reward from the same cached snapshot
  -> reset done indices
  -> refresh MID360/M4 and return post-reset policy + separate critic
```

The five direct M5 reward metrics are the original tensor objects at the
environment boundary: anchor `huber_loss`, `active`, `usable_fraction`, and ICS
`beta`, `emergency`. No cast, relabel, or synthetic replacement intervenes.

## Reset contract

Reset indices are rank-one, unique, in range, `torch.long`, and on the
environment device before mutation. Full/selective reset restores public root
pose/velocity and joint state, resets the articulation and scene sensors,
clears rotor/wrench state, clears only selected M4 history, replaces only
selected M5 pipeline/anchor instances, zeros actions/previous issued command
and task/control caches, resets elapsed steps, and marks reset inputs for the
next control boundary. Unselected public state, policy cache, rotor state, and
episode counters remain unchanged by the reset operation itself.

## Privilege boundary

The returned `policy` dictionary is recursively restricted to the two M4 keys.
Root state, pose, actual velocity, clearance, contact, critic, reward, reason,
controller values, wrench, and rotor values are absent. The `critic` tensor and
diagnostic `extras` cannot be read by `_pre_physics_step`, M5 control, Lee
target selection, or rotor action generation.
