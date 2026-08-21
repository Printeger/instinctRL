# M4 Actor-Clean Observation Contract

## Exact input boundary

`ActorObservationHistory.update` accepts exactly eight explicit tensors:

| Input | Shape | Dtype | Units/frame |
| --- | --- | --- | --- |
| `ranges_m` | `[N,21240]` | float32/float64 | metres, exact M3 serialized order/domain |
| `validity` | `[N,21240]` | bool | exact M3 mask bits |
| `sensor_frame_time_s` | `[N,1]` | matching float | seconds |
| `policy_time_s` | `[N,1]` | matching float | seconds |
| `body_angular_velocity_rad_s` | `[N,3]` | matching float | rad/s, body frame |
| `gravity_direction_body` | `[N,3]` | matching float | unit direction, body frame |
| `commanded_body_velocity_m_s` | `[N,3]` | matching float | m/s, body frame |
| `previous_issued_body_velocity_m_s` | `[N,3]` | matching float | m/s, body frame |

No full-state, generic mapping, extra-field, or keyword catch-all input exists.
Reset indices enter only through `reset`, never through the actor data path.

## Exact output boundary

The output has exactly two keys:

- `lidar_grid`: `[N,12,59,360]`, float, four oldest-to-newest frames with
  interleaved `range_m`, `validity`, `reliability` channels.
- `state_vec`: `[N,52]`, float, four oldest-to-newest 13-field frames.

Flat M3 index `elevation_index * 360 + azimuth_index` maps directly to
`[elevation_index, azimuth_index]`; there is no transpose or permutation.

## Timing, reliability, history, and reset

`age_s = policy_time_s - sensor_frame_time_s` must be finite and nonnegative.
Both timestamps are nondecreasing per environment. Repeated sensor timestamps
are permitted when policy time is nondecreasing. Reliability is exactly
`validity.float() * exp(-age_s / tau_s)`, clamped to `[0,1]`, for finite positive
configured `tau_s`; invalid reliability is exactly zero. Gravity norm tolerance
defaults to `1e-5` and is finite, positive, and configurable.

Initial older slots are zero and the first frame occupies the newest slot.
Rollover removes only the oldest frame. Selective reset clears only named
environments and their time provenance; full reset clears all history and time
provenance. Returned tensors are detached copies and cannot mutate internal history.

## Machine-readable output-slice provenance

<!-- PROVENANCE_JSON_BEGIN -->
```json
[
  {"output":"lidar_grid","slice":"[:,0::3,:,:]","source":"ranges_m","history":"oldest_to_newest"},
  {"output":"lidar_grid","slice":"[:,1::3,:,:]","source":"validity.float","history":"oldest_to_newest"},
  {"output":"lidar_grid","slice":"[:,2::3,:,:]","source":"validity.float*exp(-age_s/tau_s)","history":"oldest_to_newest"},
  {"output":"state_vec","slice":"[:,0:52:13]+[0:3]","source":"body_angular_velocity_rad_s","history":"oldest_to_newest"},
  {"output":"state_vec","slice":"[:,0:52:13]+[3:6]","source":"gravity_direction_body","history":"oldest_to_newest"},
  {"output":"state_vec","slice":"[:,0:52:13]+[6:9]","source":"commanded_body_velocity_m_s","history":"oldest_to_newest"},
  {"output":"state_vec","slice":"[:,0:52:13]+[9:12]","source":"previous_issued_body_velocity_m_s","history":"oldest_to_newest"},
  {"output":"state_vec","slice":"[:,12:52:13]","source":"policy_time_s-sensor_frame_time_s","history":"oldest_to_newest"}
]
```
<!-- PROVENANCE_JSON_END -->

Every slice derives only from an approved input or the documented deterministic
age/reliability/history transform. No translational state is reconstructed.
