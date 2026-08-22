# instinctRL Isaac Lab migration

This tree is the independent Blackwell/Isaac Lab migration root. M1B contains
only the byte-preserved TASLAB platform asset, its declarative configuration
contract, and static/read-only topology qualification. It does not register or
spawn an Isaac Lab articulation and contains no controller or simulation loop.

## M2 minimal articulation smoke

`instinctlab.assets.taslab_uav.articulation` provides the bounded Isaac Lab
`ArticulationCfg` for the byte-preserved TASLAB USD. The persisted M2 smoke
launches one headless CUDA articulation at 2 m and records two 32-step
gravity-only trajectories. It applies no rotor force, target, controller,
sensor, actor, reward, PPO, or training behavior and does not claim hover or
flight parity.

## M3 MID360 contract

`instinctlab.sensors.mid360` defines the fixed 21,240-ray MID360 geometry,
TASLAB `base_link` mount, and explicit raw-range/serialized-range/validity
contract. The bounded M3 smoke uses one static ground plane and one sensor to
qualify shape, ordering, range bounds, valid/miss handling, and repeatability;
it does not introduce actor observations, control, reward, PPO, or training.
Revision 1 composes the tracked `base_link` pose with the configured mount pose
before calculating ranges and checks every return against an independent
world-z=0 tiled-plane oracle. Parent and mounted sensor poses remain separately
identified in evidence.

## M4 actor-clean observation boundary

`instinctlab.observations` consumes only the qualified flat M3 ranges and mask,
explicit sensor/policy times, approved body angular velocity and gravity cues,
the body-frame command, and the previous issued body-frame command. It maintains
four oldest-to-newest frames and returns exactly `lidar_grid [N,12,59,360]`
plus `state_vec [N,52]`. The pure-PyTorch M4 tests and CUDA smoke do not launch
Isaac Sim or Isaac Lab and add no actor network, controller, reward, PPO,
rollout, evaluation, or training behavior.

## M5 pure-tensor control core

`instinctlab.control` consumes only the exact audited M4 observation, a bounded
four-dimensional candidate action, reset masks, and the qualified history
interval. It deterministically composes body-frame governor decoding,
measurement-space anchoring, mounted-MID360 history attenuation, and bounded
body-frame output. Scalar-first attitude is a separate controller-private input
to the final body-to-world adapter and cannot affect any earlier result or cache.

M5-001 does not instantiate an actor network, controller, articulation,
simulator, reward, PPO loop, checkpoint, rollout, evaluation, or training job.

## M5 TASLAB runtime-control boundary

M5-002 consumes M5-001 `v_final_w` only after the actor-clean pipeline and
combines it with controller-private `root_state_w` in a local, MIT-notice-
preserving TASLAB Lee velocity controller. Exact four-rotor throttle response
then produces local +Z rotor thrusts and a summed base-link local +Z reaction
moment through Isaac Lab's public Articulation wrench API. Root state, pose,
true velocity, attitude, angular rate, and controller internals never become M4
fields or actor-visible results.

The bounded qualification uses one environment on `cuda:0`, `dt=0.005 s`, a
`0.1 s` policy/history interval, four fixed scenarios, and 256 physics steps
per process. It introduces no actor network, reward, critic, PPO, collector,
checkpoint, rollout API, evaluation, training, parameter tuning, or M6 work.
