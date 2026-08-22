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

## M6 pure-tensor task contract

`instinctlab.task` defines reward, termination, and critic-only tensor
contracts without an Isaac Lab environment dependency. Primary nonzero-command
tracking requires privileged actual body-frame velocity; collision, exact
clearance, world height, and root state remain reward/termination/critic-only.
The actor surface remains exactly M4 `lidar_grid` plus `state_vec`, and the
accepted M5 command/control path does not import or consume task privilege.

M6-001 contains no actor or critic network, PPO/GAE, optimizer, collector,
rollout, replay, checkpoint, environment wiring, evaluation, training, or M7
work. Its CUDA smoke is pure tensor computation and does not launch Isaac Sim.

## M6 DirectRLEnv integration

`instinctlab.envs.TaslabDirectEnv` composes the accepted TASLAB articulation,
MID360, exact M4 policy observation, non-B0 M5 governor/control/runtime path,
and M6 reward, termination, and critic-only state. Physics runs at `0.005 s`,
one normalized `[N,4]` action is held for 20 physics steps, and each returned
policy branch contains only `lidar_grid [N,12,59,360]` and `state_vec [N,52]`.
The separate critic branch is the accepted privileged `[N,18]` tensor.

The fixed-action smoke is an environment integration qualification only. It
contains no policy/critic network, PPO/GAE, optimizer, collection framework,
checkpointing, evaluation, randomization, curriculum, or training.

## M6 pure-tensor PPO core

`instinctlab.training` adds an environment-independent feedforward actor over
exact M4 `lidar_grid [N,12,59,360]` plus `state_vec [N,52]`, one structurally
separate critic over exact M6 privileged `[N,18]`, a four-dimensional bounded
Beta policy, timeout-aware GAE, deterministic PPO-Clip minibatch updates, and
strict atomic checkpoints. The bounded CUDA qualification uses one persisted
synthetic four-sample batch and exactly two update calls per independent
process. It does not import or step the environment, collect experience, claim
learning quality, or authorize simulator training, evaluation, M7, 128k, 1M,
or scale runs.

## M6 bounded learning-pipeline integration

`instinctlab.training.rollout` is the sole adapter between the accepted
`TaslabDirectEnv` observation/action/extras surface and the accepted PPO tensor
core. It clones detached pre-action policy and critic inputs, passes only the
exact M4 pair to the actor, passes only `[N,18]` privilege to the critic, steps
the environment with the sampled normalized Beta action, reconstructs pure-
timeout value input from `extras["pre_reset"]`, computes terminal-safe GAE, and
flattens `[T,N]` in explicit time-major order without dropping samples.

The qualification is fixed to two environments, two four-step segments, eight
policy steps / 160 physics steps per process, 16 total samples, two PPO update
calls, and one checkpoint reload. It exercises one public automatic timeout
and finite post-reset continuation. It is not a production trainer, does not
checkpoint simulator state, and does not authorize evaluation, M7, 128k, 1M,
or scale training.
