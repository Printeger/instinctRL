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

## M7 pure-tensor evaluation core

`instinctlab.evaluation` audits the exact M4 policy mapping before consuming a
separate, exactly ordered evaluation-only `[T,N]` mapping. It emits explicit
tracking, null-command station, anchor, clearance, collision, ICS,
command-preservation, intervention, observability-source, and independent
termination counts as canonical JSON. Conditional metrics retain numerator,
denominator, valid count, total count, and an explicit unavailable state.

M7-001 uses only persisted synthetic tensors. It does not launch Isaac Sim,
collect a rollout, evaluate policy quality, implement or run B0-B8 ablations,
compute a range-Jacobian, train, or authorize M8, 128k, 1M, or scale work.

`instinctlab.evaluation.runtime` is the bounded M7-002 bridge. Its persisted
smoke connects exactly two accepted M6 environments for eight policy steps,
uses only `HistoryActor.mean_action`, and separates terminal pre-reset records
from post-reset actor observations and station references.

`instinctlab.evaluation.observability` is a pure evaluation-only M7 core. It
computes the exact translational range Jacobian from caller-supplied world-frame
unit surface normals and validity masks. It does not acquire simulator hits,
wire the runtime collector, define a hardware proxy, or expose normals to the
actor, control, environment, reward, critic, or training paths. The M7-002
runtime observability fields therefore remain explicitly `UNAVAILABLE`.

`instinctlab.evaluation.raycaster_observability` is an explicitly imported,
evaluation-runtime-only adapter around the frozen Isaac Lab/Warp exact
normal-producing ray cast. Its bounded qualification re-casts the accepted
live MID360 world-frame rays against the same cached mesh and validates exact
sensor hits, distances, validity, raw miss convention, unit face normals, and
the unchanged M7 exact-J handoff. It is intentionally not re-exported by the
pure evaluation package. Private RayCaster buffers are accessed only by the
hash-guarded qualification runner; no environment or sensor API changes, actor
inference, runtime-record wiring, proxy, or training behavior are introduced.

## M7 baseline isolation manifest

`instinctlab.evaluation.baselines` is a standard-library-only, non-executing
registry for the exact Handbook B0-B8 identities. It records each row's policy
and evaluation schemas, component toggles, privilege/deployability boundary,
capability evidence, optionality, and current readiness, then builds immutable
canonical per-run manifests with deterministic SHA-256 provenance. Only the
accepted B0 direct-velocity capability is currently ready; B1-B8 remain
non-executable, and unavailable optional B8 must be explicitly `SKIPPED` with a
reason. This layer does not implement an adapter, select a fallback, execute a
scenario or ablation, or make a policy-performance claim.

## M7 pure B0 direct adapter

`instinctlab.control.baseline_b0` is a non-integrated, stateless B0 execution
primitive. Its actor-approved surface is exactly commanded body velocity. A
separately named controller-private scalar-first quaternion is used only after
that boundary to rotate the exact norm-limited body command into the world
frame. The module adds no learned action, observation/history, sensor, anchor,
ICS, task, reward, evaluator, environment, simulator, or controller wiring and
does not execute a baseline or scenario.

## M7 manifest-gated B0 execution boundary

`instinctlab.evaluation.baseline_execution.execute_ready_b0` is a pure,
non-integrated dispatch boundary. It accepts only an exact, self-verifying B0
`READY` manifest with the seven accepted capability bindings, invokes the
accepted direct adapter once, and returns an immutable envelope with the
unchanged READY manifest, its exact `EXECUTED` successor, and the owned adapter
result. It is not wired to an environment, controller, runtime collector, or
scenario and does not claim that a baseline run occurred.

## M7 B0 TASLAB controller handoff

`instinctlab.evaluation.baseline_controller_execution` is a stateless,
manifest-gated bridge from the command-only B0 boundary to the accepted TASLAB
Lee controller. Root state is explicitly controller-private and never enters
the actor schema or manifest. Qualification is limited to one-articulation,
constant-command headless CUDA wiring; it is not environment integration,
scenario evaluation, baseline performance, or training.

## M7 B0 multi-step run lifecycle

`instinctlab.evaluation.baseline_run` owns one exact canonical B0 `READY`
manifest for a configured positive number of sequential controller-handoff
steps. It stages successor, counter, and an injectively framed trajectory
SHA-256 on every step, then finalizes once to an immutable receipt containing
only independent READY/EXECUTED manifests, step count, schema, and hash. It
retains no command, root state, or output tensor and does not integrate an
environment, collector, scenario, simulator, metrics, or performance logic.

## M7 command-only B0 TASLAB environment mode

`instinctlab.evaluation.baseline_environment` is a direct-import-only,
evaluation-only subclass boundary over the accepted `TaslabDirectEnv`. It owns
one exact lifecycle session, accepts only detached finite body-velocity commands
`[N,3]`, reuses each command for the exact 20 physics substeps, and applies only
the returned Lee rotor command through the accepted wrench adapter. Finalization
after the explicit bounded horizon returns the private lifecycle receipt plus
scalar coverage. The mode does not run a learned actor/control pipeline,
collector, scenario campaign, metric comparison, training, or performance gate.
