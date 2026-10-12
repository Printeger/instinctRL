# B0 horizontal root-cause successor contract

This is an additive successor to paused package `INSTINCTRL-ZERO-COMMAND-BASELINE-20260927`.
Do not resume, edit, or consume authority from that package. Do not edit any old
attempt or old analysis report.

## Frozen facts

- Vertical drift mechanism is established: Lee mass 1.1583 kg versus runtime
  articulated mass 1.1815179586 kg predicts `vz=-0.08938184245956034 m/s`;
  measured pre-reset steady `vz=-0.0893858325551264 m/s`.
- The B0 Lee path reads articulation root state directly; IMU noise is not in
  its feedback path.
- Steady horizontal drift is approximately `vx=+0.1934 m/s`,
  `vy=-0.0559 m/s`; its cause is not established.
- No mass/control/dynamics parameter may be changed before the one-shot live
  diagnostic. No P0, PLR, training, policy evaluation, optimizer update, or
  checkpoint write is authorized.

## New identities

- Workload repository:
  `/home/mint/rl_dev/instinctRL_orchestrator/implementation_records/instinctrl-first-use-20260915/state/workspaces/instinctrl-b0-horizontal-successor-20260927`
- Required future attempt namespace:
  `/home/mint/rl_dev/instinctRL_coordination/attempts/development/UNIFIED-H1-R4/UNIFIED-H1-R4-B0-HORIZONTAL-DIAG-R1`
- Package ID: `INSTINCTRL-B0-HORIZONTAL-SUCCESSOR-20260927`
- Live experiment ID: `UNIFIED-H1-R4-B0-HORIZONTAL-DIAG`
- Analysis output namespace:
  `/home/mint/rl_dev/instinctRL_orchestrator/implementation_records/instinctrl-first-use-20260915/state/analyses/UNIFIED-H1-R4-E0002/J-880cd798ee89d6eb/20260927-b0-horizontal-root-cause-successor`

## Phase 1: read-only instrumentation

Implement a workload-side observer for the existing B0 live path. Prefer a
post-physics callback or another Isaac-supported read-only hook. It must not
write any control, articulation, simulation, reset, random, timing, or actor
state. Record each physics step and lane with case, policy step, physics step,
episode ID, and reset-boundary mapping.

Required observations:

- articulation root state;
- full articulated-system COM pose, linear velocity, and angular velocity;
- every rigid body's name, mass, local COM transform, world transform and
  world velocity; total mass and reconstructed combined COM;
- Lee target velocity, velocity error, desired rotation and collective;
- each rotor's command, target/actual throttle, thrust, reaction moment,
  world application point/direction and lever arm about system COM;
- gravity, rotor, drag/aerodynamic, contact, joint/constraint and other
  available wrenches, all reduced to the system COM;
- an explicit availability record for every wrench class. An unavailable
  class is `UNAVAILABLE`, never a numeric zero;
- momentum-difference inferred net/residual wrench with exact finite-difference
  timing and closure terms;
- reset mask/reason and the last pre-reset plus first several post-reset
  controller/rotor internal states.

The existing B0 target velocity remains exactly zero. Z/T/S, seed 20260909,
initial state, curriculum, control rate, policy rate, safety thresholds and
reset behavior remain frozen.

## CPU tests

At minimum prove schema, shapes, dtypes, finiteness, lane/physics-step
continuity, body-mass sum, COM reconstruction, rotor force-to-COM wrench
translation, known/residual wrench closure, reset episode separation,
fail-closed missing fields, explicit unavailable fields, and absence of writes
to control state. Tests must run without Isaac/GPU.

The Engineer must include every changed file in its change ledger and use the
supported semantic category `ENGINEERING_FIX` for diagnostic code/tests. The
Evaluator must independently use the same supported category; descriptive
labels such as `ENGINEERING_DIAGNOSIS` are not valid classifier categories.

## Prelaunch review gate

The independent Evaluator must verify that instrumentation is observational,
does not reorder physics/control/reset operations, distinguishes COM offset,
fixed force, fixed torque, constraints, drag, rotor/frame defects and reset
state, passes CPU tests, and respects the 16 GiB output bound. Any non-PASS
verdict is terminal `PRELAUNCH_REVIEW_FAILED`; no live run and no retry.

## One-shot live authority after PASS only

- one live process start, one simulator initialization;
- Z/T/S, seed 20260909, zero B0 target velocity;
- at most 20,160 nontraining transitions;
- zero training transitions, optimizer updates, checkpoint writes, P0/PLR,
  policy evaluation and automatic retry;
- at most 910 s wall, 910 GPU allocation seconds, 16 GiB diagnostics;
- no retry after preflight, launch, runtime, timeout, or cleanup failure.

The live command must use the formal unified-H1 runner and the exact accepted
instrumentation candidate. It must produce durable raw physics-step evidence,
offline causal analysis, budget ledger and cleanup evidence.

## Post-live decision

Numerically close force, torque, COM, pose, velocity, angular velocity and
acceleration before claiming causality. Determine the sustained horizontal
wrench source, whether x/y share the z mass cause, reset clearing, and whether
one minimal repair exists. Only then may a repair candidate be prepared. A
repair must remove the mass double source vertically and change only the
specific diagnosed horizontal defect. It may not loosen gates, blindly tune
gains/add integral action, or add simulator truth to the actor.

Final outcomes are restricted to:

- `ROOT_CAUSE_ESTABLISHED_FIX_PREPARED`
- `ROOT_CAUSE_ESTABLISHED_NO_SAFE_FIX`
- `DIAGNOSIS_INCONCLUSIVE`
- `PRELAUNCH_REVIEW_FAILED`
- `LIVE_DIAGNOSTIC_FAILED`

There is no authorization for a repaired live rerun.
