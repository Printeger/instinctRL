# M2 Final Report

## Outcome

The unchanged TASLAB USD loads as one floating Isaac Lab articulation on
`cuda:0`. Both independent processes passed topology, finite-state, gravity,
drift, in-process replay, and cross-process determinism gates with zero applied
control.

## Configuration

The adapter uses `UsdFileCfg(copy_from_source=False)`, gravity enabled,
gyroscopic forces enabled, maximum depenetration velocity 10, self-collisions
disabled, solver iterations 4/0, sleep threshold 0.005, stabilization threshold
0.001, and one zero-stiffness/zero-damping actuator restricted to
`rotor_[0-3]_joint`. Initial root pose is `(0,0,2)` with identity quaternion;
all velocities and rotor joint states are zero.

## Runtime result

Each process created one CUDA `SimulationContext` at `dt=0.005 s`, spawned one
articulation, restored complete state, ran 32 gravity-only steps, restored it
again, and replayed 32 steps. Each process therefore executed exactly 64
physics steps. The two normalized JSON outputs are byte-identical.

## Boundary declaration

No force, torque, wrench, joint target, rotor command, controller, ground,
light, camera, sensor, actor input, reward, PPO, training, or M3 component was
created or called. The M1B USD/YAML/gains and frozen runtime were unchanged.

Final acceptance remains a Commander decision.
