# TASLAB original vs modern diff

## Scope

Static, symbol/config/asset-contract-level comparison only. Textual similarity is
not behavioral parity. USD stage introspection and runtime execution are M1B
validation items.

## 1. TASLAB class declaration and registry

| Aspect | Legacy `Printeger/instinctRL` | Modern OmniDrones `5045c71…` | Difference |
| --- | --- | --- | --- |
| Class | `class TaslabUAV(MultirotorBase)` in `taslab_uav.py` | absent | TASLAB must be ported; not an upstream toggle |
| Registry | `RobotBase.REGISTRY` populated by `__init_subclass__`; import in `drone/__init__.py` | same `RobotBase.REGISTRY` mechanism; `drone/__init__.py` has no TASLAB import | registry mechanism compatible; class/config/asset missing |
| Name casing | class `TaslabUAV`; registry also registers lowercase `taslabuav`; config value `"TaslabUAV"` | N/A | record exact casing and lowercase alias |
| Config selection | `drone.model_name: "TaslabUAV"` in `training/cfg/drone.yaml` | no TASLAB config/consumer | M1B must add a modern model config |

## 2. Drone import and config selection

- Legacy: `robots/drone/__init__.py` imports `TaslabUAV`; Hydra config selects
  `drone.model_name=TaslabUAV`.
- Modern: `robots/drone/__init__.py` imports only `Crazyflie`, `Firefly`,
  `Hummingbird`, `Iris`, `Neo11`, `Omav`, `Dragon`; no `TaslabUAV`.
- Evidence: `raw_logs/M1A-001/cmd_015`, `cmd_014`, `cmd_023`.

## 3. YAML schema and values

Legacy `taslab_uav.yaml` effective values:

| Field | Value | Unit |
| --- | --- | --- |
| name | `taslab_uav` | string |
| mass | `1.1583` | kg |
| inertia | `xx=0.0051350687`, `xy=0.0000000049`, `xz=0.0002842395`, `yy=0.0056075235`, `yz=0.0000000307`, `zz=0.0052743931` | kg·m² |
| l | `0.24042` | m |
| drag_coef | `0.2` | s⁻¹ |
| num_rotors | `4` | count |
| arm_lengths | `[0.12021, 0.12021, 0.12021, 0.12021]` | m |
| rotor_angles | `[5.49778714, 0.78539816, 2.35619449, 3.92699082]` | rad |
| directions | `[1.0, -1.0, 1.0, -1.0]` | sign |
| force_constants | `[1.55e-06, 1.55e-06, 1.55e-06, 1.55e-06]` | N/(rad/s)² |
| moment_constants | `[2.39e-08, 2.39e-08, 2.39e-08, 2.39e-08]` | N·m/(rad/s)² |
| max_rotation_velocities | `[2261, 2261, 2261, 2261]` | rad/s |

Conflicts/comments to track:

- YAML comments say "对于 1.2kg 的无人机" and list battery alternatives
  (`3000mah: 1.144kg`, `4000mah: 1.227kg`), while effective `mass` is `1.1583`.
- Comment claims `updated to measured calibration data (2026-02-07)`, but
  `arm_lengths` and `force_constants` still contain TODO/replace-with-measured
  language.
- Commented-out alternate inertia block remains in file; only active values are
  authoritative for YAML load.
- USD-authored mass/inertia/topology is not statically introspected; YAML and USD
  authority is unresolved and belongs to M1B stage inspection.

Modern OmniDrones has no `taslab_uav.yaml`; its asset directory contains only
standard models (`cmd_023`).

## 4. USD crate identity

| Aspect | Legacy | Modern |
| --- | --- | --- |
| Path | `.../assets/usd/taslab_uav.usd` | absent |
| Blob | `b16b1b16f21166d82579a9a5b190cb49191ddef5` | N/A |
| SHA-256 | `3e733ca1008b1a3620941a8c1cc2e82bd7006a6a4c441c921a292bf11d2aee99` | N/A |
| Size | `9563347` bytes | N/A |
| Type | `USD crate, version 0.8.0` | N/A |

Static `file`/Git evidence proves crate identity and size only. Root/default
prim, rigid bodies, joints, transforms, mass/inertia authorship, rotor prim
count/names/order/axes, and collision APIs are **M1B USD stage introspection
requirements**, not M1A claims.

## 5. Lee gains and loading path

Legacy TASLAB gains in `lee_controller_taslab_uav.yaml`:

- `position_gain: [4, 4, 4]`
- `velocity_gain: [2.2, 2.2, 2.2]`
- `attitude_gain: [0.7, 0.7, 0.035]`
- `angular_rate_gain: [0.1, 0.1, 0.025]`

Lookup path: `LeePositionController.__init__` constructs
`lee_controller_{uav_params['name']}.yaml`; `uav_params['name']` is
`taslab_uav`, so it resolves to `lee_controller_taslab_uav.yaml`.

Modern `LeePositionController` uses the same lookup convention and the same
gain schema, but there is no modern `lee_controller_taslab_uav.yaml`
(`cmd_023`).

## 6. `RotorGroup`

Legacy:

- scalar parameters: `KF`, `KM`, `throttle`, `directions`, `tau_up`, `tau_down`
- `forward(cmds)` uses in-place `self.throttle.add_`
- `make_functional(self.rotors)` exposes `rotor_params`; `MultirotorBase` expands
  them to environment shape.

Modern:

- constructor accepts `batch_shape`; `KF`, `KM`, `directions`, `tau_up`,
  `tau_down`, and `throttle` are batched `nn.Parameter`s.
- `forward` computes `new_throttle` and copies it into `self.throttle.data`
  instead of using `add_`.
- `MultirotorBase` references rotor parameters directly instead of
  `make_functional`.

Input/output semantics remain conceptually `command -> thrust/moment`, but the
batching/state ownership differs. Numerical parity is not proven.

## 7. `MultirotorBase`

Legacy differences vs modern:

- legacy imports `functorch.vmap`, `tensordict.nn.make_functional`,
  `BoundedTensorSpec`/`CompositeSpec`/`UnboundedContinuousTensorSpec`, and
  `LeePositionController`.
- modern imports `torch.func.vmap`, `Bounded`/`Composite`/`Unbounded`, and
  `ControllerBase`.
- legacy has `MultirotorCfg(force_sensor=...)` and `DEFAULT_CONTROLLER`;
  modern removes `force_sensor` state dimension path and uses a lazy
  `action_spec` property.
- modern `apply_action` reorders base-body force application before rotor force
  application for Isaac Sim 5.1 (documented in source comment).
- legacy `get_state` has force-sensor optional branch; modern `get_state` does
  not include that branch.

Shared behavior: YAML load, rotor view expression `rotor_*`, world/body
velocity conversion, state vector composition `[pos(3), rot(4), vel(6),
heading(3), up(3), throttle(n_rotors)]`, and reset throttle logic are broadly
the same shape.

## 8. `VelController`

Legacy `VelController`:

- when `yaw_control=True`, splits action into `target_vel` and `target_yaw`,
  applies `unsqueeze(1)` to both, multiplies yaw by `pi`.
- calls `self.controller(drone_state, target_vel=..., target_yaw=...)`.

Modern `VelController`:

- explicitly documented as a port for the PR#106 modern stack.
- splits yaw without `unsqueeze(1)`.
- calls `self.controller.compute(drone_state, target_vel=..., target_yaw=...)`.

The removal of the legacy `unsqueeze(1)` is a meaningful shape-semantics
difference. Controller saturation, frame semantics, and NaN handling require
runtime/unit validation and are not proven by static diff.

## 8b. Static controller/rotor contract

The following facts are visible in static source and must be preserved in any
M1B adapter:

- Learned action input key: `("agents", "action")` in both legacy and modern
  `VelController`.
- Controller-only privileged state key: `("info", "drone_state")`; both
  transforms read only the first 13 dimensions `[pos(3), quat(4), vel(3),
  ang_vel(3)]`. This state is used by the controller/adapter boundary and is not
  an authorized actor observation field.
- Motor-command overwrite: `VelController._inv_call` replaces
  `tensordict[("agents", "action")]` with the controller's normalized rotor
  commands after computing them.
- Normalized motor-output path: `LeePositionController` computes
  `acc` from position/velocity error and gravity, derives desired body axes,
  computes angular acceleration and thrust, forms
  `ang_acc_thrust = [ang_acc, thrust]`, then calculates
  `cmd = (mixer @ ang_acc_thrust.T).T`, followed by
  `cmd = (cmd / max_thrusts) * 2 - 1`. That is the normalized `[-1, 1]` rotor
  command written back to the action key.
- Clamping/saturation location: `VelController` does not clamp the normalized
  motor command before writing it. `RotorGroup.forward` later clamps the
  normalized command indirectly with `target_throttle =
  f_inv(clamp((cmds + 1) / 2, 0, 1))`. The controller's command construction
  divides by per-rotor `max_thrusts`, but no explicit `[-1, 1]` clamp appears in
  `VelController` or `LeePositionController._compute`.
- Existing NaN handling: both legacy and modern `VelController` call
  `torch.nan_to_num_(cmds, 0.)` immediately before overwriting the action key.
  This is inherited behavior/risk to document; M1A does not endorse it as a
  correctness workaround and M1B must not silently copy it as a physics fix.
- Device/dtype construction: controller transforms and the local
  `BodyToWorldVelocityAdapter` are moved to `cfg.device`; rotor/controller
  parameters are created as `torch` tensors/`nn.Parameter`s from YAML values.
  Inputs come from the TensorDict on the environment device.
- Rotor ordering and force/moment application: `RotorGroup` returns per-rotor
  `thrusts` and `moments` in YAML list order. `MultirotorBase.apply_action`
  expands actions to `(..., num_rotors)`, obtains world poses for
  `rotor_*`, derives each rotor torque axis from its world rotation, sets
  `self.thrusts[..., 2] = thrusts`, computes
  `self.torques = sum(moments * torque_axis)`, and applies rotor forces/torques
  at rotor positions. Legacy applies rotor forces before base-body forces;
  modern applies base-body forces before rotor forces because of the recorded
  Isaac Sim 5.1 force-overwrite behavior.
- Unresolved rotor-order mapping: the YAML `arm_lengths`, `rotor_angles`, and
  `directions` lists are consumed by `compute_parameters` in
  `lee_position_controller.py`; the binary USD prim enumeration and its exact
  correspondence to `rotor_*` view order are not statically proven. This remains
  an M1B USD stage-introspection requirement.
- Legacy vs modern controller API: legacy `VelController` calls
  `self.controller(...)` and legacy `LeePositionController` exposes `forward`;
  modern `VelController` calls `self.controller.compute(...)`, and modern
  `LeePositionController` inherits from `ControllerBase`, which declares
  `compute` and `process_rl_actions`.

## 9. Legacy Isaac Sim/Orbit vs modern Isaac Lab

Legacy `robot.py` imports `omni.isaac.core.*` and
`omni.isaac.core.simulation_context.SimulationContext`.

Modern OmniDrones imports `isaacsim.core.*` and
`isaacsim.core.api.simulation_context.SimulationContext`, plus
`SimulationContext.get_physics_context().get_gravity()`.

Isaac Lab 2.3.2 offers the modern asset/actuator surface:

- `isaaclab.assets.Articulation` / `ArticulationCfg`
- `isaaclab.assets.RigidObject` / `RigidObjectCfg`
- `isaaclab.actuators.ImplicitActuator` / `ImplicitActuatorCfg`
- `isaaclab.actuators.IdealPDActuator` / `IdealPDActuatorCfg`
- `isaaclab.actuators.DCMotor` / `DCMotorCfg`

These are candidates for a future M1B mapping; no behavioral equivalence is
claimed.

## 10. Required surfaces not statically proven

The following are marked `NOT PROVEN / M1B`:

- USD loadability in modern runtime
- prim path/root/default prim/body/rotor topology
- mass/inertia/joint/transform authority between YAML and USD
- rotor ordering, axes, direction-to-rotor mapping
- controller numerical parity and saturation
- command/world/body frame conversion at runtime
- actor-contract compliance
- license clearance for binary USD
