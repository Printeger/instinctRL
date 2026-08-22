# M5 TASLAB Runtime-Control Contract

Order: `M5-002`, revision 1

## Boundary and tensors

```text
M3 ranges/mask/time -> M4 lidar_grid [N,12,59,360], state_vec [N,52]
fixed normalized action [N,4] -> accepted M5-001 -> v_final_w [N,3]
v_final_w + controller-private root_state_w [N,13] -> Lee controller
raw affine rotor coordinates [N,4] -> rotor boundary -> local wrenches
```

All controller tensors are non-empty float32/float64 batches on one device and
must be finite. `root_state_w` ordering is position metres `[0:3]`, scalar-first
body-to-world quaternion `[3:7]`, world linear velocity m/s `[7:10]`, and world
angular velocity rad/s `[10:13]`. Quaternion norm tolerance is `1e-5`.

Root state is permitted only after `v_final_w` at this conventional low-level
boundary. It is absent from M4 and the exact actor-clean modules `governor.py`,
`anchor.py`, `ics.py`, `command_adapter.py`, and `pipeline.py`. Actor-visible
keys remain exactly `lidar_grid` and `state_vec`.

## Lee velocity controller

Velocity-only mode copies current position as the held position target and
extracts current yaw from the scalar-first quaternion. With world velocity
`v`, target `v*`, gain `K_v=(2.2,2.2,2.2)`, and gravity `g=9.81 m/s^2`:

```text
a = K_v * (v - v*) - [0,0,g]
b3_des = -a / ||a||
b1_des = [cos(yaw), sin(yaw), 0]
b2_des = normalize(b3_des x b1_des)
R_des = [b2_des x b3_des, b2_des, b3_des]
e_R = vee(0.5 * (R_des^T R - R^T R_des))
ang_acc = -(K_R / I_diag) * e_R - (K_omega / I_diag) * omega_body
thrust = -mass * dot(a, R[:,2])
```

Exact mass is `1.1583 kg`; diagonal controller inertia is
`(0.0051350687, 0.0056075235, 0.0052743931) kg m^2`; attitude gain is
`(0.7,0.7,0.035)` and angular-rate gain is `(0.1,0.1,0.025)`.

For rotor angles `theta`, arms `l`, directions `d`, force constants `KF`, and
moment constants `KM`, the allocation and mixer are:

```text
A = [sin(theta)*l; -cos(theta)*l; -d*KM/KF; 1]
mixer = A^T (A A^T)^-1 diag(Ixx,Iyy,Izz,1)
command = mixer [ang_acc_x,ang_acc_y,ang_acc_z,thrust]^T
coordinate = 2 * command / (KF*omega_max^2) - 1
```

The coordinate is deliberately not controller-clamped and is not claimed to
be intrinsically bounded. Degenerate/non-finite computation raises; there is
no `nan_to_num`, default substitution, or exception swallowing.

## Rotor and wrench boundary

Rotor order is exactly `rotor_0..rotor_3`, with directions
`(+1,-1,+1,-1)`, `KF=1.55e-6`, `KM=2.39e-8`, and
`omega_max=2261 rad/s` for each rotor. Noise is disabled. For coordinate `u`:

```text
target = sqrt(clamp((u+1)/2, 0, 1))
throttle_next = throttle + 0.43 * (target - throttle)
force_i = KF_i * omega_max_i^2 * throttle_next_i^2
moment_i = KM_i * omega_max_i^2 * throttle_next_i^2 * (-direction_i)
```

The clamp exists only at this physical actuator boundary. Throttle state is
per environment and selective reset clears throttle plus pending wrench
buffers.

Articulation topology must be exactly `(base_link, rotor_0, rotor_1, rotor_2,
rotor_3)` on the adapter device. Each rotor force is placed on its body local
`+Z`; reaction moments are summed and placed on `base_link` local `+Z`.
`set_external_force_and_torque(..., is_global=False)` stages the complete
wrench buffers and `write_data_to_sim()` precedes every physics step. No joint
target, drag, downwash, randomization, or guessed dynamics are added.

## Runtime composition

One TASLAB Articulation and accepted static-plane MID360 run on `cuda:0` at
`dt=0.005 s`. The M3/M4 interval is `0.1 s`. Four 64-step scenarios cover zero
command/B0, positive body X/B0, 90-degree yaw plus positive body X, and exact
reset/replay. Each control step executes accepted M3, M4, M5-001, Lee, rotor,
wrench write, and physics update in order. Complete policy and physics tensors
are persisted; independent processes are compared recursively at `1e-5`.

## Provenance and license

The Lee and rotor modules retain the MIT notice and copyright from Botian Xu,
Tsinghua University. Adapted behavior was reviewed at:

| Authority | Commit | Path | SHA256 |
| --- | --- | --- | --- |
| Formal legacy | `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3` | `.../controllers/lee_position_controller.py` | `8606b4d0b2f4eb70e27d740a79e517dd484ed5b8820be29731b4048715b53fbc` |
| Formal legacy | same | `.../actuators/rotor_group.py` | `645051c002ca53e46425108f549dd3ee7168f0e5f510423bc8f1dc72187ec4ec` |
| Formal legacy | same | `.../robots/drone/multirotor.py` | `071d6a9d0b710c8e75229042a2612522dc00d01f7debe727dd4638a507da3477` |
| Formal legacy | same | `.../utils/torchrl/transforms.py` (`VelController`) | `2ea4bac88b69cca73ec92930ef23d84ec710b06f83d6dc8210957fe65021d586` |
| Frozen OmniDrones | `5045c71cc73d224144cc08bc38d65cabeedaec8c` | `omni_drones/controllers/lee_position_controller.py` | `331df4beec0515d69e472a13e92281d1cd319c1bcaa70a32019d152aff2c4933` |
| Frozen OmniDrones | same | `omni_drones/actuators/rotor_group.py` | `4b7e794b53c9385b0dbed4dd1fd2b47aa7c299c2f42db82a9babe2dfc72b9400` |
| Isaac Lab API | `37ddf626871758333d6ed89cf64ad702aef127d0` | `source/isaaclab/isaaclab/assets/articulation/articulation.py` | `fb179aacdd848906367d01105020e1d36412bd0f1da7187be09f0781ff589309` |

The product copies equations and behavior, not runtime imports. Differences
are strict validation, explicit scalar-first helpers, stateless Lee output,
preview/commit atomicity, project-local config access, public Isaac Lab wrench
buffers, and exclusion of TorchRL transforms, drag/downwash, and noise.
