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
