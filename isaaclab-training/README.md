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
