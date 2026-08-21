# M2 Plan

## Scope

Implement the smallest Isaac Lab `ArticulationCfg` for the unchanged TASLAB
USD and qualify a single CUDA articulation with deterministic gravity-only
spawn, reset, 32-step trajectory, full-state replay, and a second independent
process. This milestone does not implement flight or learning behavior.

## Product files

- `isaaclab-training/instinctlab/assets/taslab_uav/articulation.py`
- `isaaclab-training/tests/test_m2_static.py`
- `isaaclab-training/tests/run_m2_taslab_smoke.py`
- `isaaclab-training/tests/compare_m2_smoke.py`
- `isaaclab-training/README.md`
- The six M2 phase records

## Sequence

1. Preserve M1B asset/config identities and encode only spawn/root/actuator settings.
2. Prove configuration and forbidden-surface constraints statically.
3. Execute one persisted CUDA smoke twice in independent processes.
4. Compare replay and process results at the fixed `1e-5` tolerance.
5. Record effective properties, limitations, diffs, commits, and raw evidence.

## Fixed runtime contract

Device `cuda:0`; physics timestep `0.005 s`; initial root `(0, 0, 2) m`;
identity quaternion; zero root/joint velocities; four rotor joints at zero;
32 initial plus 32 replay steps per process; zero applied control.

## Stop conditions

Do not edit/convert the USD, retune physical data, patch runtime/packages, relax
gates, or add controller/sensor/actor/reward/PPO/training/M3 work.
