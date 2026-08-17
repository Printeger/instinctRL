# M1A TASLAB provenance

## Purpose

This document is the static, evidence-backed provenance package for the legacy
`TaslabUAV` platform and its modern comparison targets. It is part of order
`M1A-001`. It does not migrate, copy, convert, import, or execute any runtime.

## Executive conclusion

`TaslabUAV` is a project-specific platform derived from the AechPro Isaac Sim 5.1
OmniDrones fork, plus local feasibility patches. It is **not** an official modern
OmniDrones model. The frozen modern OmniDrones tree contains no `TaslabUAV`,
`taslab_uav`, or `TASLAB` path/symbol.

Evidence:

- Legacy source witness: `Printeger/instinctRL` `main` at
  `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`; migration governance base
  `a0619aa4e60ecc23a016d339f5c32fa4a98ad967` has the same TASLAB blobs.
- Modern OmniDrones comparison pin: local qualified repository
  `~/rl_blackwell_test/repos/OmniDrones`, branch `blackwell-feasibility`,
  commit `5045c71cc73d224144cc08bc38d65cabeedaec8c`.
- Modern OmniDrones TASLAB absence search returned no matches in tracked paths or
  repository text (`raw_logs/M1A-001/cmd_014`, `cmd_023`).

## Source identity and authority

| Subject | Static evidence | Authority |
| --- | --- | --- |
| Formal source | remote `git@github.com:Printeger/instinctRL.git`, `main@6f6dee3…` | Authoritative legacy project source |
| Legacy vendored OmniDrones | `isaac-training/third_party/OmniDrones` under the same source commit | Third-party source copied into the project |
| Modern OmniDrones comparison | `~/rl_blackwell_test/repos/OmniDrones` `blackwell-feasibility@5045c71…`; branch tracking `aechpro/main` | Qualified frozen comparison snapshot |
| Isaac Lab comparison | `~/rl_blackwell_test/repos/IsaacLab` v2.3.2 `37ddf62…` | Qualified frozen modern API reference |

The modern OmniDrones checkout has two remotes:

- `origin` = `https://github.com/btx0424/OmniDrones.git`
- `aechpro` = `https://github.com/AechPro/OmniDrones.git`

The qualified branch is tracking `aechpro/main`. The exact upstream URL of the
local `blackwell-feasibility` commit is not stated in the qualification reports;
that ambiguity is recorded as a risk and is not resolved by guessing.

## What the legacy runtime consumes

The static call/config chain is:

1. `training/cfg/drone.yaml` sets `drone.model_name: "TaslabUAV"`.
2. The environment resolves the robot class through `RobotBase.REGISTRY`.
3. `omni_drones/robots/drone/__init__.py` imports `TaslabUAV`, causing
   `__init_subclass__` to register `TaslabUAV` and `taslabuav`.
4. `TaslabUAV` supplies `usd_path` and `param_path`.
5. `MultirotorBase.__init__` reads `taslab_uav.yaml`; `initialize` creates a
   `RotorGroup`, `RigidPrimView`s, and rotor/body state buffers.
6. The controller path is `VelController(LeePositionController(...))`; the
   TASLAB gains are loaded from `lee_controller_taslab_uav.yaml`.
7. Training/eval consumers select `TaslabUAV` through Hydra config and platform
   audit in `instinctRL/audit.py`.

This chain is supported by `raw_logs/M1A-001/cmd_015` and the manifest.

## Required inventory summary

The full 10-field inventory is in
`TASLAB_SOURCE_MANIFEST.md`. The minimum TASLAB-specific set is:

- `omni_drones/robots/drone/taslab_uav.py`
- `omni_drones/robots/assets/usd/taslab_uav.yaml`
- `omni_drones/robots/assets/usd/taslab_uav.usd`
- `omni_drones/controllers/cfg/lee_controller_taslab_uav.yaml`
- `omni_drones/robots/drone/multirotor.py`
- `omni_drones/actuators/rotor_group.py`
- `omni_drones/utils/torchrl/transforms.py` (`VelController`)
- `omni_drones/robots/robot.py` (registry/spawn)
- training/config consumers and historical docs listed in the manifest

## USD size conflict

The current frozen TASLAB USD is a binary USD crate, not a 158 KB text asset:

- path: `isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.usd`
- Git blob: `b16b1b16f21166d82579a9a5b190cb49191ddef5`
- SHA-256: `3e733ca1008b1a3620941a8c1cc2e82bd7006a6a4c441c921a292bf11d2aee99`
- bytes: `9,563,347`
- `file` type: `USD crate, version 0.8.0`

The stale statement in `docs/instinctRL_0_platform_sensor_audit.md` describes the
same path as `Binary USD visual/articulation model (~158KB)`. The object evidence
proves the current reviewed file is 9,563,347 bytes. This document does not
rewrite history; the conflict is tracked here as a stale historical statement.

## Authority conflicts and risk register

Each conflict is recorded without silent resolution.

| Risk | Evidence | Impact | Resolution owner / milestone | Acceptance check |
| --- | --- | --- | --- | --- |
| TASLAB_UAV ownership | `taslab_uav.py` header says `Copyright (c) 2024 TASLAB`; vendored OmniDrones LICENSE says `Copyright (c) 2023 Botian Xu, Tsinghua University` | License/provenance ambiguity | Commander review before M1B copy/conversion | Confirm source author/license for local TASLAB files |
| Binary USD author and license | USD crate has no static embedded license discovered by `file`/`git`; directory name cannot establish owner | Binary redistribution/compatibility risk | Commander/legal review before asset reuse | Confirm USD provenance separately |
| TASLAB YAML vs USD authority | `taslab_uav.yaml` states `mass: 1.1583`; USD mass/inertia/topology is not statically introspected | Runtime mass/inertia may differ | M1B USD stage introspection | Compare USD-authored mass/inertia/rotor topology with YAML |
| Stale 158 KB statement | `docs/instinctRL_0_platform_sensor_audit.md` vs 9,563,347-byte blob | Historical documentation error | M1A records, no rewrite | Blob/SHA/size/type evidence |
| Rotor prim topology/order/axes | Not statically proven from binary USD | Rotor mapping risk | M1B USD stage introspection | Enumerate prims, joints, rotor order/axes |
| Command frame | Legacy `VelController` expects world-frame velocity; project governor emits body-frame command and uses an adapter | Controller-frame semantic risk | M5 control path | Frame conversion tests at named boundary |
| Upstream modern pin URL | Qualification evidence gives full commit and local path but not a canonical upstream URL | Reproducibility uncertainty | Commander baseline refresh decision | Record exact qualified local path/full object; do not guess URL |
| Isaac Lab baseline dirty file | `apps/isaaclab.python.headless.kit` is modified in frozen IsaacLab checkout | Read-only baseline evidence nuance | Recorded, not changed | Confirm pre-existing status; no product impact |

## Static-only boundary

The following are **not proven** by M1A:

- USD stage loadability or prim/joint/transform introspection
- runtime mass/inertia values actually applied by USD
- physical hover/tracking parity
- controller numerical parity
- actor-contract runtime behavior
- MID360 geometry/runtime

Those items require separately authorized M1B or later validation.

## Evidence links

- Raw log index: `~/rl_dev/instinctRL_coordination/raw_logs/M1A-001/INDEX.md`
- Key command IDs referenced here: `001`, `002`, `003`, `004`, `010`–`024`
