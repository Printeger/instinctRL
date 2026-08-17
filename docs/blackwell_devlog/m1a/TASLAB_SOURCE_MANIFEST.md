# TASLAB source manifest

This manifest records every TASLAB-specific source and each directly relevant
dependency/consumer/modern counterpart identified by `M1A-001`. Every row records
the required ten fields. `unknown` is used only where static evidence is
insufficient.

Evidence source: formal source `Printeger/instinctRL` `main@6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`;
raw evidence index `~/rl_dev/instinctRL_coordination/raw_logs/M1A-001/INDEX.md`.

## 1. `isaac-training/third_party/OmniDrones/omni_drones/robots/drone/taslab_uav.py`

1. Repository/authority: `Printeger/instinctRL` legacy project source
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3` (evidence read at governance base `a0619aa4…`, same blob)
3. Path: `isaac-training/third_party/OmniDrones/omni_drones/robots/drone/taslab_uav.py`
4. Git blob: `152795482b2b8345a8e23fdf831f00547e5f7224`
5. SHA-256: `05a3698fe3a3e5e74f9f699647e3c9962bc912529a8b403352b7351cf86969cf`
6. Size/type: `994` bytes, `Python script, Unicode text, UTF-8 text executable`
7. Intro/update: introduced `c4cca8b`, updated `8c9f80a` (`git log --follow`)
8. License: file header `MIT License / Copyright (c) 2024 TASLAB`; vendored OmniDrones LICENSE is `MIT / Copyright (c) 2023 Botian Xu, Tsinghua University` — conflict recorded
9. Runtime consumer: imported by `robots/drone/__init__.py`; selected by `training/cfg/drone.yaml` `model_name=TaslabUAV`
10. Authority: `runtime-effective` + `local patch`; ownership `unresolved` where headers conflict

## 2. `isaac-training/third_party/OmniDrones/omni_drones/robots/drone/__init__.py`

1. Repository/authority: `Printeger/instinctRL` legacy source
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `isaac-training/third_party/OmniDrones/omni_drones/robots/drone/__init__.py`
4. Git blob: `31c928753e3295a8e44f2f6980c6fdf8075069c0`
5. SHA-256: `78a597835fdc5bdb6c8713a9b7196209d609c8ba54519fd0a7f5fc2a88bc97cd`
6. Size/type: `1413` bytes, `Python script, ASCII text executable`
7. Intro/update: `2a46c06` vendored OmniDrones add; TASLAB import added by `c4cca8b`
8. License: vendored OmniDrones LICENSE `MIT / Copyright (c) 2023 Botian Xu, Tsinghua University`
9. Runtime consumer: imports `TaslabUAV`, triggering registry entry through `RobotBase.__init_subclass__`
10. Authority: `runtime-effective` / `third-party source` with local TASLAB import

## 3. `isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.yaml`

1. Repository/authority: `Printeger/instinctRL` legacy source
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.yaml`
4. Git blob: `babeb587f2db83764ca9812942e8e819d0813146`
5. SHA-256: `91b7319fd844303eb8e18f946434f3369988c8789e6ba0172b67bfd89177e3e4`
6. Size/type: `3457` bytes, `Unicode text, UTF-8 text`
7. Intro/update: introduced `c4cca8b`, updated `8c9f80a`
8. License: no file header; owner `unresolved` (configuration file)
9. Runtime consumer: `TaslabUAV.param_path` → `MultirotorBase.__init__` `yaml.safe_load`
10. Authority: `configuration intent`; values are runtime-effective through `RotorGroup`/controller

## 4. `isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.usd`

1. Repository/authority: `Printeger/instinctRL` legacy source
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.usd`
4. Git blob: `b16b1b16f21166d82579a9a5b190cb49191ddef5`
5. SHA-256: `3e733ca1008b1a3620941a8c1cc2e82bd7006a6a4c441c921a292bf11d2aee99`
6. Size/type: `9563347` bytes, `USD crate, version 0.8.0`
7. Intro/update: introduced in OmniDrones history `2a46c06`; updated by `8c9f80a` to current 9,563,347-byte blob
8. License: `unknown` (binary crate; no static license text discovered)
9. Runtime consumer: `TaslabUAV.usd_path` → `RobotBase._create_prim` USD reference
10. Authority: `authored asset` / `unresolved` ownership and license

## 5. `isaac-training/third_party/OmniDrones/omni_drones/controllers/cfg/lee_controller_taslab_uav.yaml`

1. Repository/authority: `Printeger/instinctRL` legacy source
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `isaac-training/third_party/OmniDrones/omni_drones/controllers/cfg/lee_controller_taslab_uav.yaml`
4. Git blob: `d7aab1f4705898fd20759f14e0a7b0cbfe2d3a37`
5. SHA-256: `e5e0496666f3f245d604a949eedc5060370d71b95de0d2337d39617bacd1f8e6`
6. Size/type: `1202` bytes, `Unicode text, UTF-8 text`
7. Intro/update: introduced `c4cca8b`, updated `8c9f80a`
8. License: no file header; owner `unresolved` (configuration file)
9. Runtime consumer: `LeePositionController.__init__` builds path `lee_controller_{uav_params['name']}.yaml`; `name` is `taslab_uav`
10. Authority: `configuration intent`; gains are runtime-effective

## 6. `isaac-training/third_party/OmniDrones/omni_drones/robots/drone/multirotor.py`

1. Repository/authority: `Printeger/instinctRL` legacy vendored OmniDrones
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `isaac-training/third_party/OmniDrones/omni_drones/robots/drone/multirotor.py`
4. Git blob: `7c2253789591cdf7c8889f8f1ddc20e7bc96c485`
5. SHA-256: `071d6a9d0b710c8e75229042a2612522dc00d01f7debe727dd4638a507da3477`
6. Size/type: `19462` bytes, `Python script, ASCII text executable`
7. Intro/update: vendored at `2a46c06` (history also `7cb5209`, `e970cbf`)
8. License: vendored OmniDrones LICENSE `MIT / Copyright (c) 2023 Botian Xu, Tsinghua University`
9. Runtime consumer: base class for `TaslabUAV`; loads YAML, builds `RotorGroup`, applies action/state
10. Authority: `runtime-effective` / `third-party source`

## 7. `isaac-training/third_party/OmniDrones/omni_drones/actuators/rotor_group.py`

1. Repository/authority: `Printeger/instinctRL` legacy vendored OmniDrones
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `isaac-training/third_party/OmniDrones/omni_drones/actuators/rotor_group.py`
4. Git blob: `21ff282fd4dce7a593fc437a0ac1ecbce46b31a6`
5. SHA-256: `645051c002ca53e46425108f549dd3ee7168f0e5f510423bc8f1dc72187ec4ec`
6. Size/type: `2781` bytes, `Python script, ASCII text executable`
7. Intro/update: vendored at `2a46c06`
8. License: vendored OmniDrones LICENSE `MIT / Copyright (c) 2023 Botian Xu, Tsinghua University`
9. Runtime consumer: constructed by `MultirotorBase.initialize`; `forward` returns thrust/moment from commands
10. Authority: `runtime-effective` / `third-party source`

## 8. `isaac-training/third_party/OmniDrones/omni_drones/utils/torchrl/transforms.py`

1. Repository/authority: `Printeger/instinctRL` legacy vendored OmniDrones
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `isaac-training/third_party/OmniDrones/omni_drones/utils/torchrl/transforms.py`
4. Git blob: `2699f7ffe5f0fda2be7638a096bdbabec80c567a`
5. SHA-256: `2ea4bac88b69cca73ec92930ef23d84ec710b06f83d6dc8210957fe65021d586`
6. Size/type: `15399` bytes, `Python script, ASCII text executable`
7. Intro/update: vendored at `2a46c06`
8. License: vendored OmniDrones LICENSE `MIT / Copyright (c) 2023 Botian Xu, Tsinghua University`
9. Runtime consumer: `VelController` wraps `LeePositionController` in training/eval transforms
10. Authority: `runtime-effective` / `third-party source`

## 9. `isaac-training/third_party/OmniDrones/omni_drones/robots/robot.py`

1. Repository/authority: `Printeger/instinctRL` legacy vendored OmniDrones
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `isaac-training/third_party/OmniDrones/omni_drones/robots/robot.py`
4. Git blob: `2d9644685dee27a14711a756aa910ef1fdf1a1be`
5. SHA-256: `fb27bb171808c93b7b71af59f61ed369f4812d8e93d6631a3a51c6a893fe1e66`
6. Size/type: `9508` bytes, `Python script, ASCII text executable`
7. Intro/update: vendored at `2a46c06`
8. License: vendored OmniDrones LICENSE `MIT / Copyright (c) 2023 Botian Xu, Tsinghua University`
9. Runtime consumer: provides `RobotBase.REGISTRY`, `spawn`, `_create_prim`, `initialize`
10. Authority: `runtime-effective` / `third-party source`

## 10. `isaac-training/third_party/OmniDrones/omni_drones/controllers/lee_position_controller.py`

1. Repository/authority: `Printeger/instinctRL` legacy vendored OmniDrones
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `isaac-training/third_party/OmniDrones/omni_drones/controllers/lee_position_controller.py`
4. Git blob: `06ab504820d460591dde5929677d04ad07689849`
5. SHA-256: `8606b4d0b2f4eb70e27d740a79e517dd484ed5b8820be29731b4048715b53fbc`
6. Size/type: `13799` bytes, `Python script, Unicode text, UTF-8 text executable`
7. Intro/update: vendored at `2a46c06`
8. License: vendored OmniDrones LICENSE `MIT / Copyright (c) 2023 Botian Xu, Tsinghua University`
9. Runtime consumer: instantiated by controller path; TASLAB gains loaded by `name`
10. Authority: `runtime-effective` / `third-party source`

## 11. `isaac-training/training/cfg/drone.yaml`

1. Repository/authority: `Printeger/instinctRL` legacy source
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `isaac-training/training/cfg/drone.yaml`
4. Git blob: `88bae956eb8f9527a9592e9aa87eed69a586599b`
5. SHA-256: `c2da932defd3d36909902e77443b7021e903c8847a1c1c83d87bd45151657eed`
6. Size/type: `1353` bytes, `Unicode text, UTF-8 text`
7. Intro/update: `unknown` for exact introduction from static log; runtime config in source tree
8. License: no file header; repository-level root LICENSE `MIT / Copyright (c) 2025 Zhefan Xu`
9. Runtime consumer: Hydra config selects `drone.model_name: "TaslabUAV"`
10. Authority: `configuration intent`

## 12. `isaac-training/training/scripts/env.py`

1. Repository/authority: `Printeger/instinctRL` legacy source
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `isaac-training/training/scripts/env.py`
4. Git blob: `3391ec58276e998bde96f147eaa5847fb03a4187`
5. SHA-256: `072295d5eff1707333dfdebd92557f4de4b5ccd89d92cf717949c44ceb82d4bb`
6. Size/type: `83808` bytes, `Python script, Unicode text, UTF-8 text executable`
7. Intro/update: not separately traced in M1A static log
8. License: no file header; repository-level root LICENSE `MIT / Copyright (c) 2025 Zhefan Xu`
9. Runtime consumer: builds `NavigationEnv`, resolves TASLAB base link dynamically
10. Authority: `runtime-effective` / local project code

## 13. `isaac-training/training/scripts/instinctRL/audit.py`

1. Repository/authority: `Printeger/instinctRL` legacy source
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `isaac-training/training/scripts/instinctRL/audit.py`
4. Git blob: `9ac002dd1e52632d4b75e42e2db4f8ffb62b0880`
5. SHA-256: `7bf51e9d9b8f1c39ebe64094914b626b268ce916a5c028932937c22ae8d2b49a`
6. Size/type: `10823` bytes, `Python script, Unicode text, UTF-8 text executable`
7. Intro/update: not separately traced in M1A static log
8. License: no file header; repository-level root LICENSE `MIT / Copyright (c) 2025 Zhefan Xu`
9. Runtime consumer: `check_platform_lock` asserts `drone.model_name == TaslabUAV`
10. Authority: `runtime-effective` / local project code

## 14. `isaac-training/training/scripts/instinctRL/task_metrics.py`

1. Repository/authority: `Printeger/instinctRL` legacy source
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `isaac-training/training/scripts/instinctRL/task_metrics.py`
4. Git blob: `124b0cbe80fb6b6391e574f6c091433580a3a44b`
5. SHA-256: `0cee0eb0f2c0c4c210586a42604ea53208cf122b622037f426288e9408433d2a`
6. Size/type: `86027` bytes, `Python script, ASCII text executable`
7. Intro/update: not separately traced in M1A static log
8. License: no file header; repository-level root LICENSE `MIT / Copyright (c) 2025 Zhefan Xu`
9. Runtime consumer: body/world-frame velocity rotation helper for TASLAB controller path
10. Authority: `runtime-effective` / local project code

## 15. `docs/TASLAB_UAV_README.md`

1. Repository/authority: `Printeger/instinctRL` legacy source
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `docs/TASLAB_UAV_README.md`
4. Git blob: `cc61056505600a81b7b0103c580af0416fb884e9`
5. SHA-256: `93af4bc9986eb777704eba20915747984f8f5b038aa8a1535578522b0624d745`
6. Size/type: `2715` bytes, `Unicode text, UTF-8 text`
7. Intro/update: `unknown` for exact introduction from static log
8. License: no file header; repository-level root LICENSE `MIT / Copyright (c) 2025 Zhefan Xu`
9. Runtime consumer: historical instructions only, not runtime-effective
10. Authority: `historical documentation`

## 16. `docs/instinctRL_0_platform_sensor_audit.md`

1. Repository/authority: `Printeger/instinctRL` legacy source
2. Exact commit: `6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`
3. Path: `docs/instinctRL_0_platform_sensor_audit.md`
4. Git blob: `5216a37372023e295c6bdc2b9bc9d4421ac410bb`
5. SHA-256: `538f2bb550bf5af8ce15dea2e70948624f6ff17bd43d50372a84afe6f7558f94`
6. Size/type: `28323` bytes, `Unicode text, UTF-8 text, with very long lines (517)`
7. Intro/update: `unknown` for exact introduction from static log
8. License: no file header; repository-level root LICENSE `MIT / Copyright (c) 2025 Zhefan Xu`
9. Runtime consumer: historical audit; contains stale ~158KB USD statement
10. Authority: `historical documentation`

## 17. Modern OmniDrones TASLAB_UAV absence

1. Repository/authority: `~/rl_blackwell_test/repos/OmniDrones` `blackwell-feasibility`
2. Exact commit: `5045c71cc73d224144cc08bc38d65cabeedaec8c`
3. Path: any `TaslabUAV` / `taslab_uav` / `TASLAB` path
4. Git blob: `N/A` — proven absence by `git ls-files` and `rg`
5. SHA-256: `N/A`
6. Size/type: `N/A`
7. Intro/update: `N/A`
8. License: `N/A`
9. Runtime consumer: none
10. Authority: `proven absent`

## 18. Modern `omni_drones/robots/robot.py`

1. Repository/authority: `~/rl_blackwell_test/repos/OmniDrones` `blackwell-feasibility`
2. Exact commit: `5045c71cc73d224144cc08bc38d65cabeedaec8c`
3. Path: `omni_drones/robots/robot.py`
4. Git blob: `a17cc02095f2a2a439199c26328c2f5e9787ff26`
5. SHA-256: `2f2a332be1ec504dfee6fa9000ddaee6ad3380c2c3539baecc60ce62e52fe3de`
6. Size/type: `9658` bytes, `Python script, ASCII text executable`
7. Intro/update: not separately traced in M1A static log
8. License: modern LICENSE `MIT / Copyright (c) 2023 Botian Xu, Tsinghua University`
9. Runtime consumer: modern registry/spawn API; `TaslabUAV` absent
10. Authority: `runtime-effective` / modern comparison reference

## 19. Modern `omni_drones/robots/drone/multirotor.py`

1. Repository/authority: `~/rl_blackwell_test/repos/OmniDrones` `blackwell-feasibility`
2. Exact commit: `5045c71cc73d224144cc08bc38d65cabeedaec8c`
3. Path: `omni_drones/robots/drone/multirotor.py`
4. Git blob: `c96cb017c0faa0b47a6d1f9c8da1cc03d057d2b3`
5. SHA-256: `39ea4cfe2a6748a04a7d13b3ae4e60c606ea75251be95f643ec011ac546a5930`
6. Size/type: `18548` bytes, `Python script, ASCII text executable`
7. Intro/update: not separately traced in M1A static log
8. License: modern LICENSE `MIT / Copyright (c) 2023 Botian Xu, Tsinghua University`
9. Runtime consumer: modern multirotor base; no `TaslabUAV`
10. Authority: `runtime-effective` / modern comparison reference

## 20. Modern `omni_drones/actuators/rotor_group.py`

1. Repository/authority: `~/rl_blackwell_test/repos/OmniDrones` `blackwell-feasibility`
2. Exact commit: `5045c71cc73d224144cc08bc38d65cabeedaec8c`
3. Path: `omni_drones/actuators/rotor_group.py`
4. Git blob: `539cad2de88fa9ab72c4c1af4cc3a362dfeac109`
5. SHA-256: `4b7e794b53c9385b0dbed4dd1fd2b47aa7c299c2f42db82a9babe2dfc72b9400`
6. Size/type: `3248` bytes, `Python script, ASCII text executable`
7. Intro/update: not separately traced in M1A static log
8. License: modern LICENSE `MIT / Copyright (c) 2023 Botian Xu, Tsinghua University`
9. Runtime consumer: modern `MultirotorBase` constructs batched `RotorGroup`
10. Authority: `runtime-effective` / modern comparison reference

## 21. Modern `omni_drones/controllers/lee_position_controller.py`

1. Repository/authority: `~/rl_blackwell_test/repos/OmniDrones` `blackwell-feasibility`
2. Exact commit: `5045c71cc73d224144cc08bc38d65cabeedaec8c`
3. Path: `omni_drones/controllers/lee_position_controller.py`
4. Git blob: `4f8af6e51215c5d80aea5969a01e897788de1b64`
5. SHA-256: `331df4beec0515d69e472a13e92281d1cd319c1bcaa70a32019d152aff2c4933`
6. Size/type: `13447` bytes, `Python script, ASCII text executable`
7. Intro/update: not separately traced in M1A static log
8. License: modern LICENSE `MIT / Copyright (c) 2023 Botian Xu, Tsinghua University`
9. Runtime consumer: `VelController` calls `LeePositionController.compute` in modern stack
10. Authority: `runtime-effective` / modern comparison reference

## 22. Modern `omni_drones/utils/torchrl/transforms.py` `VelController`

1. Repository/authority: `~/rl_blackwell_test/repos/OmniDrones` `blackwell-feasibility`
2. Exact commit: `5045c71cc73d224144cc08bc38d65cabeedaec8c`
3. Path: `omni_drones/utils/torchrl/transforms.py`
4. Git blob: `c193f224312e7297afcfc982059ae34f191ce4f9`
5. SHA-256: `841de861e2bde09dd0acdfad87faafea7b4e3ae832b49d59d38a98ed759a20dc`
6. Size/type: `14567` bytes, `Python script, ASCII text executable`
7. Intro/update: modern branch port of NavRL `VelController` (`5045c71`)
8. License: modern LICENSE `MIT / Copyright (c) 2023 Botian Xu, Tsinghua University`
9. Runtime consumer: modern velocity action transform; `TaslabUAV` absent
10. Authority: `runtime-effective` / modern comparison reference

## 23. Isaac Lab `ArticulationCfg` / `Articulation` and `RigidObjectCfg` / `RigidObject`

1. Repository/authority: `~/rl_blackwell_test/repos/IsaacLab` v2.3.2
2. Exact commit: `37ddf626871758333d6ed89cf64ad702aef127d0`
3. Path: `source/isaaclab/isaaclab/assets/articulation/articulation_cfg.py`, `articulation.py`, `rigid_object/rigid_object_cfg.py`, `rigid_object.py`
4. Git blob: not individually recorded for this summary row; class declarations are in `cmd_020`
5. SHA-256: not individually recorded for this summary row
6. Size/type: source Python; static class declarations
7. Intro/update: Isaac Lab v2.3.2 snapshot
8. License: Isaac Lab repo license not individually copied here; frozen baseline reference only
9. Runtime consumer: future M1B Isaac Lab asset migration candidate
10. Authority: `runtime-effective` / modern comparison reference

## 24. Isaac Lab actuator APIs

1. Repository/authority: `~/rl_blackwell_test/repos/IsaacLab` v2.3.2
2. Exact commit: `37ddf626871758333d6ed89cf64ad702aef127d0`
3. Path: `source/isaaclab/isaaclab/actuators/actuator_pd_cfg.py`, `actuator_pd.py`
4. Git blob: not individually recorded for this summary row
5. SHA-256: not individually recorded for this summary row
6. Size/type: source Python; classes `ImplicitActuatorCfg`, `IdealPDActuatorCfg`, `DCMotorCfg`
7. Intro/update: Isaac Lab v2.3.2 snapshot
8. License: Isaac Lab repo license not individually copied here
9. Runtime consumer: future M1B actuator mapping candidate
10. Authority: `runtime-effective` / modern comparison reference
