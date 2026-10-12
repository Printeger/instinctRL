# R5 diagnostic closure — current evidence index

Started 2026-10-12 at main `ebc27e2e47d4190efccac9a9179d4e291bd933da`; remote main was the same commit (git ls-remote). Development cwd is `/home/mint/rl_dev/instinctRL-dev`. Initial untracked `docs/goal/` is retained; its supplied requirements are copied without changes to `docs/goals/R5_DIAGNOSTIC_CLOSURE_REQUIREMENTS.md`.

Goal remains IN_PROGRESS. Target is READY_FOR_PAIRED_PILOT_WITH_FIRST_ROLLOUT_GATE. No technical readiness or live release is claimed by this index.

| Evidence | Identity / conclusion | Treatment |
| --- | --- | --- |
| `docs/history/consolidation-20261012/r5/task_inputs/prior/ZERO_COMMAND_BASELINE_FIX_AND_VALIDATION_REPORT.md` | Historical B0: 20,160 interactions; vertical mass mechanism supported; horizontal unresolved | Raw data and asset binding still to verify; no repeat launch justified yet |
| `docs/history/consolidation-20261012/r5-live-failure/B0_ENHANCED_LIVE_ROOT_CAUSE_REPORT.md` | Most recent enhanced B0 failed at first reset; zero physics rows/transitions | Failure preserved; not a control baseline |
| `docs/history/consolidation-20261012/r5/r5_delivery/R5_CONTROL_CHAIN_DIAGNOSIS.md` | NOT_STARTED template | Not a result |
| `coordination/PROJECT_STATE.json` historical review | BUDGET_EXHAUSTED; MUST_FIX result schema rejected | Not resumed or relabelled accepted |
| `pilot-cpu-v1/` | Real-tensor regression initially failed: scalar tensor byte view in state digest | Failure retained; prompted a minimal diagnostic digest fix |
| `pilot-cpu-v2/` | 5 real CPU tensor cases and 10 existing pilot cases pass | No optimizer steps, no live simulation; not full §5 or technical acceptance |

Actual tools: Python shell interpreter 3.14.6; Codex CLI 0.161.0. CPU tests use existing product interpreter from tools/run_cpu_checks.py with CUDA_VISIBLE_DEVICES empty. The frozen isolation contract remains 0.153.4; installed-CLI compatibility is not yet certified. No dependencies installed or frozen runtime changed.

Observed additional open defects: analyzer crosses reset boundaries, groups only case/lane, compares root acceleration to articulated momentum, and promotes heuristic ranking to causal verdict. Product observer response is root-based; changing the analyzer must explicitly distinguish this from COM. These findings do not establish any physical drift cause.

Next: obtain independent focused review of committed pilot changes; close analyzer semantics and CLI compatibility; verify raw historical evidence and current full asset; then exact-candidate live pre-review/Commander release if required. Enhanced zero-target and command-passthrough are separate conditions.

## Added CPU evidence and independent findings

- `pilot-cpu-v3/`: 7 real-tensor tests pass, including default HistoryActor and PrivilegedCriticH1, sixteen 32-lane distribution captures bound to 512 rows, clone isolation and wrong input/order/concentration rejection at unchanged tolerances. No optimizer body is executed.
- `pilot-cpu-v4/`: 8 real-tensor cases plus 10 existing cases pass. Includes manual callback lifecycle of eight boundaries; simulated callbacks are not actual optimizer updates.
- `reviews/pilot-review-e6f1ff6.json`: independent MUST_FIX on missing real network collection coverage and attempted/completed step semantics. Implementation closure is awaiting next exact-candidate review; no self-approval.
- `analyzer-cpu-v1/`: 6 boundary/COM/causal-clue unit fixtures and 29 existing enhanced protocol tests pass. Fixtures using a mock validator are explicitly not full live coverage. Analyzer schema v2 reports separate segments and never promotes heuristic scores to a root cause.
- `cli-doctor-v1.json`: installed 0.161.0 candidate passes installed registry census (local rejecting endpoint, zero model calls), Landlock/seccomp, role write scopes, protected paths and GPU execution denial. This is a new project compatibility check, not historical release certification. `cli-isolation-cpu-v1.*`: 18 tests pass, including both previously failing installation checks. Invocation used existing control Python, PYTHONPATH=.:tools/rlflow:tools/rlflow/tests, pytest test_cli_compatibility.py and test_isolation.py with CUDA invisible.
- Historical time-series raw path exists and SHA-256 matches `d775d2f073f829c1ecb3bcd73d24a1b7626c90ede2bf629e19cd9ccddb75de28`. Its source points to the R4-B0-R1 control trace, not all 164,800 transitions in its parent collection job; only 20,160 B0 transitions are relevant. Chunk verification is pending.

Product Python lacks a directly importable pxr module; direct CPU asset-stage inspection has not succeeded. This does not authorize dependency installation or replacing mass from a historical decimal. The current articulation config preserves USD masses without a mass override. Full composed asset and actual plant mass remain to bind before a physical correction.

## Actual asset and historical raw verification

`historical-b0-verified-v1.json`: all 630 R4-B0 trace chunk size/SHA-256 checks pass; 20,160 historical transitions; steady window values recomputed from CPU-loaded raw tensors. This confirms provenance, not enhanced-equivalent mechanics or repaired-control acceptance. New verification script preserves source evidence and writes only the new output.

`current-usd-mass-cpu-v2.json`: composed current USD read using existing USD libraries in a CPU process, no SimulationApp or PhysX. Base mass is 1.1583074331 kg; three rotors have authored mass 0.0032754999 kg, rotor_3 has zero/unresolved authored mass and needs runtime collider-derived mass. Total runtime mass is intentionally UNKNOWN. The source asset SHA-256 is 3e733ca1008b1a3620941a8c1cc2e82bd7006a6a4c441c921a292bf11d2aee99. This is why summing authored numbers or replacing base mass by the historical total would be incorrect. Initial library import failures were resolved by per-process Python/library paths, without installing dependencies or editing runtime.

The frozen installed Isaac Lab articulation_data.py documents body_state_w as link pose plus COM velocities; body_link_state_w provides link pose and link velocities. The observer formerly added omega cross local COM to body_state_w, double-counting the offset. Minimal diagnostic repair selects body_link_state_w, and root_link_state_w for same-point root response, while controller velocity-error evidence retains existing root_state_w feedback. No controller, actor, reward or plant change. `product-mechanics-cpu-v1/`: one executable mocked-articulation real-tensor snapshot regression and 20 existing source/mock wiring cases pass. This is not actual live integration. A spinning body with offset COM demonstrates the defect and repair.

Reviewer call 2 failed with model-capacity error; no verdict or release. Counted in new goal budget. Next call is reserved for the exact committed delta, with 6 critical calls still reserved.

## Independent numeric finding closure candidate and live-input compatibility

`reviews/review-5da1439.json` closes PILOT-R1/R2 independently, but raises MECHANICS-R1. The new candidate retains the 1e-8 consistency bound and derives system/rotor quantities from the same serialized primitives; no precision threshold was widened. `product-mechanics-cpu-v2/` adds 0/100/700m origins, rotated nonzero COM/omega, next-physics momentum difference, and corrupt COM/momentum/wrench rejection. Current plant mass and inertia now use public root_physx_view.get_masses/get_inertias rather than initialization defaults; `measurement-source-cpu-v1/` verifies that poisoned defaults cannot replace actual measurements. These are executable mock integrations, still not real simulation.

Handoff v2 corrects metadata to actual dt=.005, decimation=20, 200Hz control/10Hz policy. Enhanced zero-target binds no policy checkpoint; schema rejects checkpoint use. New experiment ID is UNIFIED-H1-R5-READINESS-B0-BASELINE-20261012, distinct from failed historical tasks. New per-source hashes and formal validate-only CPU result are recorded; candidate remains not live authorized.

The rlflow dispatcher now resolves to the consolidated project implementation; immutable snapshots may use exactly identical reviewed dispatcher bytes at their own path, with complete H1 candidate/pin checks retained. Historical postrun identities remain explicitly distinct; a new caller must verify the new experiment/package identity instead of inheriting old scope. `live-tool-binding-cpu-v1` initially exposed an outdated app.close test stub; v2 fixes the stub to the already used public close kwargs, retaining actual cleanup expectations. v2: 37 tests pass. This failure and its outputs are preserved.

Committed baseline_snapshot_config.json is a new path binding for a future read-only snapshot, not an accepted historical candidate. No snapshot/live release has yet been issued.

## Snapshot preparation boundary

The first read-only snapshot at 81bc968 was prepared without authorization or live execution. Its exact handoff validates, but full config validation fails because proposal_path points to the development tree rather than the snapshot-local approved method. All v1 inputs and outputs are preserved as PRELAUNCH_REJECTED. A v2 input corrects this path with the same approved method bytes/hash. The runner also explicitly binds ISAACSIM_ASSET_ROOT from the exact config instead of relying on the supervisor environment. CPU tool regression v3 passes 38 cases. No live budget was consumed.

The corrected v2 read-only snapshot is main@918fec6f8f37a3700d47cf615168eb9cad2603b1, with only the committed baseline_v2_snapshot_config.json overlay, diff SHA e610de483653a7332a5c160bae7184b25963cf51cab3597317a96c85ffbe0c35. Both formal validate-only and handoff checks PASS. Input manifest specifies per-run initialization/reset/probe/output and remaining-envelope bounds. Expected authorization is a proposed byte file under docs; actual attempt commands/AUTHORIZATION.json is absent. Independent pre-review is pending; no live release is implied.
