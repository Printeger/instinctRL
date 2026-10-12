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
