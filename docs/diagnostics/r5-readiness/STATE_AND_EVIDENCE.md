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
