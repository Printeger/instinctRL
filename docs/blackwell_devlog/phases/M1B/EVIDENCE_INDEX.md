# M1B Evidence Index

## Attempt history

- `M1B-001-D0001` is immutable and failed its revision-0 unqualified
  base-to-HEAD `git diff --check` gate. Its raw command and diagnostics remain
  unchanged under
  `/home/mint/rl_dev/instinctRL_coordination/attempts/development/M1B-001/M1B-001-D0001/`.
- `M1B-001-D0002` applies Order revision 1's composite hygiene gate. Its raw
  evidence is under
  `/home/mint/rl_dev/instinctRL_coordination/attempts/development/M1B-001/M1B-001-D0002/`.

| D0002 evidence | Raw location |
| --- | --- |
| Inline plan and terminal attempt state | `ATTEMPT.json` |
| Exact command records and stdout/stderr hashes | `COMMANDS.json`, `commands/` |
| Source/target object ledger | `OBJECT_LEDGER.json` |
| Final static suite runs | `commands/final_static_run_1.*`, `commands/final_static_run_2.*` |
| Final normalized USD inspections | `usd_inspection_final_1.json`, `usd_inspection_final_2.json` |
| Composite hygiene gate | `git/canonical_diff_check.*`, `git/artifact_excluded_diff_check.*`, `COMPOSITE_GATE.json` |
| Required-test/correctness-gate map | `INDEX.md` |
| Base-to-HEAD and per-commit Git evidence | `git/` |
| Frozen package/repository integrity | `integrity/` |
| Executor delivery | `/home/mint/rl_dev/instinctRL_coordination/reports/M1B-001.md` |
