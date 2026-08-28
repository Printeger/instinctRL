# Exact Observability Evidence Index

## Stable product evidence

| Evidence | Stable location |
| --- | --- |
| Pure implementation | `isaaclab-training/instinctlab/evaluation/observability.py` |
| Synthetic, negative, ownership, and actor tests | `isaaclab-training/tests/test_m7_observability_contract.py` |
| Public evaluation-only export | `isaaclab-training/instinctlab/evaluation/__init__.py` |
| Boundary summary | `isaaclab-training/README.md` |
| Mathematical/API contract | `docs/blackwell_devlog/phases/M7/OBSERVABILITY_CONTRACT.md` |
| Test outcomes | `docs/blackwell_devlog/phases/M7/OBSERVABILITY_TEST_REPORT.md` |
| Limitations and closeout | adjacent observability limitation/final records |

## Reproduction commands

Use the frozen `navrl_blackwell_test` Python with
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=isaaclab-training`, and
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

```text
/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python -m pytest -q isaaclab-training/tests/test_m7_observability_contract.py
```

The full pure replay appends the accepted M7 evaluation/runtime contract tests,
M3 static, M4 observation, M5 control/runtime-control, and M6 task/environment/
PPO/rollout contract tests. It is executed twice. M2 static is executed
separately. `git diff --check` and base-to-HEAD name-status/stat/full diff
qualify scope and hygiene.

Raw streams, exact argv/cwd/environment/times/exits, validator source hashes,
semantic outputs, package/frozen/formal/external integrity, and the formal
attempt identity are retained only in the separate coordination Evidence Index
and Executor report. This product record intentionally contains no mutable
coordination candidate path or candidate hash.
