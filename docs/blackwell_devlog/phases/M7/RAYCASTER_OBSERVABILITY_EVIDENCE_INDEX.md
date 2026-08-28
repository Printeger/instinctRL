# RayCaster Exact-Normal Evidence Index

## Stable product evidence

| Evidence | Location |
| --- | --- |
| Adapter | `isaaclab-training/instinctlab/evaluation/raycaster_observability.py` |
| Focused contract | `isaaclab-training/tests/test_m7_raycaster_observability_contract.py` |
| Live runner | `isaaclab-training/tests/run_m7_raycaster_observability_smoke.py` |
| Complete comparator | `isaaclab-training/tests/compare_m7_raycaster_observability_smoke.py` |
| Boundary summary | `isaaclab-training/README.md` |
| Contract/results/limitations | adjacent `RAYCASTER_OBSERVABILITY_*` records |

## Frozen sources

| Frozen Isaac Lab 2.3.2 path | SHA256 |
| --- | --- |
| `source/isaaclab/isaaclab/utils/warp/ops.py` | `6202538ae05e3f424a0dfdd4df7e00cc272e4213827cceafa5a0a72df8baea84` |
| `source/isaaclab/isaaclab/utils/warp/kernels.py` | `b38780baabb26b035595e6b6710e26381a18e99ed3bfbbc82887321aee7f82bf` |
| `source/isaaclab/isaaclab/sensors/ray_caster/ray_caster.py` | `b6a84cb0e506f4a4724d3262e0f533ccb31d75128e1e46282220fcfa688e286c` |

The frozen repository remains at its qualified commit and known kit-only
status. The adapter copies no frozen implementation.

## Stable reproduction

Use the frozen `navrl_blackwell_test` environment. Pure tests set
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=isaaclab-training`, and
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Live runs use `conda run`, the same product
path, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, then invoke the tracked runner with
a full product commit and output path. Invoke the tracked comparator on two
independent outputs.

The full pure suite comprises the focused adapter, all accepted M7 pure
contracts, and relevant M3-M6 pure contracts; it runs twice. M2 static runs
separately.

Exact argv/cwd/environment/times, program/source hashes, raw stdout/stderr/
exits, complete output trees, mutation artifacts, package/formal/frozen/
external integrity, and formal attempt identity belong only to the coordination
Evidence Index and Executor report. No mutable coordination candidate path or
candidate hash is embedded here.
