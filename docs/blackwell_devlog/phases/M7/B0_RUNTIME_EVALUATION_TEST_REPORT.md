# M7-012 Exact B0 Runtime Evaluation Test Report

## Result

`PASS` — accepted formal candidate `M7-012-D0001/formal_candidates/0001`.
The candidate ran from clean functional commit
`9eff51c789abe25717877d70ea3f5c80b62678ba` and recorded 37 commands with
exact argv, environment, UTC times, exits and raw stdout/stderr hashes.

## Pure and static checks

| Check | Runs | Result |
| --- | ---: | --- |
| Focused B0 runtime/environment/lifecycle/execution/isolation suite | 2 × 195 | PASS |
| Accepted M7 pure suites, including M7-012 | 2 × 498 | PASS |
| Relevant M3-M6 behavior regressions | 2 × 203 | PASS |
| M2 static suite | 10 | PASS |
| Source/API/import/locked-hash/allowlist audit | 1 | PASS |
| Factory state, privacy and retention audit | 1 | PASS |
| Independent atomic negative matrix | 14 cases | PASS |

The M3-M6 command used only the standing Order-authorized deselection
`test_m6_environment_contract.py::test_accepted_files_are_unmodified_from_order_base`;
it was deselected once in each of the two runs. There were no skipped tests.
Across the recorded pytest invocations, 1,802 tests passed and two tests were
deselected.

## CUDA checks

| Check | Exact coverage | Result |
| --- | --- | --- |
| Accepted learned M7 runtime regression | 2 env / 8 policy / 160 physics / 339,840 rays | PASS |
| Accepted M7-011 B0 environment regression | 2 env / 8 policy / 160 physics / 320 environment samples | PASS |
| New B0 runtime evaluation process 1 | 8 rows / 16 metric samples / 160 lifecycle+wrench / 339,840 rays | PASS |
| New B0 runtime evaluation process 2 | 8 rows / 16 metric samples / 160 lifecycle+wrench / 339,840 rays | PASS |
| Complete process comparator | 41 tensors / 3,360 elements / zero difference | PASS |
| Comparator mutation matrix | missing, extra, reordered, tensor-bit, metric-count, receipt, record, private-field | 8/8 rejected |

Each new process recorded zero actor, training and reset calls, exactly eight
uses of the collector-staged command as the environment input, one matched
environment receipt and one canonical record. Every independent metric oracle
matched the unchanged accepted formulas exactly.

## Metric wiring values

These values are non-blocking wiring/aggregation evidence, not performance
thresholds or comparisons.

| Metric | Available | Value | Numerator | Denominator | Valid / total |
| --- | --- | ---: | ---: | ---: | ---: |
| tracking RMSE (m/s) | yes | 2.530957146175167 | 102.49190521240234 | 16 | 16 / 16 |
| null station drift mean (m) | yes | 0.08049054940541585 | 0.4829432964324951 | 6 | 6 / 16 |
| null station speed mean (m/s) | yes | 0.27453966935475665 | 1.64723801612854 | 6 | 6 / 16 |
| anchor error mean (m) | no | unavailable | unavailable | 0 | 0 / 16 |
| minimum clearance (m) | yes | 2.714244842529297 | 2.714244842529297 | 1 | 16 / 16 |
| collision rate | yes | 0.0 | 0.0 | 16 | 16 / 16 |
| ICS violation rate | yes | 0.0 | 0.0 | 16 | 16 / 16 |
| command preservation ratio | yes | 0.7600000381469727 | 7.600000381469727 | 10 | 10 / 16 |
| intervention frequency | yes | 0.0 | 0.0 | 16 | 16 / 16 |
| observability mean | yes | 0.0 | 0.0 | 16 | 16 / 16 |

## Integrity

`git diff --check`, clean start/end, exact source hashes, functional history,
12-path functional allowlist and full base-to-functional diff passed. Frozen
formal source, Isaac Lab, OmniDrones and NavRL identities/statuses matched the
accepted baselines. Python remained 3.11.15, PyTorch remained 2.7.0+cu128 and
the device was NVIDIA GeForce RTX 5070 with driver 580.173.02 observed by the
headless processes.
