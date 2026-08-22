# M7-001 Evaluation Test Report

## Revision 1 correction

M7-001-D0002 `formal_candidates/0001` is **PASS**: 44/44 commands and
7/7 summary assertions. It rejects the exact Commander cross-branch alias plus
state offset/strided and LiDAR-originating views; 38 M7 tests pass. The accepted
`HistoryActor` test restores identical RNG and compares exact alpha, beta,
sampled action, and log probability under materially changed evaluation data,
then proves an actor-visible policy perturbation changes the distribution. The
D0001 result below remains historical and is not rewritten.

| D0002 check | Result | Measured outcome |
| --- | --- | --- |
| Commander alias reproduction | PASS | `REJECTED_CROSS_BRANCH_ALIAS` |
| M7 focused/retained suite | PASS | 38 passed |
| Accepted M3-M6 pure suites, round 1 | PASS | 204 passed, 33 subtests passed |
| Accepted M3-M6 pure suites, round 2 | PASS | 204 passed, 33 subtests passed |
| M2 static | PASS | 10 passed |
| D0001 evidence integrity | PASS | 315 files byte-identical |
| Two independent records | PASS | D0001-identical SHA256 |
| Complete comparison | PASS | 78 numeric leaves, maximum error 0.0 |
| Five record mutations | PASS | five expected exit 1 rejections |
| Six-path/unaffected/frozen/package checks | PASS | all expected exits |

All D0002 formal executable validation used persisted file paths. No stdin,
heredoc, here-string, process substitution, shell/interpreter `-c`, `python3 -`,
or `python -` command was used.

## D0001 formal result (historical)

Candidate `formal_candidates/0002`: **PASS**, 38/38 commands and 7/7 summary
assertions. All formal executable validation used persisted program paths; no
stdin, heredoc, here-string, process substitution, shell/interpreter `-c`, or
`python -` command was used.

| Check | Result | Measured outcome |
| --- | --- | --- |
| M7 hand/negative/isolation/ownership/JSON suite | PASS | 34 passed |
| Accepted M3-M6 pure suites, round 1 | PASS | 204 passed, 33 subtests passed |
| Accepted M3-M6 pure suites, round 2 | PASS | 204 passed, 33 subtests passed |
| M2 static | PASS | 10 passed |
| Source and reverse-import audit | PASS | 7/7 assertions |
| Two independent record processes | PASS | byte-identical SHA256 |
| Complete ordered comparator | PASS | 78 numeric leaves, maximum error 0.0 |
| Missing/extra/reordered/nonfinite/any-error mutations | PASS | five expected exit 1 rejections |
| Accepted paths and frozen repositories | PASS | unchanged/clean, exact frozen heads |
| Implementation allowlist/history/diff checks | PASS | exact four paths/one first commit |

## Hand-computed table

The fixed `T=2,N=2` table has tracking squared errors `[1,1,0,4]`, producing
RMSE `sqrt(1.5)`. Its null drift values are `1,3` (mean 2), null actual speeds
`1,0` (mean 0.5), valid anchor errors `0.1,0.3` (mean 0.2), clearance minimum
0.5 from three valid samples, collision rate 1/4, ICS violation rate 2/4,
preservation ratios 0.8 and 1.1 (mean 0.95), intervention 1/4, and valid
observability values 2,4,6 (mean 4).

Termination includes two timeout samples simultaneous with termination; cause
counts remain collision 1, below 1, above 1, timeout 2, while precedence reason
counts remain none 1, collision 1, below 1, above 1, timeout-only 0.

## Retained development results

- Default `/home/mint/miniconda3/bin/python3` lacked pytest: NOT RUN at
  collection, exit 1; no package was changed.
- First frozen-environment attempt auto-loaded an incompatible ROS pytest
  plugin and failed before collection: NOT RUN, exit 1. Formal execution fixes
  the environment with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- First collected M7 run: FAIL, 32 passed/2 failed; both fixture defects were
  corrected, followed by 34 passed.
- Formal candidate `0001`: FAIL only because the self-audit expected 109 rather
  than the measured 78 numeric leaves. Its 38 command exits, byte equality,
  zero maximum error, regressions, and negatives otherwise passed. Candidate
  `0002` retained a new namespace and reran the complete ladder.

No failed or unrun check is represented as PASS.
