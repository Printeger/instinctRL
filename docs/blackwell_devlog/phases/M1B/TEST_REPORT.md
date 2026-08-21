# M1B Test Report

## Attempt history

| Attempt | Check | Result | Evidence |
| --- | --- | --- | --- |
| D0001 | Unqualified base-to-HEAD `git diff --check` | FAIL, exit 2 | Immutable `M1B-001-D0001/git/diff_check.*` |
| D0002 | Unfiltered canonical diagnostics restricted to eight copied-YAML lines | PASS | `M1B-001-D0002/git/canonical_diff_check.*` |
| D0002 | Base-to-HEAD check excluding only three copied artifacts | PASS, exit 0 | `M1B-001-D0002/git/artifact_excluded_diff_check.*` |

## D0002 results

| Check | Result | Evidence |
| --- | --- | --- |
| Revision-1/start-HEAD/source/authorization preflight | PASS | D0002 index and object ledger |
| Three source-object/target byte identities | PASS | final static runs and object ledger |
| Provenance/notice and exact YAML/config/gain values | PASS | final static runs |
| New-tree independence and forbidden actor/runtime surfaces | PASS | final static runs and USD dependency inventory |
| Read-only source/target USD inventory twice | PASS | `usd_inspection_final_1.json`, `usd_inspection_final_2.json` |
| Four-rotor ordered topology mapping | PASS | normalized USD mapping rows |
| Static suite repeatability | PASS | two runs, ten tests and exit 0 each |
| USD semantic repeatability | PASS | two byte-identical normalized JSON files |
| Additive three-commit chain, allowed paths, final clean state | PASS | final exact-HEAD Git evidence |
| Frozen runtime/repository and D0001 integrity | PASS | integrity ledger |

No test is represented as PASS without execution. The USD inspector opens only
existing stages and performs schema queries. It flushes complete results and
uses the Order-permitted `os._exit` handling for the baseline-known headless
close behavior; it does not create physics or simulation state.

## Rotor mapping method

The four candidates are named `/taslab_uav/rotor_0` through `rotor_3` and each
has a revolute joint to `base_link`. Their world XY positions include a common
asset-origin offset. Subtracting the four-candidate centroid produces four
unique nearest matches to the ordered YAML
`arm_length * (cos(angle), sin(angle))` locations. Each nearest index also
equals the `rotor_N` index. Errors are approximately `1.85e-6 m`; the recorded
`5e-6 m` tolerance covers YAML angle precision and USD floating-point storage.
