# M1B Test Report

## Results

| Check | Result | Evidence |
| --- | --- | --- |
| Governance/base/source/authorization preflight | PASS | `M1B-001-D0001/INDEX.md` |
| Three source-object/target byte identities | PASS | static runs and object ledger |
| Provenance and notice completeness | PASS | static runs |
| Vehicle YAML active values | PASS | static runs |
| Lee YAML active values | PASS | static runs |
| Standard-library config parity and units | PASS | static runs |
| New-tree path independence | PASS | static runs and USD dependency inventory |
| Forbidden import/runtime/actor surfaces | PASS | static runs and negative-search evidence |
| Read-only source/target USD inventory | PASS | `usd_inspection_run_1.json`, `usd_inspection_run_2.json` |
| Four-rotor ordered topology mapping | PASS | normalized USD mapping rows |
| Static suite repeatability | PASS | two runs, ten tests and exit 0 each |
| USD semantic repeatability | PASS | two byte-identical normalized JSON files |
| Git/diff/two-commit/final-clean gates | PASS | final exact-HEAD evidence ledger |
| Frozen runtime/repository integrity | PASS | before/after integrity ledger |

No test was represented as PASS without execution. The USD inspector opens
only existing stages and performs schema queries. It flushes complete results
and uses the Order-permitted `os._exit` handling for the baseline-known
headless close behavior; it does not create physics or simulation state.

## Rotor mapping method

The four candidates are named `/taslab_uav/rotor_0` through `rotor_3` and each
has a revolute joint to `base_link`. Their world XY positions include a common
asset-origin offset. Subtracting the four-candidate centroid produces four
unique nearest matches to the ordered YAML
`arm_length * (cos(angle), sin(angle))` locations. Each nearest index also
equals the `rotor_N` index. Errors are approximately `1.85e-6 m`; the recorded
`5e-6 m` tolerance covers the YAML angle precision and USD floating-point
storage while remaining far below the separation between rotor locations.

