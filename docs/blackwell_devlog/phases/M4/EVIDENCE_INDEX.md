# M4 Evidence Index

Raw evidence root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M4-001/M4-001-D0001/`

## Development evidence

| Subrun | Contents | Result |
| --- | --- | --- |
| `0002` | initial 16-test suite | FAIL preserved |
| `0003` | repaired implementation/source suite | FAIL preserved |
| `0004` | 16-test suite | PASS |
| `0005` | six-file syntax compilation | PASS |
| `0006`, `0007` | independent pure-CUDA smokes and complete `.pt` outputs | PASS |
| `0008` | comparator with unavailable system Torch | FAIL preserved |
| `0009` | frozen-Conda complete-output comparator | PASS |
| `0010` | complete M3 static suite twice | PASS |
| `0011` | 16-test M4 suite twice | PASS |
| `0012` | expanded 18-test M4 suite | PASS |
| `0013` | first ordered commit | PASS |
| `0014` | final expanded 18-test M4 suite twice | PASS |

Each subrun retains command, working directory, environment description,
start/end time, exit code, and raw stdout/stderr.

## Determinism artifacts

- CUDA summaries are byte-identical at SHA256
  `c732b69d06ef8ac2bbff42d8a5e5147b10f33a206e52e79c5c662101c211dc65`.
- Full comparison JSON SHA256 is
  `11be50e3ccce917b28edc38e70e413441a65ef0be9e82243a920453eadbbc762`.
- The comparator loads all six outputs from both processes and compares every
  `lidar_grid` and `state_vec` element: 3,059,184 elements, exact equality,
  maximum absolute error `0.0`.

## Final evidence

The immutable release, complete command ledger, self-audit, checksums, Git
proofs, frozen-integrity records, and final repeated suites are under
`release/v0001` and `final_candidates/0001` in the raw evidence root.
