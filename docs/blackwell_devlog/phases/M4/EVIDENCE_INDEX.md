# M4 Evidence Index

Raw evidence root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M4-001/M4-001-D0002/`

## Development evidence

| Subrun | Contents | Result |
| --- | --- | --- |
| `0002` | focused revision-1 suite, 22 tests | PASS |
| `0003`, `0004` | independent coherent pure-CUDA smokes and complete `.pt` outputs | PASS |
| `0005` | unchanged full-output comparator | PASS |
| `0006` | syntax and both diff-check surfaces | PASS |
| `0007` | final expanded M4 suite twice | PASS |
| `0008` | complete M3 static suite twice | PASS |

Each subrun retains command, working directory, environment description,
start/end time, exit code, and raw stdout/stderr.

## Determinism artifacts

- CUDA summaries are byte-identical at SHA256
  `a2b578971da6a85a38d586a18f371312fda95c5d4c331c7f02bd95d70541ae8e`.
- Full comparison JSON SHA256 is
  `f9f7549a1ac67390fb364d4e7fc57d92eef35177780a3333354d89a1ace05afd`.
- The comparator loads all six outputs from both processes and compares every
  `lidar_grid` and `state_vec` element: 3,059,184 elements, exact equality,
  maximum absolute error `0.0`.

## Final evidence

The immutable release, complete command ledger, self-audit, checksums, Git
proofs, frozen-integrity records, and final repeated suites are under
`release/v0001` and `final_candidates/0001` in the raw evidence root.

D0001, its release, its failed final candidate, and the first two product
commits remain immutable evidence.
