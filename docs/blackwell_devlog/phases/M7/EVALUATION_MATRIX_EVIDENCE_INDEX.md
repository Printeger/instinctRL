# M7 Evaluation Availability Matrix Evidence Index

## Attempt root

Fresh raw streams and metadata are stored under:

`instinctRL_coordination/attempts/development/M7-021/M7-021-D0001/`

Each `commands/<label>/metadata.json` binds exact argv, working directory,
selected environment, start/end times, exit code and SHA-256 identities of
stdout and stderr. `COMMAND_INDEX.json` is the final machine-readable index.

## Principal evidence

| Labels | Contents |
| --- | --- |
| `001`, `005` | focused matrix development checks |
| `002` | canonical matrix identity, counts and complete 90-cell blocker map |
| `003`-`010` | functional diff, staging and commit evidence, including disclosed identity-only retry |
| `011`, `012` | focused matrix/baseline/scenario qualification twice |
| `013`, `014` | all accepted M7 pure suites twice |
| `015`, `016` | relevant M3-M6 regression twice |
| `017` | M2 static qualification |
| later indexed labels | commits, diff hygiene, identities, immutability and clean-state gates |

## Canonical artifact

The probe output binds base product commit
`39206507367e0fdd65ed99b6f3113dd5901d43bc`, all 90 status/blocker cells and
matrix SHA-256
`a0d1c1cdb8cedf18705a7ab4144a682b75ba259ef5e83809d85947546f540b65`.
It records 7 `READY`, 74 `NOT_READY`, 9 `SKIPPED` and false execution for every
cell.
