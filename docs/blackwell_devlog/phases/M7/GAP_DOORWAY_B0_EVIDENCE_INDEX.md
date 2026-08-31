# M7 Gap/Doorway B0 Evidence Index

Raw evidence root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-020/M7-020-D0001`

Each command directory contains exact argv, working directory, selected
environment, start/end timestamps, exit code and SHA-256-bound stdout/stderr.
The coordination `COMMAND_INDEX.json` and `EVIDENCE_INDEX.md` are generated
from those retained files.

## Primary artifacts

- `formal/doorway_1.json`, `formal/doorway_1.pt`
- `formal/doorway_2.json`, `formal/doorway_2.pt`
- `formal/doorway_compare.json`
- Two JSON/PT bundles and one comparison JSON for each of `unsafe`, `clutter`,
  `corridor`, `corner`, `flat` and `open`
- `formal/learned_runtime.pt`, `formal/b0_environment.json`,
  `formal/b0_collector.json`

The doorway summaries bind catalog, manifest, run plan, fixture source,
evaluation record, qualification and bundle hashes. They also retain complete
geometry inventory, hit-count matrices, body/world commands, quaternion norms,
intersection parameters/coordinates, four margin matrices and all gates.

Functional commit: `62494f1894a83e7c57fc0fcc62ada4a07160ca66`.
