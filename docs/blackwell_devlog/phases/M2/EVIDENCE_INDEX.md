# M2 Evidence Index

Raw evidence is stored in the separate coordination repository at
`attempts/development/M2-001/M2-001-D0001/`.

| Evidence | Location | SHA-256 / status |
| --- | --- | --- |
| Static/syntax subrun | `dev_runs/0001/` | PASS |
| First independent smoke JSON | `dev_runs/0002/smoke.json` | `8ac4bd9587b5ec08e7e614dc7c1d07d9199eb43d4cb592e214600c02749f29c4` |
| Second independent smoke JSON | `dev_runs/0003/smoke.json` | same exact hash |
| Cross-process comparison | `dev_runs/0003/comparison.json` | `dc24c32222ea1d143a5d9b5e5ba8b0da380cfb0aaf1dd7cdffb78786a6011eca` |
| First commit attempt | `dev_runs/0004/` | expected identity failure; no config changed |
| Adapter commit | `dev_runs/0005/` | `aa5f26ad423297b09f40f28c3b7f6631d540b4cf` |

The final D0001 coordination index maps the frozen evidence-runner ledger to
governance, commits, diffs, static repeats, both runtime processes, comparator,
integrity, prohibited-work audit, and final clean status.
