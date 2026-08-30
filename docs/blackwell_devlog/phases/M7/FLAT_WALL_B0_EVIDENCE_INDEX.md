# M7-015 Evidence Index

Fresh evidence root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-015/M7-015-D0001/`

The canonical command ledger is `DEVELOPMENT_COMMAND_LEDGER.jsonl`; every row
records argv, cwd, selected environment, UTC start/end, expected/actual exit,
stdout/stderr path and hashes. Raw logs are under `raw_logs/`; JSON/Torch
bundles, comparison results, summary and requirement map are under `formal/`.

| Evidence | Ledger ordinals / artifact |
| --- | --- |
| Preflight/order/locked hashes | 1–5 |
| Rejected development candidates and passing repair | 6–8 |
| Syntax, focused suites twice, diff check | 9–12 |
| Functional commit | 13 |
| Flat-wall processes | 14–15; `formal/flat_wall_{1,2}.{json,pt}` |
| Open-field v2 processes | 16–17; `formal/open_field_v2_{1,2}.{json,pt}` |
| Complete comparators | 18–19; `formal/*_comparison.json` |
| M2/M7/M3–M6 suites | 20–24 |
| Learned/M7-011/M7-012 CUDA regressions | 25–27 |
| Package, driver and frozen identities | 28–39, 40 |
| Functional fail-closed audit | 41; `formal/SUMMARY.json`, `formal/REQUIREMENT_MAP.json` |

Important identities:

```text
functional commit: fdc81997ac5055a626571f244d68890cf1a2a425
catalog: 581e894f108315306a7ed818323de3eb30fbb9d6b53a4555d229056eb492036e
flat fixture: ed170f337abf9cd2aff288b54473b9404824be7885b4c3fa61e75484de88bdcf
flat qualification: 2592b2e6dddd0a27fccc126ab98acee644eb3d22fa1f79d6eac4004579cb8497
open fixture: 04c743b6c3ef31ff74b5ca9e5d4c18a808f5a171fb832e3932aa12c82950d976
open qualification: 15368f7e8feabb61bf39ac1d4251cb97a3ecc65ff47cce1271e3c812b68474e2
known IsaacLab kit override: 0a0dee5fbbdcbf66f8124a07eced9a87c2f70e5df6ee892ec297444c1d1b01c2
```

M7-014 D0003 evidence had no file newer than its Commander approval timestamp;
the base-to-functional diff also excludes all prior M7-014 records/runners.

