# M7-015-D0002 Evidence Index

Fresh evidence root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-015/M7-015-D0002/`

The canonical command ledger is `DEVELOPMENT_COMMAND_LEDGER.jsonl`; every row
records argv, cwd, selected environment, UTC start/end, expected/actual exit,
stdout/stderr path and hashes. Raw logs are under `raw_logs/`; JSON/Torch
bundles, comparisons, summary and requirement map are under `formal/`.

| Evidence | Ledger ordinals / artifact |
| --- | --- |
| Preflight, Order/review/probe hashes, locks, D0001 snapshot | 1–6 |
| Syntax, exactness probe, focused suites twice, functional diff | 7–12 |
| Exact functional staging and commit | 13–15 |
| Four fresh scenario processes | 16–19; `formal/{flat_wall,open_field_v2}_{1,2}.{json,pt}` |
| Unchanged complete comparators | 20–21; `formal/*_comparison.json` |
| M2/M7/M3–M6 suites | 22–26 |
| Learned/M7-011/M7-012 CUDA regressions | 27–29 |
| Python/package/driver/locked/frozen identities | 30–43 |
| Disclosed audit diagnostics and passing fail-closed audit | 44–49; `formal/SUMMARY.json`, `formal/REQUIREMENT_MAP.json` |

Important D0002 identities:

```text
functional commit: cbd51657bdecfb76408bc5f4531f77635449b7e8
fixture id: m7-015-flat-wall-b0-two-env-eight-step-v1
flat source: 900a9147cd74cd1750d831afbba5319acef62c30f748e9599053266d83c87e6e
catalog: 0ff2a806ffa5452c784b60dd7a65d7f9ba4b23a2d669af8f990274cc37612769
flat plan: 19017a0c378cf43af8a7b239c1f9c7290809d4d434505ca31ffc92c664bb8218
flat manifest: 5a48153a4621e3ddeb67189e12b588ac68c5d8b5913de179c7ce4d5bd28e83c3
flat receipt: 3f3176f4f6a91a8c7dd0a66bb6ba7a2694bbc6bf3892f222e337cac2888eed0f
flat record: 2948ddc64371832eb472492c323c080ae38d5f7e11b7554df441fa34c2413f9c
flat qualification: 8de863aaeb2b369fe9c0038fc33f20bca234851b600ef173143c708653f3163c
open source: 04c743b6c3ef31ff74b5ca9e5d4c18a808f5a171fb832e3932aa12c82950d976
open plan: 344e1311f7961c52f5b02e1048d4073d4f3429cfff1e09411f04b29fb308aec0
open manifest: 10b63205f9ed7d78dc053d5224789194466c146a67396b5f0ca38ee6efaac67c
open receipt: 4c1ea869f74e729a42c4ea7cf1db9bbaf09f4688af2f8f43490a0073852e0f77
open record: 13d2802319d10ab3fe2b86d9a3a0caf5f41008b7efddd105975d014b3142db48
open qualification: afa353106b02d9a2e243383d343f9a6477540cce17c1f58eaaa0cf9693e0858f
known IsaacLab kit override: 0a0dee5fbbdcbf66f8124a07eced9a87c2f70e5df6ee892ec297444c1d1b01c2
```

The ordinal-6 full-file SHA-256 snapshot is revalidated by the passing audit,
proving the complete D0001 evidence namespace remained immutable.
