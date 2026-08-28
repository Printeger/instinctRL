# M7-007 B0 Direct Adapter Evidence Index

Accepted evidence root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-007/M7-007-D0001/formal_candidates/0003/`

The 40-row command ledger records exact argv, cwd, isolated environment, UTC
timestamps, expected/actual exits, stdout/stderr paths, and stream hashes.

| Artifact | SHA-256 |
| --- | --- |
| `SUMMARY.json` | `1cdeee0d3dcfb1352781e0ef657de09f10d56cc713f07bfbf5553c45a6ab01df` |
| `COMMAND_LEDGER.json` | `c52f3b6f160038579ac8308803fa6b7291314f6a190475f7b871eed07622e208` |
| `fixtures/FIXTURE_INDEX.json` | `bd88f438f434041ce19f9cf524029500852130e33554a5f28e8ea94b2e5e6dd5` |
| `negative_matrix/INDEX.json` | `97072ee38d8a4443f3c6f2e4662f9a5797df9a1f50f163264394fd68ddd6d87b` |
| `process_records/process_1.json` | `0ee1f23c46727779cc536a709ab6d1b2bbbf2a2721faffe98172403a734c5ddf` |
| `process_records/process_2.json` | `0ee1f23c46727779cc536a709ab6d1b2bbbf2a2721faffe98172403a734c5ddf` |

The process record binds exact float64 tensor bytes:

```text
body bytes:  8c1ae125eed69834723eb6f0564f652d48ba63c5d3bcfd86f0001060ac2192e2
world bytes: bbf5a3792f182ce24398c87122269f55a7f1f4786304a763b6ee0dd3528104a6
```

## Successor manifests

All use product commit `bf92c054378954cb14635c3cec2af93c3e62364c`,
scenario `S1.corner`, and seed 7.

| Baseline/status | Manifest SHA-256 | File SHA-256 |
| --- | --- | --- |
| B0 READY | `7db8594b5a9657371ae7cd44dbc5d3d02b26122ae81706bc5de4dac94c5c988a` | `e7eadd4874045f9fe7caa7f8a276682e7bada363c463ce3b69ca357dc4241123` |
| B0 EXECUTED | `8d7f50f8dfeebdd62be5c25a0a32d62599f201973fc92ad5dca6c96dcc627a67` | `363aea4be82b250a09c26551d588737c7c98cbffbfff5ca299d2f8d740b26e03` |
| B1 NOT_READY | `cae061f68584a41eb4c09b428572917d8eacf3470ac18ec0938a0c0227b1f375` | `d54d1338460dea8fe21488f1cdd1ec232fe82b2500c161de61bf92466f5dfc7f` |
| B8 SKIPPED | `74367f30885b9773422714257b1805f6b2174e08425101539711287a80af6a9a` | `b10085a9be15fc5e5fafc8abf2fbeaf8350f1c96f25c0e8e91a532dda502df6a` |

M7-006 historical fixture bytes/hashes remain immutable. Successor hashes
change because the B0 capability list and functional product commit changed;
B1-B8 registry row source is byte-identical to the approved base.

Formal candidates `0001/` and `0002/` remain immutable and are excluded from
acceptance: `0001` preceded the persisted final-audit validator; `0002` had an
overbroad summary regex that counted a phrase in full-diff output. Candidate
`0003` corrected only evidence orchestration and reran every command.
