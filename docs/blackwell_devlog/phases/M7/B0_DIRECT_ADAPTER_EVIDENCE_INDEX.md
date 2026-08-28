# M7-007 B0 Direct Adapter Evidence Index

Accepted evidence root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-007/M7-007-D0001/formal_candidates/0001/`

The 40-row command ledger records exact argv, cwd, isolated environment, UTC
timestamps, expected/actual exits, stdout/stderr paths, and stream hashes.

| Artifact | SHA-256 |
| --- | --- |
| `SUMMARY.json` | `febf0bd1b317c4d29c10ab230a0f2f5db7ff97fad3bcfa68c7dbce7ac631407f` |
| `COMMAND_LEDGER.json` | `721c4bf7d83a447b556b410528e30cf9e5e20a38fd0530b191f0b9febcc8e43a` |
| `fixtures/FIXTURE_INDEX.json` | `689363a22b7c20cfb4b074b3eea5a2e55b7a73920db772703301549f9d7e6c9a` |
| `negative_matrix/INDEX.json` | `9dcfe1f31c6cf68b58caec0b714901008dd90e0642e70f6725b76ee5c08c272c` |
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
