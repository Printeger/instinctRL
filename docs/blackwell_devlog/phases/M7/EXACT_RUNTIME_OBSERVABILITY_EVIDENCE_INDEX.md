# M7-005 Exact Runtime Observability Evidence Index

Evidence root: `/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-005/M7-005-D0001/formal_candidates/0002/`

The accepted formal run contains 42 ledgered commands and 22 expected-negative comparator executions. Every ledger row records argv, cwd, selected environment, UTC start/end, actual/expected exit, stdout/stderr path, and stream SHA-256.

| Evidence | SHA-256 |
| --- | --- |
| `SUMMARY.json` | `d74b9173d44ef5cffbbb825b69f93e3cf635cdf413d005c3ec75e9c426182724` |
| `COMMAND_LEDGER.json` | `8a85c5638ee5d1149480c7577c69fdf4508e2b9782fa06900b762d7ef3ddce5d` |
| `derived/live_1.pt` | `6af4c26caa23a882f4ebf951ad8f3c09ab2dd1c4424c27948d43afa3e4da78ff` |
| `derived/live_2.pt` | `560b5cf9c8f29cc0a844756e96396397168309b6297e6c8db4241ebbcd6e9cdd` |
| `derived/inspection_1.json` | `15bcf8fe9be52c051e17b9beed3ef2d33ff2ecdbb3d5641777fb145c766667bc` |
| `derived/inspection_2.json` | `1b47da42b60a8a4e6e621119f8f348dba9d37ed398b65a73787d75a661784823` |
| `derived/comparison.json` | `b35f413ab9762f6fea701d4704c218e050d4f659672234527e9492698650b319` |
| `raw_logs/009_live_1.stdout` | `bc3780fd4731fb4e6cb2bfe4e7535370197315f0dc2fac5909e01096b06cc07b` |
| `raw_logs/010_live_2.stdout` | `1963ec9d254f13cdfe40555b4041702f3a210d2003464ea70cb244f931f604b7` |
| `raw_logs/037_integrity.stdout` | `09dbac7833fa614f47604a882a3434fcf2ae309d27e89c3d18fb126071908e18` |

Formal program hashes and every derived mutation/artifact hash are enumerated in `SUMMARY.json`. The failed `formal_candidates/0001/` retry is retained: it used an incorrect caller-supplied provenance SHA and then failed the original over-strict acquisition-error inspector. It is not acceptance evidence and was not overwritten.
