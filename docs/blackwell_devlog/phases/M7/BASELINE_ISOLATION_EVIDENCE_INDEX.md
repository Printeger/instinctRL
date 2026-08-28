# M7-006 Baseline Isolation Evidence Index

Accepted evidence root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-006/M7-006-D0001/formal_candidates/0002/`

Every one of the 42 ledger rows records exact argv, cwd, selected environment,
UTC start/end, expected/actual exit, stdout/stderr path, and stream SHA-256.

| Evidence | SHA-256 |
| --- | --- |
| `SUMMARY.json` | `ae30cb84cc93140b8e0a8a141cb8460a4e4dacde4d7aa55a31d1084153ce805e` |
| `COMMAND_LEDGER.json` | `68641c5dcbf6ef776a72fd0e221c678b913e0825951cc77b1f2a55d9a5042cb8` |
| `fixtures/FIXTURE_INDEX.json` | `c4151d146efa5c4383e79e98f073628d694cba231486fce1c22d8d7be31035ec` |
| `negative_matrix/INDEX.json` | `a22ca5fdc7a1fe4925a77a13439d2e1987e7742e8957d303fa4fdb3bfa2ab9a3` |

## Canonical manifest fixtures

All fixtures bind product commit
`93602e05238cf207bfefea235cbaadef35c957f6`, platform `TASLAB_UAV`, sensor
`Livox MID360`, scenario `S1.corner`, and seed 7.

| Fixture | Status | Manifest SHA-256 | File SHA-256 |
| --- | --- | --- | --- |
| B0 | READY | `14ec38cb1b451ee5a9c4f32bed282e34f74151bd0f7cb6202319be5b2af58f5f` | `186ceff7aa7e4defbb867ad2cf34d93b68668c8210d1bfe77dfc326a39ccbf62` |
| B0 | EXECUTED | `5219b1c70a073ba70c10429412bb17b26b16ae546d6c3fe343552a87f57322b0` | `0f588432cf981e2c0fc242506f4c6b6604b0d8709679fdad6b5e5df87eb899c9` |
| B1 | NOT_READY | `72ec3759770e4357a6c18fadc871255ab7802e171ff28efdd06c971d96a20328` | `48ed229ef5a12c59dd2137e9a651db192f2c0228f7c13abf31b99df4da01b183` |
| B8 | SKIPPED | `1dd8893ff79942e5f652acecc512a6af9581d462193f882e4422484b52f17e6e` | `f2ac78915d74cd292c2536848625803b6498d8b7b5e9e7a5c05d2d1cb8b5ecc6` |

The formal summary enumerates SHA-256 for every program, fixture, mutation,
stdout, and stderr artifact. `formal_candidates/0001/` is retained but rejected:
its caller supplied nonexistent SHA
`93602e07d6c474770306ae1a01cd0cb0ed5d066a`; integrity command 029 exited 1
before pytest. No file from candidate 0001 supports acceptance.
