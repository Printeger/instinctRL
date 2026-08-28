# M7-006 Baseline Isolation Evidence Index

## D0002 correction evidence

Accepted evidence root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-006/M7-006-D0002/formal_candidates/0001/`

The 55-row `COMMAND_LEDGER.json` records exact argv, cwd, isolated environment,
UTC start/end, expected/actual exit, stdout/stderr path, and stream SHA-256.
`SUMMARY.json` inventories every persisted program and artifact hash. The
coordination-level `M7-006-D0002/EVIDENCE_INDEX.md` and Executor report record
the external SHA-256 of the completed summary, ledger, and final audit.

Key evidence is indexed as follows:

| Evidence | Location |
| --- | --- |
| Four byte-preserved positive manifests | `formal_candidates/0001/fixtures/` |
| 27 persisted malformed manifests, including 10 boolean-type cases | `formal_candidates/0001/negative_matrix/` |
| Both Commander reproduction programs and exits | ledger 039-040 and `raw_logs/` |
| Exact source/import/allowlist/D0001/frozen integrity | ledger 038 and 041 |
| Repeated focused/M7/regression/static suites | ledger 042-048 |
| Full history, two diff scopes, stat, full diff, hygiene, clean state | ledger 049-055 |
| Corrected-HEAD immutable audit | `final_audit_0001/` |

## Preserved D0001 evidence

The accepted D0001 root remains:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-006/M7-006-D0001/formal_candidates/0002/`

| Evidence | Preserved SHA-256 |
| --- | --- |
| `SUMMARY.json` | `ae30cb84cc93140b8e0a8a141cb8460a4e4dacde4d7aa55a31d1084153ce805e` |
| `COMMAND_LEDGER.json` | `68641c5dcbf6ef776a72fd0e221c678b913e0825951cc77b1f2a55d9a5042cb8` |

The D0002 fixture generator binds the preserved functional commit
`93602e05238cf207bfefea235cbaadef35c957f6`, so all four D0001 fixture files,
canonical payload hashes, field order, and bytes remain exact. The corrected-
HEAD final audit compares each D0002 fixture byte-for-byte with D0001.
