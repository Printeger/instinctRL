# M7-012 Exact B0 Runtime Evaluation Evidence Index

## Accepted candidate

Coordination root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-012/M7-012-D0001/formal_candidates/0001`

| Evidence | SHA-256 |
| --- | --- |
| `SUMMARY.json` | `ae7d44af4b183361038251bdc729089cb235273cf4a35f234b3333e0a59858fd` |
| `COMMAND_LEDGER.json` | `4d4b6158a46015fc6b42ce251dd873d9b09f3369c298a73c44f1fb4a8efee308` |
| `REQUIREMENT_MAP.json` | `f982a8bd4247c2d0135a10baf2aeff9845bc833c39b817507eef0dc322730ff8` |
| `SMOKE_VALIDATION.json` | `7eecebef9acc0fa7fa6d0aeba1fdbb0c44139109bc59cfe0eff7303e770773c2` |
| `COMPARATOR_MUTATIONS.json` | `ab7da661695d379bffd5228515069fb58197948776c956632708811133bae39b` |
| `smokes/runtime_exact_comparison.json` | `f0077d74709def05c1d74521967de3e05bdb5a0357b01a53097fc63f996d701b` |

The command ledger indexes raw stdout and stderr for all 37 formal commands.
The requirement map maps each Order gate to those commands. `SUMMARY.json`
also hashes every persisted formal program and candidate artifact.

## Process identities

| Artifact | Process 1 | Process 2 |
| --- | --- | --- |
| Complete bundle SHA-256 | `1c58d555e41897a8f9fd70143df3544784c9d31cb622f7dab0c37d02f8c293b1` | `d0649ac67190d83d1a0ad2a059239569faf4fec2e2a9ac667eddd0656bbd32fa` |
| Canonical record SHA-256 | `b5ce5b6342efe151a8013a8031af3ea299f89c27cc98774df665e97f07d6f51d` | `b5ce5b6342efe151a8013a8031af3ea299f89c27cc98774df665e97f07d6f51d` |
| Receipt summary SHA-256 | `56c8aab64cd91f9470f0a942cecb2708c1d4f8f45e6b90680cba8f46e1f5967f` | `56c8aab64cd91f9470f0a942cecb2708c1d4f8f45e6b90680cba8f46e1f5967f` |

The two Torch container byte hashes differ because they are independent archive
files; their loaded complete ordered content is exactly equal. The comparator
proved 41 tensor values and metadata covering 3,360 elements plus every scalar,
string, list, tuple, mapping order, record and receipt field.

## Auditors and raw evidence

Fresh persisted programs under the M7-012 attempt root include
`audit_source.py`, `audit_integrity.py`, `audit_call_state_privacy.py`,
`negative_matrix.py`, `mutate_comparator.py`, `validate_smoke_outputs.py` and
`run_formal_validation.py`. Formal raw logs are under candidate
`0001/raw_logs/`; complete CUDA bundles and summaries are under
`0001/smokes/`; rejected mutated bundles are under
`0001/comparator_mutations/`.

No PNG, video, checkpoint or large log was committed to the product repository.
