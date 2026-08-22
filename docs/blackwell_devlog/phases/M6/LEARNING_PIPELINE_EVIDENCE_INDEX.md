# M6-004 Bounded Learning-Pipeline Evidence Index

Fresh evidence is under
`attempts/development/M6-004/M6-004-D0001/` in the separate coordination
repository.

## Formal candidate 0001

- `COMMAND_LEDGER.json`: 41 exact argv/cwd/environment/timestamp/program-hash/
  stream/expected-exit/actual-exit records.
- `VALIDATION_SUMMARY.json`: eight aggregate assertions, all PASS.
- `OUTPUT_MANIFEST.json`: complete derived-tree byte sizes and SHA256.
- `raw_logs/`: stdout, stderr, and exit files for all formal commands.
- `derived/fixture.pt`: fixed ordered 2-env, 2x4-step input configuration.
- `derived/process_{1,2}/smoke.pt`: complete independent pipeline trees.
- `derived/process_{1,2}/checkpoint.pt`: strict update-one checkpoints.
- `derived/process_{1,2}/inspection.json`: 23 semantic assertions each.
- `derived/comparison.json`: exact 136-tensor result.
- `derived/mutations/`: five complete persisted negative trees.
- `derived/accepted_ppo/`: accepted M6-003 tensor smoke/checkpoint replay.
- `derived/accepted_environment.pt`: accepted M6-002 runtime replay.

## Persisted programs

`generate_fixture.py`, `inspect_smoke.py`,
`generate_comparator_mutations.py`, `diagnose_comparison.py`,
`audit_rollout_source.py`, and `run_formal_validation.py` were persisted before
their relevant formal execution. Final history/allowlist self-audit is indexed
in the coordination `INDEX.md` after the documentation commit.

## Formal candidate SHA256

- command ledger: `e4286cf880350a3611ff73ec5fffba5a6f8b50d64c68f409e4bd168ab354270a`
- validation summary: `92810252042ccd8e337e653216c5ff12eca9422a37669abd20586c7c70d7d5bd`
- output manifest: `88622616aa57fe50596725fe7f08a94eafc9cbce4311a9d3877c7ac74983dd49`
- fixture: `29589407f527b9c16bb386783a14d77dfb1ec3cda9e781cb0e814ba6dbc9e955`
- each complete smoke: `264d8995f2852662d1d528650198d66b5aa772d5407f345bf4fd95267be66f04`
- each checkpoint: `e64f3553a0baa6231ec04d97637922b902c406be1299a309c44c41a44ade35ab`
- comparison JSON: `f270facf072bccf6e4faa9fb95d67fd98d4286d71d4437fa69cd28323efcd4cc`
