# M7-001 Evaluation Evidence Index

## Revision 1 / D0002

Fresh coordination root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-001/M7-001-D0002`

D0002 retains revised source audit, D0001 evidence verifier, fixture and record
generators, exact comparator/mutations, formal runner, final audit, command
ledgers, complete output trees, requirement map, and Executor report. D0001 is
consumed read-only and its complete 315-file manifest is checked byte-for-byte.

Accepted `formal_candidates/0001/`:

- `COMMAND_LEDGER.json` — 44/44 expected exits; SHA256
  `df66dea5cd42da181bc538d39eaa020d0c6c96b75ddce0b6eb57f9cf569dc7e7`.
- `VALIDATION_SUMMARY.json` — PASS, 7/7 assertions; SHA256
  `40c8f77d64a72659f481faf648d62ee0339c3668b505eec823efc3ad6cd1d857`.
- `OUTPUT_MANIFEST.json` — complete derived tree; SHA256
  `2c5b778c9903fe11429526a71407ebb9cfb23afd4660f0ddad013c2a4fdc5b5c`.
- `derived/fixture.pt` — SHA256
  `17c686bcb7eff6c80cd9c3f8f8b31bcae9242cb529d16835ef63a7f96d2ed28f`.
- both `derived/process_*/record.json` — D0001-identical SHA256
  `38d1f7462247b2211a636e1c2160b9dcade5153ca4ca707b97213d9fdadd2cea`.
- `derived/comparison.json` — PASS, 78 numeric leaves, maximum error 0.0.
- `D0001_EVIDENCE_BASELINE.json` — 315-file immutable baseline; SHA256
  `c17231b865fc760667b839c41cebf7a06e97c2cb35511b5e28106c10412ee6e6`.

## D0001 (historical)

Coordination root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-001/M7-001-D0001`

### Persisted programs

- `generate_fixture.py`: exact hand table and provenance writer.
- `generate_record.py`: independent-process canonical record writer.
- `compare_records.py`: complete ordered, finite, exact comparator.
- `generate_mutations.py`: five fail-closed comparator mutations.
- `audit_source.py`: AST/source/information-flow audit.
- `run_formal_validation.py`: command runner and ledger producer.

### Accepted formal candidate

`formal_candidates/0002/` contains:

- `COMMAND_LEDGER.json` — 38 exact argv/cwd/environment/time/program-hash/
  stream/expected-exit/actual-exit records; SHA256
  `17d9fd285f7379ce807a3103c1fe8e72e8070ee4bfec0ba63b0e70248b36dfc8`.
- `VALIDATION_SUMMARY.json` — PASS; SHA256
  `4d564af99b9b1304250ba0af2a42eb803f1cdc006eceaebb954306db79919cb1`.
- `OUTPUT_MANIFEST.json` — complete derived-output tree; SHA256
  `2c5b778c9903fe11429526a71407ebb9cfb23afd4660f0ddad013c2a4fdc5b5c`.
- `raw_logs/` — stdout, stderr, and exit file for every command.
- `derived/fixture.pt` — SHA256
  `17c686bcb7eff6c80cd9c3f8f8b31bcae9242cb529d16835ef63a7f96d2ed28f`.
- `derived/process_1/record.json` and `process_2/record.json` — each SHA256
  `38d1f7462247b2211a636e1c2160b9dcade5153ca4ca707b97213d9fdadd2cea`.
- `derived/comparison.json` — PASS, 78 numeric leaves, maximum error 0.0.
- `derived/mutations/` — missing, extra, reordered, nonfinite, any-error inputs.

Rejected formal candidate `formal_candidates/0001/` is retained unchanged and
identified in the test report. It is not the accepted evidence candidate.

### Source provenance at accepted base

| Source | SHA256 | Consumed semantic |
| --- | --- | --- |
| `observations/audit.py` | `9d96f31852ba06b373c4c42513b6742cde6f385cd27b085e8c48ca33a58af929` | exact M4 actor audit |
| `control/ics.py` | `87dba0e3f4d2e252b74fb8b89954b0673e07fe07748cf8b3a36dea6f7c03b6ac` | beta/emergency/worst margin |
| `control/anchor.py` | `2d73af1c2f58fee40e616826771b9e97b07758556f6ce5ab66b398bbcaea4c8d` | anchor error/validity source |
| `task/rewards.py` | `a9b6bae7b8bdf46feb06a23782ceb7faea0e86b999648b2d0bccfec69f308ccc` | tracking/preservation gates |
| `task/termination.py` | `90be9cf7d68d54f520866737a1aef3c449f5b3b52bfe62a9666d405756b59d89` | independent masks/precedence |
