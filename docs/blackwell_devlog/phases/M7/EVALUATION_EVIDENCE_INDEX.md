# M7-001 Evaluation Evidence Index

Coordination root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-001/M7-001-D0001`

## Persisted programs

- `generate_fixture.py`: exact hand table and provenance writer.
- `generate_record.py`: independent-process canonical record writer.
- `compare_records.py`: complete ordered, finite, exact comparator.
- `generate_mutations.py`: five fail-closed comparator mutations.
- `audit_source.py`: AST/source/information-flow audit.
- `run_formal_validation.py`: command runner and ledger producer.

## Accepted formal candidate

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

## Source provenance at accepted base

| Source | SHA256 | Consumed semantic |
| --- | --- | --- |
| `observations/audit.py` | `9d96f31852ba06b373c4c42513b6742cde6f385cd27b085e8c48ca33a58af929` | exact M4 actor audit |
| `control/ics.py` | `87dba0e3f4d2e252b74fb8b89954b0673e07fe07748cf8b3a36dea6f7c03b6ac` | beta/emergency/worst margin |
| `control/anchor.py` | `2d73af1c2f58fee40e616826771b9e97b07758556f6ce5ab66b398bbcaea4c8d` | anchor error/validity source |
| `task/rewards.py` | `a9b6bae7b8bdf46feb06a23782ceb7faea0e86b999648b2d0bccfec69f308ccc` | tracking/preservation gates |
| `task/termination.py` | `90be9cf7d68d54f520866737a1aef3c449f5b3b52bfe62a9666d405756b59d89` | independent masks/precedence |
