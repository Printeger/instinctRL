# M7-008 B0 Manifest Execution Evidence Index

## Namespace

`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-008/M7-008-D0001/`

## Persisted programs

- `generate_execution_record.py` and `compare_execution_records.py`
- `generate_manifest_fixtures.py` and `generate_negative_matrix.py`
- `validate_execution_manifest.py`
- `audit_execution_source.py` and `audit_integrity.py`
- `run_formal_validation.py`
- `final_audit.py` and `run_final_audit.py`

## Candidates

| Candidate | Status | Location |
| --- | --- | --- |
| `0001` | REJECTED: one obsolete M7-007 reverse-import assertion | `formal_candidates/0001/` |
| `0002` | PASS, superseded by corrected final-audit program index | `formal_candidates/0002/` |
| `0003` | ACCEPTED: all 44 commands matched expected exits | `formal_candidates/0003/` |

Candidate directories, command ledgers, stdout, stderr, fixtures, negative
manifests, process records, and hashes are immutable after execution.

Accepted candidate 0003 exact indexes:

- `SUMMARY.json` SHA-256:
  `56b1e7380f3fd099a9ca7dd865bdf834abf6b502851c5a8a541530005aa14a7e`
- `COMMAND_LEDGER.json` SHA-256:
  `7ed05f951d02aa6a593aeab27d625a5bf7fd2353a039dc0dc1693923c0bf628f`
- Functional commit: `025f22f0f20f81eff121c7dfef14a4bdfa35de6e`
- Import-boundary repair commit: `dea6a723d3daffe2ece7881e2460d59a3dc6663d`
- Candidate 0003 product HEAD: `6f6da160b65839bc9575e14007d2218218c29a4e`
