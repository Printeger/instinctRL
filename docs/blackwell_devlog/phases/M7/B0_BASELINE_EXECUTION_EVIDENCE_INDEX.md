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
| `0002` | ACCEPTED: all 44 commands matched expected exits | `formal_candidates/0002/` |

Candidate directories, command ledgers, stdout, stderr, fixtures, negative
manifests, process records, and hashes are immutable after execution.

Candidate 0002 exact indexes:

- `SUMMARY.json` SHA-256:
  `1d366626fe28dfca273a018e4c427f49aec7e7903866e6c9ee04cf4c015ceb24`
- `COMMAND_LEDGER.json` SHA-256:
  `0b7d51e0ef0cc39b0ce2cd4ba2da3241360887ea4f55589cce9070aca8b5b9d4`
- Functional commit: `025f22f0f20f81eff121c7dfef14a4bdfa35de6e`
- Import-boundary repair commit: `dea6a723d3daffe2ece7881e2460d59a3dc6663d`
