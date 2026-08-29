# M7-011 Exact B0 TASLAB Environment Evidence Index

## Accepted candidate

Coordination root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-011/M7-011-D0001`

Accepted formal candidate: `formal_candidates/0002`.

| Evidence | Path | SHA-256 / fact |
| --- | --- | --- |
| formal summary | `formal_candidates/0002/SUMMARY.json` | status PASS; 1590 passed |
| command ledger | `formal_candidates/0002/COMMAND_LEDGER.json` | 37 exact command records |
| requirement map | `formal_candidates/0002/REQUIREMENT_MAP.json` | environment correctness gates |
| smoke validation | `formal_candidates/0002/SMOKE_VALIDATION.json` | `5aa33d710467f8acae0bd6ae38e6134898c52ebf8a4c3208729c244546b045ba` |
| environment process 1 | `formal_candidates/0002/smokes/environment_process_1.pt` | `d9821bf8a20990464b50df66fb939b49d6b30140ae5f27ade80e70e4aa2c472b` |
| environment process 2 | `formal_candidates/0002/smokes/environment_process_2.pt` | `da5e4ec27de0e854086a55b72756b8508731ae7e254aca72ef9f240acd0f830d` |
| exact comparator | `formal_candidates/0002/smokes/environment_exact_comparison.json` | `9dd3e14871ca33edee3cbc25cab928caff1844bc029672d72704755b7faacddf` |
| comparator mutations | `formal_candidates/0002/COMPARATOR_MUTATIONS.json` | `c08788625e617993c13db5661787cff6ebdf6065eb785188cab7347834e4e7cd` |
| M6 CUDA regression | `formal_candidates/0002/smokes/m6_environment.pt` | `d95fcb6302bffc2405f5e4292907167b98a49d60e5a31e0b3d08c2dd45ad5428` |
| M7-009 CUDA regression | `formal_candidates/0002/smokes/m7_controller.pt` | `fa2f3e384a59db3316d7ff7a3f0ae51de60f735c1451e36ae824cfe9f1c4a3cf` |
| raw stdout/stderr | `formal_candidates/0002/raw_logs/` | indexed by command ledger |

## Persisted programs

The accepted summary binds the SHA-256 of `audit_source.py`,
`audit_integrity.py`, `audit_call_state_privacy.py`, `negative_matrix.py`,
`mutate_comparator.py`, `validate_smoke_outputs.py`,
`run_formal_validation.py`, `final_audit.py`, and `run_final_audit.py`. The
product smoke runner and exact comparator are committed and bound by the
delivery commit. The final audit program and its ledger are indexed by the
Executor report after that commit exists.

## Runtime, determinism and privacy evidence

- Environment processes: `raw_logs/019_m7_environment_process_1.stdout` and
  `raw_logs/020_m7_environment_process_2.stdout`; each records 8 policy, 160
  physics, 160 lifecycle and 160 wrench calls across 320 samples.
- Exact comparison: `raw_logs/021_compare_environment_processes.stdout`; 43
  tensors and 35,256 elements exact, trajectory digest
  `15205de4ff92eb1a12ae9b6c32c43721397bab7375f0d5549244fc6db7657f5b`.
- Mutation sensitivity: `raw_logs/022_comparator_mutations.stdout`; all seven
  missing/extra/order/bit/count/receipt/private mutations rejected.
- Smoke gate validation: `raw_logs/023_validate_smokes.stdout`; count,
  finiteness, bounds, identity, disabled-core, finalization and cache-clear
  gates PASS.
- Source/import/allowlist audit: `raw_logs/006_source_audit.stdout`; exact new
  source SHA-256
  `714080d04dee16e73c408f6c0e8c7c8d4a5d0b5a603e698714c5a330c8509e5d`.
- Call/state/privacy and negative audits:
  `raw_logs/008_call_state_privacy_audit.stdout` and
  `raw_logs/009_negative_matrix.stdout`; exact ownership, state boundary and
  fail-closed behavior PASS.
- Frozen integrity audit: `raw_logs/007_integrity_audit.stdout`; formal source,
  Isaac Lab known state, OmniDrones, NavRL, Python, Torch and GPU identities
  PASS.

Candidate `0001` is immutable and superseded because its coordination-only
mutation driver had an indexing defect; its files are not acceptance evidence.
