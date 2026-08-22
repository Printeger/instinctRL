# M6-003 PPO Tensor-Core Evidence Index

Fresh coordination namespace:
`attempts/development/M6-003/M6-003-D0001/`.

## Passing functional candidate

- `functional_candidates/0002/COMMAND_LEDGER.json`: 41 exact commands with
  cwd, selected environment, UTC times, program hashes, expected/actual exits,
  stdout and stderr paths.
- `functional_candidates/0002/VALIDATION_SUMMARY.json`: all seven aggregate
  assertions PASS.
- `functional_candidates/0002/OUTPUT_MANIFEST.json`: every derived artifact
  size and SHA256.
- `functional_candidates/0002/raw_logs/`: raw streams and exit files.
- `functional_candidates/0002/derived/synthetic.pt`: persisted shared batch.
- `derived/process_{1,2}/smoke.pt`: complete independent output trees.
- `derived/process_{1,2}/checkpoint.pt`: atomic strict checkpoints.
- `derived/process_{1,2}/inspection.json`: semantic assertions.
- `derived/comparison.json`: exact complete-tree result.
- `derived/comparator_mutations/` and `checkpoint_mutations/`: persisted
  negative fixtures and results.
- `derived/environment_regression.pt`: the one authorized accepted environment
  regression.

## Program inventory

`generate_synthetic_batch.py`, `inspect_ppo_smoke.py`,
`generate_comparator_mutations.py`, `generate_checkpoint_mutations.py`,
`validate_checkpoint_mutations.py`, `audit_training_source.py`, and
`run_functional_validation.py` were persisted before formal execution.

## Key hashes

- functional ledger: `4d4333d096fe78986289bcafaa627882c0a0d2186ebe2c98ef4c5400bb960510`
- functional summary: `ed2f5f28d9f7aebc217f8ad93bc8239b24653cabfddee4d9273e46a989bee761`
- output manifest: `35fe62b41408f39066a4a21075eecf26d8a9ff4fe91d1e9faaf8634914b2f285`
- each complete smoke: `cee1883680360fd1a1028b74813225202933556148143af4e75866d0a68b0ca9`
- comparison JSON: `e735ae1147582f0d4c589075bb78a3f7a6660ba7c77b21c723849b367b30006a`

`functional_candidates/0001/` is retained FAIL with an explicit failure
summary. No expected-nonzero or failed command is silently classified PASS.
