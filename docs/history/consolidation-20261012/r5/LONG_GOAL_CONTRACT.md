# UNIFIED-H1-R3 next-run readiness contract

## Outcome

Produce one independently checked, CPU-only final candidate and readiness dossier that can be used to request a future real H1 run. The result may say `READY_TO_REQUEST_REAL_RUN` only after every CPU condition in the approved work package is independently covered on the same final candidate.

This work does not authorize any real run. It does not approve or create D0006, release D0002, retry D0005, query a GPU, construct AppLauncher, step a simulator, train, write a real checkpoint, merge, push, or change the frozen runtime.

## Fixed history

- D0005 is sealed and single-use. Its accounting remains: one live start, one simulator-initialization slot reserved, zero AppLauncher constructions, zero physics ticks, zero transitions, zero PPO updates, zero optimizer steps, zero checkpoints, zero retries, and no remaining owned process.
- D0005 failed at `G1_PRELAUNCH_RESOURCE_WINDOW`. The old collector discarded the three samples. Their timestamps and GPU values cannot be reconstructed or invented.
- A0007 CPU work is historical and remains on HOLD. Its unused role calls and wall time are not part of this package.
- The prior 12-call A0009 pilot proposal remains unapproved and must not be started.

## Frozen scientific and runtime boundaries

- Method: `unified-h1-v1`, SHA256 `4d0186ecef8d1feda7af6d27766e50bbf78c9db9f4446ab224c074736f71b612`.
- Product: commit `2248ebb380260af86536f78b70e1235fcb401a7a`, tree `e44363a4b8504fda9e8f1b58a1fb942ef75c586e`.
- A0009 source candidate: commit `fd47dd321d571c4d6c983048f77bb3913e7e7b01`, tree `f2ccbe684e9d0fb56a861152b91ea4b4870500ee`, diff SHA256 `82b56cecc7680f8d57f23353c1caf5c53898ac55f1b76327ab3f8c703a194672`.
- Source config SHA256: `824feeb87fccb33ddb5b56b401b6591688b042606fda7d45f4d5f92c041f110b`. This is the frozen semantic reference. Final integration may update only repository path/commit/tree/branch/source-identity fields needed to bind the new isolated candidate. Method, evaluation, thresholds, seeds, budgets, schedules and all other scientific values must stay identical and the new config hash must be reported.
- Frozen Python: `/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11`, SHA256 `efa5468dc4ff3850e950cb68c0e430b3a1621e2c6bd9e0ef48f784ba90a9667a`.
- Frozen runtime: `/home/mint/rl_blackwell_test`, read-only.
- Reviewed local dispatcher SHA256: `84c1e65b2099eb7115e45187430f86b959a1f13f6eae98d7f5e206e06844b3d8`.
- Reviewed adapter SHA256: `fc089a87d14d14750cf45fec5a77d266dde6bc82506f99be52581ee3ba6e4dac`.
- Resource gate stays at three samples whose first-to-last timestamp span is at least ten seconds, utilization at most five percent, used memory at most 1024 MiB, free memory at least 10240 MiB, and only the already allowed desktop compute app at at most 256 MiB. No threshold or sample-count change is allowed.
- Actor, critic, reward, evaluation, geometry, ray, PPO, checkpoint, cleanup and training contracts stay unchanged.

## Authorized engineering scope

The isolated candidate may change only the approved workload modules, their CPU tests, and new `readiness_artifacts/` records. A repair must be necessary to satisfy the fixed contract and must be independently rechecked. The product snapshot, control snapshot, review bundle, historical evidence and this contract are read-only. The candidate config is writable only for the final repository-identity binding described above.

The prepared baseline intentionally preserves the source config's absolute workload path. That source checkout now contains an older untracked bytecode cache, so an unguarded `load_config()` rejects it as dirty. The final integrated candidate must close this as an identity-binding issue inside the isolated repository and then rerun the affected CPU suite; it must not delete or rewrite the source checkout's cache.

CPU-only fault injection may exercise the held-child lifecycle, STOP and duplicate-start gates, cleanup, evidence collection, late evidence, unknown accounting and idempotent reconciliation. Test doubles must make GPU, AppLauncher, simulator and training construction fail if reached.

## Required final dossier

The final candidate must be a clean commit in this isolated repository. `readiness_artifacts/` must identify the base and final commits and trees, cumulative diff hash, changed files, exact protected-input hashes, commands and exit codes, independent condition evidence, discovered failures and repairs, D0005 preservation, resource accounting, and remaining limits. It must distinguish process exit, engineering validity, scientific validity and formal approval.

The dossier may conclude only one of:

- `READY_TO_REQUEST_REAL_RUN`: all approved CPU conditions independently pass on the same clean candidate; a future real-run authorization is still required.
- `NOT_READY`: list the exact unresolved condition, preserved evidence and required next authority or input.
