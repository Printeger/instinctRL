# M7-006 Baseline Isolation Plan

## Scope

Add only the standard-library `instinctlab.evaluation.baselines` identity and
manifest layer, its explicit package exports, one focused pure test module, the
precise README note, and the six bounded M7-006 records. No baseline adapter,
scenario, simulator, evaluator execution, policy ranking, or training is in
scope.

## Sources of truth

- Order `M7-006` and base `dbc05a58ca98958c9100fa5e002fe6b9083e40ec`.
- Locked Handbook `Baselines and Ablations`, `Experiment Pipeline`, and audit
  requirements.
- Locked architecture actor boundary.
- Accepted M4 `APPROVED_INPUT_NAMES`, M5 B0 governor, and M7
  `EVALUATION_INPUT_KEYS`/source-manifest separation.

## Implementation sequence

1. Freeze exact B0-B8 IDs, switches, descriptions, execution families,
   schemas, toggles, capabilities, optionality, privilege, and readiness.
2. Validate the complete ordered registry and reject aliases, duplicate
   identities/providers, forbidden deployable fields, and false readiness.
3. Resolve exact IDs/switches and validate immutable ordered capability
   bindings against accepted evidence hashes.
4. Build and parse exact canonical JSON manifests with explicit statuses,
   actor audit, provenance, schema/toggles, and payload SHA-256.
5. Prove positive fixtures, strict negatives, mutation isolation, import
   isolation, and accepted-source byte identity twice.
6. Run accepted M2-M7 pure regressions twice, audit the allowlist/history/
   frozen inputs, persist evidence, and create exactly two nominal commits.

## Write and non-write ledger

Writes are limited to the ten `ALLOWED_PATHS` in M7-006. In particular,
accepted observation, control, environment, task, training, runtime,
observability, metrics, RayCaster, source-manifest, formal/frozen/external,
legacy, and Git-configuration paths remain unchanged.

## Stop conditions and risks

Stop on an identity/privilege conflict, unlisted-path dependency, accepted-core
semantic change, base/branch/clean mismatch, or missing external method
authority. Main risks are accidentally normalizing a selector, overclaiming an
unaccepted adapter, hiding B8 omission, allowing privileged input on a
deployable row, or serializing caller-owned/unordered state.

No CUDA, Isaac simulator, runtime smoke, package change, training, M8, scale,
merge, push, or history rewrite is planned or required.
