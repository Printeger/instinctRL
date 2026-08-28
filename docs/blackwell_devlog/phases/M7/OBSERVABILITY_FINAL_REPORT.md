# Exact Observability Final Report

## Scope completed

The bounded pure evaluation core now validates exact caller-owned normal/mask
tensors, forms `J_i = -n_i^T`, and returns rank, ascending spectrum,
`sigma_min`, explicit condition state, explicit unique weak direction, and
valid-ray/fraction statistics. All results are finite, detached, owned, and
non-aliased. Degenerate states are explicit rather than hidden.

Hand-computed open-field, wall, corridor, isotropic-corner, anisotropic, and
mixed batches pass in float32/float64. Permutations and sign flips preserve
invariants; repeated singular values never fabricate a direction. Malformed,
nonfinite, non-unit, placeholder, view, alias, dtype/device, and contract
violations fail closed.

## Actor and runtime boundary

Recursive source audits find no observability-core import or surface-normal
field in actor, observation, control, environment, task, or training modules.
Perturbing exact normals changes the observability result but cannot change the
accepted real `HistoryActor` alpha, beta, sampled action, or log probability;
changing an approved policy field is discriminated. Existing M7-002
observability fields remain exactly finite zero, invalid, and `UNAVAILABLE`.

## Correctness gates

- Exact Jacobian, scale-aware rank, condition availability, weak-subspace
  uniqueness, canonical sign, and zero-hit rules: PASS.
- Synthetic, validation, ownership, determinism, and actor isolation: PASS.
- Accepted pure/static regressions twice: PASS.
- Evaluation-only import boundary and accepted M7 byte integrity: PASS.
- Package/runtime/formal/frozen/external integrity and exact Git scope are
  recorded in coordination evidence.

## Exclusions and recommendation

No RayCaster/Isaac Lab acquisition, runtime wiring, finite-difference or
hardware proxy, plots, scenario/ablation execution, PPO/update, training,
large-scale run, M8 work, merge, push, or history rewrite was performed.

Executor recommendation: `READY_FOR_REVIEW`. This is not self-approval and
does not authorize the next milestone or any deferred integration.
