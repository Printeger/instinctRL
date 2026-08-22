# M6 Pure-Tensor Task Final Report

Order: `M6-001`, revision 1

## Scope completed

- Added strict actual-velocity-primary reward computation with exact exposed
  components, null-command anchor/station semantics, safety, ICS compliance,
  intervention, smoothness and secondary command proxies.
- Added separate collision/height `terminated` and timeout `truncated` masks,
  exact boundary semantics and deterministic simultaneous-cause precedence.
- Added fixed-order `[N,18]` critic-only root/body-velocity/clearance/collision
  schema with scalar-first quaternion validation.
- Added analytic, negative, gradient, actor-isolation, pure CUDA persistence and
  complete zero-error comparison tests.
- Closed revision-1 review findings by consuming accepted M5 floating binary
  metrics directly, requiring usable fraction with exact `0.1` anchor gating,
  and replacing the stale failed-candidate pointer with the stable D0002 root.

## Changed-file ledger

| File/group | Change | Requirement | Semantic impact |
| --- | --- | --- | --- |
| `instinctlab/task/rewards.py` | config, validation, components and total | reward boundary | reward-only computation |
| `instinctlab/task/termination.py` | masks, codes and precedence | task reset semantics | done classification |
| `instinctlab/task/critic.py` | fixed privileged schema | asymmetric critic boundary | critic-only tensor |
| `instinctlab/task/__init__.py` | public exports | stable task surface | exposes M6 APIs |
| three M6 test programs | CPU/CUDA/negative/complete comparison | correctness evidence | test only |
| README and six M6 records | scope/contract/results/provenance/limits | traceability | documentation only |
| `instinctlab/task/rewards.py` revision-1 delta | exact M5 floating labels and usable gate | interface closure | anchor validity only |
| M6 unit/CUDA tests revision-1 delta | direct M5 metrics and strict negatives | review closure | test only |
| five M6 records revision-1 delta | corrected interface and stable D0002 root | traceability closure | documentation only |

## Correctness result

All task tensors fail closed on missing/malformed/domain/device/dtype/nonfinite
inputs. Actual body velocity, not `v_final_b`, drives primary tracking. ICS
attenuation removes ordinary tracking punishment while intervention usage stays
penalized. Every exposed component has `[N,1]` shape and the total is its exact
unscaled sum. Privileged perturbations change reward and critic tensors but not
the exact M4 actor or accepted M5 command/control outputs.

Accepted M5 `active`, `usable_fraction`, `huber_loss`, `beta`, and `emergency`
tensors now flow directly into reward computation. Floating labels must be
finite exact 0/1 and usable fraction must be finite in `[0,1]`. Anchor reward
requires null command, active label, and usable fraction at or above `0.1`;
insufficient anchors contribute zero. Collision remains boolean. No unrelated
formula, component order/total, termination, critic, actor or control semantic
changed.

Two independent pure CUDA outputs compare 110 tensors and 1,020,440 elements
with maximum error zero. Accepted M1-M5, frozen packages/runtime/external
repositories and Git configuration remain unchanged.

The committed Evidence Index now resolves through the stable
`M6-001-D0002/INDEX.md` root. D0001 and all its candidates remain preserved as
historical failed-review evidence; no failed candidate is claimed as revision-1
acceptance evidence.

## Stop declaration

```text
NO PPO/TRAINING/ENVIRONMENT-INTEGRATION/M7 OR LATER WORK STARTED.
AWAITING COMMANDER REVIEW.
```
