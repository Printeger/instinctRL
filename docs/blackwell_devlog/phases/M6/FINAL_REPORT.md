# M6 Pure-Tensor Task Final Report

Order: `M6-001`, revision 0

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

## Changed-file ledger

| File/group | Change | Requirement | Semantic impact |
| --- | --- | --- | --- |
| `instinctlab/task/rewards.py` | config, validation, components and total | reward boundary | reward-only computation |
| `instinctlab/task/termination.py` | masks, codes and precedence | task reset semantics | done classification |
| `instinctlab/task/critic.py` | fixed privileged schema | asymmetric critic boundary | critic-only tensor |
| `instinctlab/task/__init__.py` | public exports | stable task surface | exposes M6 APIs |
| three M6 test programs | CPU/CUDA/negative/complete comparison | correctness evidence | test only |
| README and six M6 records | scope/contract/results/provenance/limits | traceability | documentation only |

## Correctness result

All task tensors fail closed on missing/malformed/domain/device/dtype/nonfinite
inputs. Actual body velocity, not `v_final_b`, drives primary tracking. ICS
attenuation removes ordinary tracking punishment while intervention usage stays
penalized. Every exposed component has `[N,1]` shape and the total is its exact
unscaled sum. Privileged perturbations change reward and critic tensors but not
the exact M4 actor or accepted M5 command/control outputs.

Two independent pure CUDA outputs compare 108 tensors and 1,020,432 elements
with maximum error zero. Accepted M1-M5, frozen packages/runtime/external
repositories and Git configuration remain unchanged.

## Stop declaration

```text
NO PPO/TRAINING/ENVIRONMENT-INTEGRATION/M7 OR LATER WORK STARTED.
AWAITING COMMANDER REVIEW.
```
