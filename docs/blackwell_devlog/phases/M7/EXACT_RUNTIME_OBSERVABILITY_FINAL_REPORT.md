# M7-005 Exact Runtime Observability Final Report

## Scope completed

The environment now exposes a named detached raw source beside unchanged `extras.pre_reset`. It captures rays once at the exact task-snapshot sensor update and preserves the terminal source through automatic reset. The environment never imports evaluation/Warp, resolves a mesh, or names/stores a surface normal.

The M7-004 adapter now owns frozen mesh-cache resolution and validates the pinned RayCaster/Warp baseline before exact normal acquisition. The unchanged M7-003 core computes rank, spectrum, `sigma_min`, condition, weak direction, and valid-ray statistics. A normal-free, independently owned and SHA-256-sealed result crosses into the collector. The canonical M7 metric uses exact `sigma_min`, validity true, and source `EXACT`, including zero.

## Correctness gates

- Source schema, same-update ownership, and reset binding: **PASS**.
- Missing/extra/reordered/type/shape/dtype/device/nonfinite/alias/stale/duplicate/skipped/step/reset/post-reset/mutation rejection: **PASS**.
- Unchanged `pre_reset`, policy, critic, actor, controller, reward, and task semantics: **PASS**.
- Actor isolation: **PASS**.
- Exact metric/detail mapping and aggregation compatibility: **PASS**.
- Two fresh live processes: **PASS**, each 8 policy steps, 16 transitions, 160 physics steps, and 339,840 rays.
- Complete-tree determinism: **PASS**, 73 tensors and 1,022,282 elements at maximum error zero.
- Frozen/package/formal-source integrity: **PASS**.

## Change ledger

| File | Change | Requirement | Semantic effect |
| --- | --- | --- | --- |
| `taslab_direct_env.py` | Same-update owned raw ray source | Source boundary/lifecycle | Adds evaluation-only sibling; no observation/control/reward change |
| `raycaster_observability.py` | Guarded mesh resolver and ordered-source acquisition | Confine private cache/normal work | Exact acquisition from detached public source |
| `runtime.py` | Sealed distilled result, exact canonical mapping, detail tree | Collector/detail contract | Replaces runtime unavailable placeholder with exact values |
| Runtime runner/tests | Live wiring, strict negatives, comparator mutations | Proof obligations | No deployed runtime semantic change beyond evaluation logging |

## Recommendation

`READY_FOR_REVIEW`

No proxy, plot, B0-B8, training, evaluation expansion, M8, 128k/1M/scale run, package mutation, frozen-tree mutation, history rewrite, merge, or push was performed.

`NO NEXT-PHASE WORK WAS STARTED.`
`AWAITING COMMANDER REVIEW.`
