# RayCaster Exact-Normal Acquisition Contract

## Frozen call boundary

`instinctlab.evaluation.raycaster_observability` is evaluation-runtime-only and
is deliberately absent from `instinctlab.evaluation.__init__`. It lazy-imports
the frozen public `isaaclab.utils.warp.raycast_mesh` and calls it exactly with:

```text
raycast_mesh(ray_starts_w, ray_directions_w, mesh=mesh,
             max_dist=maximum_distance_m,
             return_distance=True, return_normal=True)
```

No kernel is copied or patched. The adapter does not accept an environment,
actor, policy, reward, controller, or task object.

## Ordered inputs

| Input | Contract |
| --- | --- |
| `ray_starts_w` | owned finite float32 `[N,R,3]`, world metres |
| `ray_directions_w` | owned finite unit float32 `[N,R,3]`, world axes |
| `mesh` | frozen Warp mesh handle with identity/device metadata |
| `maximum_distance_m` | finite positive scalar metres |
| `accepted_ray_hits_w` | owned float32 `[N,R,3]`; finite valid hits, exact positive-infinity misses |
| `accepted_beam_valid` | owned bool `[N,R]` accepted MID360 mask |

All tensors share one device, own exact storage, and do not alias. `N` and `R`
are positive. Default immutable unit and distance-consistency tolerances are
`1e-5` and `1e-5 m`. The cast must not mutate any input or mesh identity.

## Qualification and handoff

The official four-result order is hit, distance, normal, face ID. Face ID must
remain `None` because it was not requested. Result shapes/dtypes/devices are
exact. Returned finite-hit classification must equal the accepted mask, and
returned hits—including every miss—must be tensor-exact with accepted sensor
hits, so hit maximum error is exactly zero.

For valid beams, returned distances are finite and agree with the Euclidean
same-state start-to-hit geometry within `1e-5 m`; normals are finite unit
vectors within `1e-5`. For invalid beams, returned hit, distance, and raw normal
preserve the frozen positive-infinity miss convention. The adapter then creates
owned exact-zero normal placeholders only for accepted invalid beams; it never
normalizes, flips, repairs, substitutes, or relabels a valid normal.

Qualified `[N,R,3]` normals/mask become owned `[1,N,R,3]` and `[1,N,R]` inputs
to the unchanged M7-003 `compute_exact_observability`. Direct acquisition
outputs and every nested exact-observability tensor are detached, independently
owned, and non-aliased with sources and each other. Raw invalid outputs may be
positive infinity as an audited simulator fact; qualified handoff and exact-J
outputs are finite.

## Live boundary

Only the hash-guarded test runner reads current private RayCaster
`_ray_starts_w`, `_ray_directions_w`, and the class mesh cache after an ordinary
sensor update. It uses the accepted two-environment static-plane scene, captures
the initial scan and one scan after one fixed public action/20 physics steps,
and performs zero actor inference, sampling, gradients, optimizer calls, or
training.

Valid plane normals must be unit world ±Z without sign modification. Each
environment therefore has exact translational rank one, zero `sigma_min`, an
unavailable condition number, and no unique weak direction because its
nullspace is two-dimensional. Open/upward rays remain explicit misses.

M7-002 runtime records remain finite-zero/invalid/`UNAVAILABLE`; no record or
source-manifest wiring is part of this contract.
