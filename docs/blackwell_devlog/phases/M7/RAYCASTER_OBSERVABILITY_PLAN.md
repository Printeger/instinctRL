# RayCaster Exact-Normal Qualification Plan

## Goal

Add one explicitly imported evaluation-runtime adapter around the frozen public
Isaac Lab/Warp `raycast_mesh` normal path. Re-cast the exact accepted live
MID360 rays against the same mesh and maximum distance, prove same-state hit,
distance, validity, and miss fidelity, and hand qualified normals to the
unchanged M7 exact-J core.

## Sources

- Approved M7-003 exact observability core and actor-isolation boundary.
- Frozen Isaac Lab 2.3.2 RayCaster, Warp operation, and kernel source.
- Accepted M3 MID360 geometry/range/validity and M6 two-environment static plane.
- Accepted M7-002 explicit runtime `UNAVAILABLE` observability state.

## Steps and tests

1. Hash and statically audit the frozen API, kernel, buffers, and mesh cache.
2. Implement strict owned-input validation and the official
   `return_distance=True`, `return_normal=True` call.
3. Validate complete return metadata, exact hits/mask, distance consistency,
   raw miss convention, unit normals, source immutability, and owned handoff.
4. Qualify focused positives/negatives and no-normal reverse dependencies.
5. Run two independent two-env/two-scan live processes, exact complete-tree
   comparison, and eleven fail-closed mutations.
6. Replay accepted M7 and relevant M3-M6 pure tests twice, M2 static, Git scope,
   and frozen/package/formal/external integrity.

## Exclusions and stops

No accepted sensor/environment/core/runtime-record change, proxy, plot,
ablation, policy inference/ranking, optimizer, training, M8, or scale run is
included. Stop only for an unusable frozen normal path, required frozen/system/
unlisted-path change, geometry/method contradiction, or external research
decision.
