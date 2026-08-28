# M7-005 Exact Runtime Observability Plan

Order: `M7-005`
Base: `a8a88de1a12f24011646a9e43615d7c3e0b0f2df`
Branch: `blackwell-migration`

## Goal

Capture one detached, ordered raw evaluation ray source from the exact sensor update used by each pre-reset task snapshot. Resolve the frozen mesh and acquire normals only inside the M7 evaluation adapter, run the unchanged M7-003 exact core, and pass a normal-free sealed result to the M7-002 runtime collector. Preserve every actor, critic, control, reward, task, and `extras.pre_reset` contract.

## Steps

1. Add a raw `extras.evaluation_ray_source` sibling with owned world rays, accepted hits/validity, mesh path, maximum distance, global source step, and reset mask.
2. Reconstruct world rays from frozen RayCaster public local ray geometry and its public pose at the task update; never use its private world-ray buffers in environment or runner.
3. Confine guarded `RayCaster.meshes` resolution, Warp normal acquisition, commit/version/signature/source-hash checks, and minimum-range fail-closed behavior to `raycaster_observability.py`.
4. Distill exact outputs into independently owned tensors sealed by source/result SHA-256 identities; pass no rays, normals, mesh, sensor, environment, or actor privilege into the collector.
5. Map canonical observability to exact `sigma_min`, validity true, source `EXACT`, including exact zero.
6. Prove source/reset lifecycle, actor isolation, detailed-field validation, 22 complete-tree mutations, two fresh live processes, and zero-error deterministic comparison.

## Stop conditions

Stop on an unlisted/frozen change, inability to retain the terminal pre-reset source, actor/critic input change, required package/system mutation, or failure of exact source, coverage, finiteness, or determinism gates.
