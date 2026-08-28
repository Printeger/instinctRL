# Exact Observability Core Plan

## Goal

Add one pure evaluation-only computation that maps caller-supplied world-frame
surface normals and validity masks to exact translational range-Jacobian
observability results. Preserve every accepted actor, control, environment,
training, metric, audit, and runtime boundary.

## Source of truth

- The locked Handbook's Observability Logger Implementation Plan permits exact
  offline `J` from surface normals only in evaluation/analysis.
- The architecture lock forbids surface normals and evaluation privilege from
  entering the deployed actor.
- M7-001 defines the evaluation source labels and actor-isolation contract.
- M7-002 deliberately emits only `UNAVAILABLE` observability at runtime.

## Implementation

1. Validate exact owned tensors and immutable configuration without mutation,
   normalization, repair, or silent exclusion.
2. Form `J_i = -n_i^T`, compute its singular spectrum, apply the prescribed
   scale-aware numerical tolerance, and make unavailable states explicit.
3. Return detached independently owned tensors on the input device.
4. Qualify hand-computed geometries, tolerance and storage negatives,
   invariants, recursive imports/names, and real actor discrimination.
5. Replay accepted M7 and relevant M3-M6 pure contracts twice plus M2 static.

## Guaranteed exclusions

No RayCaster/Isaac Lab import, surface-normal acquisition, runtime collector
wiring, finite-difference or hardware proxy, plots, scenarios, ablations,
training, policy ranking, M8 work, scale run, merge, or push is included.

## Stop conditions

Stop if exact mathematics conflicts with the locked Handbook, an accepted
contract or unlisted product path must change, the qualified baseline is
unavailable, or an external research decision is required.
