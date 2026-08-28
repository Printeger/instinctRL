# RayCaster Exact-Normal Known Limitations

- Current exact acquisition depends on private RayCaster world-ray buffers and
  class mesh cache, read only inside the version/hash-guarded qualification
  runner. A future public integration boundary is required before runtime use.
- The adapter is qualified only against the accepted static-plane scene and
  single frozen-mesh path. It does not qualify dynamic or multiple meshes.
- Normal orientation sign is intentionally not canonicalized; exact-J results
  are sign invariant. Only unit world ±Z is asserted for the plane.
- Distance agreement uses a declared `1e-5 m` float32 tolerance; hit positions
  and complete cross-process trees remain tensor-exact.
- The acquisition is not wired to M7-002 runtime records, which remain
  `UNAVAILABLE`.
- No finite-difference/hardware proxy, reliability/staleness model, drift
  correlation, scenario, plot, threshold, ablation, or policy result exists.
