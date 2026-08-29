# M7-010 Exact B0 Run Lifecycle Known Limitations

- The lifecycle is synchronous and single-owner. Threading, async use,
  concurrent calls, resume, checkpoint, abort, retry and reset are not defined.
- The session has no environment, collector, scenario, metric, performance,
  simulator or training integration. It executes only the already accepted B0
  controller handoff.
- Only exact CPU float32/float64 tensors accepted by the handoff are framed.
  CUDA tensors and other dtypes are deliberately outside this Order.
- The receipt proves ordered bytes and manifest identities, not flight quality,
  convergence, performance, or suitability of B1-B8.
- Public receipt construction validates schema, manifest relationship, step
  count and digest syntax. Trusted trajectory provenance is created by the
  factory-owned session; a bare syntactically valid digest string cannot by
  itself prove execution.
- Formal validation is pure CPU. No simulation, controller runtime smoke,
  MID360, PPO, training, evaluation campaign, M8 or scale work was authorized
  or performed.
