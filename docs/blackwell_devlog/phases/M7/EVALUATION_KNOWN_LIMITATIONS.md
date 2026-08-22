# M7-001 Known Limitations

- The evidence is a fixed synthetic pure-tensor table, not a simulator rollout,
  hardware/bag replay, policy-quality evaluation, or flight-performance claim.
- Observability is accepted only as evaluation-provided values with EXACT,
  PROXY, or UNAVAILABLE classification. No exact Jacobian, proxy algorithm,
  normals, raycast, thresholds, or deployed safety input is implemented.
- World position and actual velocity are privileged evaluation inputs; this
  contract proves their separation from the actor but does not define their
  future runtime collection mechanism.
- Minimum clearance and observability require explicit validity from a future
  evaluator. Invalid samples remain unavailable and are never repaired.
- No B0-B8 baseline/ablation switch, scenario matrix, performance gate,
  production logger, dashboard, video, export, M8 work, or scale run exists.
