# Exact Observability Known Limitations

- The core consumes already available exact simulator/offline surface normals;
  it does not acquire RayCaster hits or validate their physical provenance.
- It is not connected to the M7 runtime collector. Existing runtime
  observability remains `UNAVAILABLE` until a separately authorized integration
  supplies and audits exact normals.
- No finite-difference estimator, reliability weighting, staleness model,
  hardware/MID360 proxy, commanded probe, or drift correlation is provided.
- Numerical comparisons follow the input dtype and device. No cross-device
  bitwise identity claim is made.
- The configured unit-normal tolerance validates supplied normals; the core
  does not normalize them or quantify normal-estimation uncertainty.
- No scenario runner, threshold, plot, dashboard, ablation, policy comparison,
  learning-quality claim, or performance qualification is included.
