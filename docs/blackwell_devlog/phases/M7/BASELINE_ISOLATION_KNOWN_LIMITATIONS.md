# M7-006 Baseline Isolation Known Limitations

- This Order implements identity, capability, and manifest validation only. It
  does not implement or execute B1-B8 adapters or an evaluator matrix.
- Only the accepted M5 B0 direct-velocity function has an accepted provider and
  source hash. B1-B8 are intentionally `NOT_READY` until later Orders add and
  prove their distinct capability bindings.
- B8 remains optional. Its absent capability is represented only by an explicit
  `SKIPPED` manifest with a reason; no B8 result or comparative claim exists.
- The per-run product commit is supplied by the caller and validated as exact
  lowercase 40-hex provenance. The registry does not read moving `HEAD` or a
  coordination candidate ID.
- Updating accepted B0 source bytes, any row identity/schema/toggle, or a later
  adapter capability requires a separately authorized registry revision and
  new evidence; importability alone never changes readiness.
- No scenario, CUDA/simulator runtime, baseline behavior, policy performance,
  plot, correlation, hardware proxy, real robot, training, M8, 128k, 1M, or
  scale claim was tested or made.
