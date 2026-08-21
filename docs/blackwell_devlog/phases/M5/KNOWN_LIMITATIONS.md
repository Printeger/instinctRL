# M5 Known Limitations

- This Order qualifies only deterministic pure-PyTorch control logic; it does
  not instantiate or execute a TASLAB velocity controller or simulator.
- Governor action values are supplied as test fixtures. No actor network,
  weights, inference, distribution, or export exists yet.
- Optional range-rate filtering is a deterministic four-frame arithmetic mean,
  not a fitted dynamic model.
- ICS is scalar attenuation with a zero brake command, not a formal CBF, QP,
  CMDP, surface-normal method, or performance guarantee.
- Anchor and ICS values are correctness fixtures, not hover, tracking, collision,
  or policy-quality metrics.
- Determinism is proven across two independent processes on the frozen local
  CUDA runtime; no cross-hardware bitwise claim is made.
