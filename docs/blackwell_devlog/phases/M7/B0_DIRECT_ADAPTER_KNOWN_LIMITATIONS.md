# M7-007 B0 Direct Adapter Known Limitations

- The adapter is a pure execution primitive and is not wired into
  `TaslabDirectEnv`, an evaluation collector, the low-level controller, or a
  scenario.
- Controller-private quaternion is caller-supplied and validated; this Order
  does not qualify its acquisition or timing in an integrated runtime.
- CPU float32/float64 pure-tensor behavior is qualified. Device preservation is
  statically/unit tested, but no CUDA computation is authorized or claimed.
- B0 READY denotes an exact source/capability contract, not a completed
  baseline run or a policy-performance result.
- B1-B8 remain not ready and are neither implemented nor executed here.
- The historical M6 whole-control-directory immutability assertion is
  inapplicable after an explicitly authorized M7 control addition; all
  remaining M3-M6 behavior regressions pass and exact source/allowlist checks
  replace that assertion for this increment.
