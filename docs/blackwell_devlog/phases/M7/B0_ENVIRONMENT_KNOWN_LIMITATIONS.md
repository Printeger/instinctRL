# M7-011 Exact B0 TASLAB Environment Known Limitations

- This is one exact, bounded, headless B0 command-only execution mode. It is
  not a general environment API, collector, scenario matrix, metric suite,
  performance benchmark, evaluation campaign or training runtime.
- The mode accepts only exact finite contiguous Torch commands shaped `[N, 3]`
  with the environment's dtype and device. Other shapes, dtypes and devices are
  deliberately rejected.
- The validated configuration is fixed to two environments on `cuda:0`,
  decimation 20, no action noise, and a positive exact policy-step horizon.
- Reset after motion, early or repeated finalization, horizon overrun and
  rendering are intentionally rejected. Resume, checkpoint, retry, async and
  concurrent ownership are not defined.
- The bounded smoke uses fixed command vectors and validates migration
  correctness and deterministic records. It does not prove tracking quality,
  collision performance, reward, convergence or policy quality.
- Control-core, anchor and ICS behavior is intentionally disabled in B0. Their
  enabled behavior remains outside this mode.
- The public final receipt is privacy-preserving and contains hashes and
  counts, not trajectory tensors. Detailed state/control tensors remain only
  in coordination evidence and are not actor inputs.
- The exact cross-process comparator proves equality of accepted public
  records for the bounded smoke on the frozen host. It does not claim
  portability to a different GPU, driver, package stack or configuration.
- No collector, new scenario, metrics integration, MID360 change, reward, PPO,
  training, M8 or scale work was authorized or performed.
