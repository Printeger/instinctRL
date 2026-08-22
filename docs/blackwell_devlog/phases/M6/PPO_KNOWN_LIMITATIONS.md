# M6-003 PPO Tensor-Core Known Limitations

- Correctness is proven on one persisted four-sample, two-step trajectory and
  exactly two PPO update calls per independent CUDA process. This is not a
  rollout horizon, learning run, or convergence claim.
- The actor CNN architecture is bounded and contract-qualified, not tuned for
  task performance or deployment latency.
- GAE consumes explicitly supplied values and masks; no collector or
  environment adapter is provided by this Order. Simultaneous termination and
  timeout is qualified with termination precedence; explicit terminal-value
  bootstrap remains required for pure timeouts only.
- Checkpoint atomicity is local-filesystem atomic replacement. Cross-filesystem
  transport, distributed writers, export, and deployment packaging are not
  qualified.
- The accepted M6-002 environment is run once only as a regression. It is not
  connected to the PPO package and supplies no learning samples.
- No recurrent actor, cost critic, CMDP/PPO-Lagrangian, replay, collector,
  curriculum, randomization, evaluation, logging dashboard, ablation, M7,
  128k, 1M, scale run, or policy-quality evidence exists in this delivery.
