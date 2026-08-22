# M6 Pure-Tensor Task Known Limitations

- This Order defines task tensors only. It does not provide actor/critic
  networks, PPO/GAE, optimization, rollout, checkpoint, environment integration,
  evaluation, training readiness, convergence or policy performance.
- Reward weights are migration defaults from active/compatible evidence; their
  relative learning performance is not tuned or accepted here.
- Binary ordinary-compliance gating uses `beta >= 0.999`; the threshold is
  explicit configuration, not an inferred safety guarantee.
- Exact clearance and collision require future simulator wiring, which is not
  authorized by this Order. Missing inputs fail rather than fabricate safety.
- World height is used only for termination here; no height reward is added.
- Critic root state contains world linear velocity while the schema also names
  actual body linear velocity. This deliberate frame-explicit redundancy is not
  a claim that either field may enter the actor.
- The pure CUDA determinism result applies to the frozen local runtime; no
  simulator, cross-hardware or learning determinism is claimed.
