# M7-009 B0 TASLAB Controller Handoff Known Limitations

- This qualifies a stateless evaluation-side B0-to-Lee-to-wrench handoff. It
  does not integrate `TaslabDirectEnv`, a runtime collector, or a persistent
  baseline lifecycle.
- The smoke uses one TASLAB articulation, four constant-command/reset fixtures,
  64 steps per fixture, and 256 steps per process. It is not a scenario matrix
  or performance evaluation.
- MID360, learned policy inference, control pipeline, anchor, ICS, task,
  reward, PPO, training, collision, and clearance behavior are not exercised.
- B1-B8 remain unchanged and non-executable. This Order adds no implementation
  or readiness claim for them.
- Exact equality is established for two independent processes on the frozen
  Blackwell runtime. No cross-machine or cross-version determinism claim is
  made.
- Tracking, hover, displacement, reward, and policy convergence are neither
  measured nor claimed.
