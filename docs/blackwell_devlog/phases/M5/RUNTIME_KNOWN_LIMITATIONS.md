# M5 TASLAB Runtime-Control Known Limitations

- This is fixed-action migration-correctness qualification. It contains no
  actor network, learned weights, reward, critic, PPO, collector, checkpoint,
  rollout API, evaluation, training, or M6 integration.
- Current-position/current-yaw velocity mode is retained exactly from the
  reviewed `yaw_control=False` behavior; position and yaw are not independently
  commanded by this interface.
- The rotor recurrence is the reviewed per-step `0.43` response. `dt=0.005 s`
  is fixed and recorded, but no continuous-time reinterpretation is claimed.
- Translational drag, downwash, rotor noise, and unresolved legacy drag
  semantics are intentionally excluded rather than guessed.
- The current public Isaac Lab 2.3.2
  `set_external_force_and_torque` API emits a future-deprecation warning. It is
  valid and functional in the frozen baseline; no future API compatibility is
  claimed.
- Zero-command drop and command tracking are short-smoke diagnostics, not
  hover-performance, stability, or flight-envelope guarantees.
- Determinism is proven for two independent processes on the frozen local
  RTX 5070/R580 runtime within `1e-5`; no cross-hardware claim is made.
