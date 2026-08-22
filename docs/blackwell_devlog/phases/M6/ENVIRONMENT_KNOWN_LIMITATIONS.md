# M6-002 Environment Known Limitations

- The environment qualifies fixed zero-command/neutral-governor fixtures and a
  deterministic static plane only. It makes no flight-quality, tracking,
  collision-rate, throughput, or convergence claim.
- Exact clearance is the minimum qualified MID360 RayCaster range, not a
  continuous signed-distance field. It follows the accepted finite range/fill
  contract and is privilege-only.
- Collision uses public normal contact force with the fixed `1.0 N` threshold;
  tangential force is not part of this label.
- The accepted runtime adapter uses Isaac Lab's public
  `set_external_force_and_torque`, which emits a frozen-runtime deprecation
  warning but remains functional at the qualified commit.
- The frozen Isaac Lab checkout retains the known pre-existing modification to
  `apps/isaaclab.python.headless.kit`; this attempt did not touch it.
- Reset/replay determinism is proven for two environments and 32 policy steps,
  not for large vector counts or long horizons.
- No actor/critic network, PPO/GAE, optimizer, collector, checkpoint, learning,
  evaluation, curriculum, randomization, M7 work, 128k, 1M, or scale training
  was implemented or run.
