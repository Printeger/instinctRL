# M3 Known Limitations

- Qualification uses one deterministic infinite ground plane. It proves the ordered geometry and valid/miss contract, not performance in representative environments.
- The pattern is intentionally static. Dynamic scanning, noise, dropout, occlusion, timestamps, and reliability are outside M3.
- No actor observation, history, normalization, controller, reward, PPO, environment integration, or training behavior exists in this delivery.
- The frozen Isaac Lab checkout retains its baseline-known modification to `apps/isaaclab.python.headless.kit`; this order did not modify it.
- Startup/runtime warnings are preserved in coordination stderr. They did not alter the normalized outputs or correctness gates.
