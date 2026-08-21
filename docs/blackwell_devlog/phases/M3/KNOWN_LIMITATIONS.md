# M3 Known Limitations

- Revision-1 qualification uses one deterministic 84 m by 84 m tiled mesh whose topological surface is world z=0. Small triangles are required for the frozen float32 Warp intersection to meet the `1e-5` analytic range gate. This proves the ordered geometry and valid/miss contract, not performance in representative environments.
- The pattern is intentionally static. Dynamic scanning, noise, dropout, occlusion, timestamps, and reliability are outside M3.
- No actor observation, history, normalization, controller, reward, PPO, environment integration, or training behavior exists in this delivery.
- The frozen Isaac Lab checkout retains its baseline-known modification to `apps/isaaclab.python.headless.kit`; this order did not modify it.
- Startup/runtime warnings are preserved in coordination stderr. They did not alter the normalized outputs or correctness gates.
- The analytic maximum error is close to, but below, the fixed threshold (`7.62939453125e-6 m` versus `1e-5 m`); no tolerance was relaxed.
