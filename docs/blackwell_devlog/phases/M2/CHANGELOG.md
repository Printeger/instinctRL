# M2 Changelog

| File | Change | Reason | Order requirement | Semantic impact |
| --- | --- | --- | --- | --- |
| `isaaclab-training/README.md` | Document bounded M2 runtime surface | Prevent hover/control overclaim | Scope documentation | Documentation only |
| `isaaclab-training/instinctlab/assets/taslab_uav/articulation.py` | Add one `ArticulationCfg` | Spawn unchanged USD with public Isaac Lab APIs | Articulation contract | Enables gravity-only articulation; no control |
| `isaaclab-training/tests/test_m2_static.py` | Add standard-library static gates | Protect values, hashes, paths, and forbidden surface | Required static suite | Test only |
| `isaaclab-training/tests/run_m2_taslab_smoke.py` | Add normalized two-trajectory CUDA smoke | Prove spawn/reset/step/update viability | Runtime smoke contract | Test runtime only; zero applied control |
| `isaaclab-training/tests/compare_m2_smoke.py` | Add semantic process comparator | Prove independent-process determinism | Comparator tolerance `1e-5` | Test only |
| `docs/blackwell_devlog/phases/M2/*.md` | Add plan/results/evidence/limitations/report | Preserve review evidence | Required phase records | Documentation only |

The USD, YAML, Lee gains, M1 records, legacy runtime, and frozen repositories
were not changed. No mass, inertia, transform, collision, joint, or material
override was added.
