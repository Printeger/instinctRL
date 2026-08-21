# M1B Changelog

| File | Change | Reason | Order requirement | Semantic impact |
| --- | --- | --- | --- | --- |
| `isaaclab-training/README.md` | Defined the bounded new migration root | Separate new and legacy runtimes | M1B skeleton | Documentation only |
| `instinctlab/__init__.py`, `instinctlab/assets/__init__.py`, `taslab_uav/__init__.py` | Added minimal package boundaries | Importable independent data contract | M1B skeleton | Exposes only `TASLAB_UAV` |
| `taslab_uav/taslab_uav.usd` | Copied exact base Git blob | Preserve authorized platform asset | Byte identity | No byte or semantic change |
| `taslab_uav/taslab_uav.yaml` | Copied exact base Git blob | Preserve active vehicle values | Byte identity | No byte or semantic change |
| `taslab_uav/lee_controller_taslab_uav.yaml` | Copied exact base Git blob | Preserve active Lee gains | Byte identity | No byte or semantic change |
| `taslab_uav/PROVENANCE.json` | Bound targets to source objects and authorization | Machine-readable provenance | Provenance gate | No runtime effect |
| `taslab_uav/NOTICE.md` | Recorded bounded reuse basis | Preserve authorization context without broader rights claim | Notice gate | No runtime effect |
| `taslab_uav/config.py` | Declared exact active YAML values, units, list order, proven rotor prim paths, and unresolved runtime semantics | Independent standard-library contract | Config parity | Data only; no runtime implementation |
| `tests/test_m1b_static.py` | Added ten static checks | Persist byte/value/path/boundary qualification | Static gates | Test only |
| `tests/inspect_taslab_usd.py` | Added deterministic read-only stage inventory | Prove source/target and four-rotor topology | USD gate | Query only; no simulation activity |
| `docs/blackwell_devlog/phases/M1B/*` | Added plan, results, evidence map, limitations, and report | Reviewable phase evidence | Delivery record | Documentation only |

