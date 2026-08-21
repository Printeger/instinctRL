# M4 Final Report

## Scope completed

M4 adds the pure-PyTorch actor-clean MID360 observation/history boundary and a
fail-closed exact-output audit. It consumes only the eight approved explicit
tensors, maintains four per-environment frames, and returns exactly
`lidar_grid [N,12,59,360]` and `state_vec [N,52]`.

## Provenance

| Authority | Path | SHA256 / identity |
| --- | --- | --- |
| Accepted M3 implementation | `isaaclab-training/instinctlab/sensors/mid360.py` | `a37a1c423d8818194f23023ba969f55025dc867d2280579633a699807a5aea44` |
| Legacy observation evidence | `isaac-training/training/scripts/instinctRL/observation.py` | `ee4a87e648a0293f1f64dd9784ac94f4a4e3a29281d1c4fc142db1eb5170f520` |
| Legacy audit evidence | `isaac-training/training/scripts/instinctRL/audit.py` | `7bf51e9d9b8f1c39ebe64094914b626b268ce916a5c028932937c22ae8d2b49a` |
| Legacy observation tests | `isaac-training/training/unit_test/test_instinctrl_observation.py` | `4a282aadfed4c5c2f08d1d612b1b0fc9417c8158443d2c451f563eb757c72fe8` |
| Legacy audit tests | `isaac-training/training/unit_test/test_instinctrl_actor_audit.py` | `032157c45c3c0175fd3ea0fbeffb4cdaeddc4efdf9add92d6a59dc66f5206f6a` |

Legacy four-frame/13-field history was behavioral evidence only. M3 controls
the `[59,360]` elevation-outer/azimuth-fastest mapping. Legacy full-state input,
range reconstruction, `[360,59]` layout, and weak key-only audit were not copied.

## Contract and correctness

- Range values and boolean mask enter without recomputation or reordering.
- Every invalid input beam must carry M3's exact serialized 40 m fill; valid
  beams may also be exactly 40 m because the M3 validity interval is closed.
- LiDAR channels are interleaved range/validity/reliability per oldest-to-newest frame.
- State frames are angular velocity `[0:3]`, gravity `[3:6]`, command `[6:9]`,
  previous issued command `[9:12]`, and sensor age `[12:13]`.
- Reliability is exactly `validity.float()*exp(-age/tau)`, clamped to `[0,1]`.
- Sensor and policy time regressions, negative age, malformed tensors, gravity
  norm violations, forbidden nested fields, and extra actor keys fail closed.
- Selective/full reset clears the corresponding history and time provenance;
  returned outputs do not alias internal buffers.
- Machine-readable output-slice provenance is asserted against code by tests.
- Revision 1 audits each frame as either completely zero-filled or a coherent
  populated M3 frame, including exact pair and unit-gravity domains.
- A persisted AST test directly binds M4's 360, 59, 21,240, 0.1 m, 40 m, and
  flattening semantics to the unchanged accepted M3 source without importing it.

## Changed-file ledger

| File group | Change | Reason | Semantic effect |
| --- | --- | --- | --- |
| `instinctlab/observations/{actor,audit,__init__}.py` | new strict builder/audit/export | exact M4 boundary | introduces only pure-tensor M4 observation behavior |
| three `tests/*m4*` files | semantic/negative/CUDA/comparator coverage | correctness gates | no runtime product behavior |
| `docs/blackwell_devlog/phases/M4/*` | contract, plan and evidence reports | ordered records | documentation only |
| `isaaclab-training/README.md` | M4 boundary summary | discoverability | documentation only |

## Results

The revision-1 22-test suite passes twice, M3's 8-test static suite passes twice, and both CUDA
processes produce byte-identical summaries. The full comparator proves exact
equality over 3,059,184 tensor elements with maximum error `0.0`. Complete raw
results and later final Git/integrity checks are indexed by `EVIDENCE_INDEX.md`.

## Integrity and exclusions

Driver changed: NO

Conda changed: NO

PyTorch changed: NO

Isaac changed: NO

Frozen baseline changed: NO

NO M5/CONTROLLER/GOVERNOR/ANCHOR/ICS/REWARD/PPO/TRAINING WORK STARTED.

Executor recommendation: `READY_FOR_REVIEW`.
