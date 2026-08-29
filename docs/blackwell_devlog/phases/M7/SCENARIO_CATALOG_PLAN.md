# M7-013 Canonical Evaluation Scenario Catalog Plan

## Goal

Add a pure-Python, immutable and direct-import-only identity boundary for the
ten ordered Handbook evaluation scenario groups. Bind one exact scenario/seed
selection to an unchanged canonical baseline manifest while keeping every
scenario truthfully unavailable and exposing no execution path.

## Source locks

| Source | SHA-256 | Use |
| --- | --- | --- |
| `docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex` | `2ea4dd3da74ef8be67ae4b4e7a5906e6b4e13537fe9b86514517fd7f092dbe24` | normative scenario groups, metrics and actor privacy |
| `evaluation/baselines.py` | `725d6d1bf57a5c168d96deaa7cb7179382caa64b101ad0fd79172626dc8272b2` | platform/sensor and canonical manifest parser |
| `evaluation/baseline_runtime.py` | `c3f1e6f73f48bdcbf995c6f8178d5b93be2d6107af00d81b0ceac69c4a358e66` | accepted B0 collector remains unchanged |
| `evaluation/metrics.py` | `a37cf88a04d55ac4e32fee93f5504a6ff446f8804694a310013fb704438a3960` | test-only exact metric-name oracle |
| `evaluation/audit.py` | `03a6acb86c1aed86385b7ff3d265940791656ee8e38de07f242ac3128b35fd0e` | accepted audit remains unchanged |
| `evaluation/scenarios.py` | `a5d6f302a9ec48a319db83d32f8754bd5886ac73eeac79e9163e7ecc9b0262de` | new catalog and run-plan identity bytes |

## Implementation and evidence

1. Define exact enums and frozen non-subclassable evidence, spec, selection,
   catalog and run-plan records with immutable tuple fields.
2. Publish only independently immutable copies of the exact ten-row registry;
   reject aliases, normalization, duplicates, reordered rows and unknown IDs.
3. Encode the exact 8/2 domain split, sole optional dynamic row, all-NOT_READY
   evidence state, command-only actor forwarding, privileged generator fields,
   required log identity and exact accepted metric-name obligations.
4. Reparse and own the exact baseline manifest before requiring identical
   scenario ID/seed and locked platform/sensor; emit non-executable canonical
   run-plan JSON without changing baseline or scenario readiness.
5. Prove pure/static behavior twice, accepted M3-M7 regressions, exact locked
   bytes, two fresh serialization processes, complete decoded comparison and
   required mutation rejection. No CUDA or simulator is run.

## Stop conditions

Stop on any need to change locked/unlisted source, mark a catalog row READY,
invent provider/evidence, relabel an existing smoke, forward privileged state
to the actor, duplicate metric formulas, change a baseline manifest, or run a
scenario, CUDA/simulator, comparison campaign, training, M8 or scale work.
