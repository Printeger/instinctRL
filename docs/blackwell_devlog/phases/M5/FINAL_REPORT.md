# M5-001 Final Report

## Scope completed

The exact M5-001 pure-tensor control core is implemented: strict normalized
governor decode and B0; per-environment measurement anchor; exact mounted M3
ray geometry and four-frame M4 ICS attenuation; strict controller-private
body-to-world rotation; and one atomic composed pipeline.

## Provenance and semantic impact

Accepted M3 fixes ray geometry/order/mount and accepted M4 fixes observation
shape, interleave, domains, and actor audit. The Handbook and four legacy
modules/tests provide formula and lifecycle evidence only. Experimental
vertical/null gates, privileged height handling, weak clamping, runtime and PPO
coupling were not migrated.

| Path group | Change | Semantic effect |
| --- | --- | --- |
| `instinctlab/control/governor.py` | strict configs/decode/B0/norm limit | bounded body governor command |
| `instinctlab/control/anchor.py` | hysteresis/capture/reset/masked Huber | actor-clean measurement anchor |
| `instinctlab/control/ics.py` | mounted rays/history attenuation | bounded body final command |
| `instinctlab/control/command_adapter.py` | scalar-first tensor rotation | final world-frame command only |
| `instinctlab/control/pipeline.py` | audit/preflight/composition/isolation | atomic exact dataflow |
| three M5 test programs | CPU/CUDA/full comparison | qualification only |
| README and six M5 records | contract/evidence truth | documentation only |

## Correctness gates

- Exact M4 actor surface and fail-closed audit: PASS.
- Governor formulas, bounds, smoothing and B0: PASS.
- Anchor lifecycle, reset precedence, robust mask and atomicity: PASS.
- Exact 21,240 mounted body rays and M3 order: PASS.
- ICS bounds, braking oracle, stale/invalid exclusion and optional controls: PASS.
- Quaternion rotation and complete pre-adapter non-interference: PASS.
- Two-process CUDA complete equality and finiteness: PASS.
- M4/M3 regressions, allowlist, integrity and diff hygiene: PASS.

## Runtime integrity

Driver, Conda packages, PyTorch, Isaac Sim/Lab, formal source, frozen external
repositories, and M1-M4 product/evidence were not changed.

## Stop declaration

```text
NO VELCONTROLLER/SIMULATOR/REWARD/PPO/TRAINING/M6 WORK STARTED.
NO NEXT-PHASE WORK WAS STARTED.
AWAITING COMMANDER REVIEW.
```
