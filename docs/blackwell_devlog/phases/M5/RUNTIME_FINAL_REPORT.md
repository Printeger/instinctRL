# M5 TASLAB Runtime-Control Final Report

Order: `M5-002`, revision 1

## Scope completed

- Added a strict local MIT-notice-preserving TASLAB Lee velocity controller
  from accepted M5-001 world velocity plus controller-private root state.
- Added exact four-rotor command mapping, per-environment response, thrust,
  signed reaction moments, reset, and atomic preview/commit behavior.
- Added topology/device-strict public Isaac Lab local-wrench integration and
  explicit simulation writes.
- Composed accepted M3, M4, M5-001, Lee, rotor, and TASLAB Articulation in four
  bounded fixed-action CUDA scenarios.
- Reconciled the obsolete M5-001 directory scan without changing its forbidden
  tokens or any accepted M5-001 formula/default.

## Changed-file ledger

| File/group | Change | Reason | Semantic effect |
| --- | --- | --- | --- |
| `assets/taslab_uav/articulation.py` | named body/rotor topology constants | adapter preflight | no spawn/physics retune |
| `control/__init__.py` | runtime exports | stable project surface | exposes low-level APIs |
| `control/lee_velocity_controller.py` | strict TASLAB Lee equations | world velocity to rotor coordinates | controller-private stabilization |
| `control/rotor_actuation.py` | exact rotor response | physical actuator boundary | bounded thrust/moment |
| `control/runtime_adapter.py` | public local-wrench buffers | Isaac Lab integration | force/torque application |
| `test_m5_control_core.py` | exact five-module actor scan | revision-1 correction | preserves actor prohibition |
| three M5-002 test programs | analytic/negative/runtime/comparison | correctness evidence | none outside tests |
| README and six runtime records | contract/results/provenance/limits | traceability | documentation only |

## Correctness result

Controller/rotor constants, mixer/order/signs, frames, local +Z application,
finite validation, topology/device preflight, reset and failed-call atomicity
pass analytic and runtime gates. Actor-visible keys and M4 schema are unchanged;
root state is consumed only after M5-001 `v_final_w`. Two independent 256-step
CUDA processes compare 96 tensors and 4,778,624 elements with maximum error 0.

Driver, Conda packages, PyTorch, Isaac Sim/Lab, formal source, frozen external
repositories, accepted M3/M4/M5-001 implementation, YAML, USD, and Git
configuration were not modified. No merge or push occurred.

## Stop declaration

```text
NO ACTOR/REWARD/PPO/TRAINING/M6 OR LATER WORK STARTED.
AWAITING COMMANDER REVIEW.
```
