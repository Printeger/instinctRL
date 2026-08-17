# TASLAB migration plan (M1B candidate only)

## Status

This is a planning recommendation for a future `M1B` order. It is not an order.
No `isaaclab-training/` path is created, no asset is copied or converted, no
class/config/controller is implemented, and no runtime parity is claimed.

## Objective for a future M1B

Port the TASLAB_UAV platform into the reserved modern root under
`isaaclab-training/` without mutating `isaac-training/` legacy evidence and
without making the legacy tree masquerade as the new runtime.

## Candidate target layout

```text
isaaclab-training/
  assets/
    taslab_uav.usd
    taslab_uav.yaml
  robots/
    taslab_uav.py
    rotor_group.py or equivalent actuator adapter
  controllers/
    lee_controller_taslab_uav.yaml
    velocity_controller_adapter.py
  configs/
    drone.yaml
  tests/
    static_parity.py
    runtime_smoke.py
```

This layout is a candidate, not an implementation. It must not be created by
M1A.

## Dependency order

1. Resolve provenance/licensing prerequisites (binary USD, TASLAB header,
   AechPro/OmniDrones license).
2. Confirm USD crate prim/joint/rotor topology through read-only stage
   introspection.
3. Decide copy-vs-rewrite for text modules.
4. Add a modern asset/config adapter under `isaaclab-training/`.
5. Add static parity tests.
6. Only then run bounded runtime smoke/parity tests.

## Copy-vs-rewrite decisions

- Binary USD: **copy or provenance-tracked reference only**; do not hand-edit or
  convert during a first parity pass. Preserve original path, source commit,
  blob, SHA-256, and file type.
- `taslab_uav.yaml` and Lee gains: **copy with provenance**, then adapt only if
  the modern API requires an explicit config shape; do not silently retune.
- `taslab_uav.py`, rotor/multirotor/controller adapters: **new implementation
  under `isaaclab-training/`**, not in-place edits to `isaac-training/`.
- Do not combine refactor/package cleanup with migration until parity is
  established.

## Compatibility adapters

The future implementation should consider, without choosing unresolved
topology by assumption:

- OmniDrones `RobotBase.REGISTRY` → modern asset/config class registration.
- Legacy `RotorGroup` scalar parameterization → modern batched rotor/actuator
  state.
- Legacy `VelController` shape semantics → modern `LeePositionController.compute`
  contract.
- Legacy `omni.isaac.core` imports → modern `isaacsim` / Isaac Lab APIs.
- Command frame: keep body-frame velocity-governor method unchanged and put the
  body-to-world conversion at a named controller boundary, not inside actor
  observation.

## Static and runtime validation candidates

Static:

- exact provenance/commit/blob/SHA-256/size/type manifest
- allowlist diff
- symbol/config/asset-contract diff
- negative search proving no accidental legacy import or actor leakage

Runtime, only if separately authorized:

- USD loadability and prim/joint/rotor introspection
- mass/inertia/force/moment parity against frozen legacy source
- deterministic reset/step/update smoke
- controller command parity and frame/saturation checks
- actor-contract positive/negative schema assertions

## Mandatory vs alternatives

Mandatory:

- provenance and license resolution before any asset/class move
- explicit units/frames/rotor topology and controller mapping
- one bounded M1B order and one accepted M1A report before starting
- no mutation of frozen baselines or legacy source

Alternatives:

- exact modern class/module layout can vary while preserving the contract
- actuator abstraction choice (`ImplicitActuator`, `IdealPDActuator`, or a
  custom rotor adapter) is a design alternative, not a license to guess physics

## Licensing prerequisites

Before M1B can copy or convert the binary USD or AechPro-derived code, resolve:

- `taslab_uav.py` header `Copyright (c) 2024 TASLAB` vs vendored OmniDrones MIT
  license.
- ownership/license of `taslab_uav.usd` (currently unknown from static evidence).
- ownership/license of TASLAB-specific YAML/gains.
- whether AechPro fork and the local feasibility patches permit the intended
  migration/distribution.

## Acceptance recommendation for M1B

A future M1B report should pass only when:

1. every moved/created file has exact provenance;
2. the product diff stays within an explicit M1B allowlist;
3. static parity checks pass;
4. any runtime checks are deterministic and separately authorized;
5. no frozen baseline or legacy source is modified;
6. actor-contract boundary remains clean.
