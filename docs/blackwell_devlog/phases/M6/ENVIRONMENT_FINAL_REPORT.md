# M6-002 Environment Final Report

## Order and history

```text
ORDER_ID: M6-002
PHASE: M6
BASE_COMMIT: 8295f6622354988ad9e00734126ebfa4a9eeaf73
FUNCTIONAL_COMMIT: 07746d6e95c218dfe5803c64c86088b976b4cb91
FUNCTIONAL_PARENT: 8295f6622354988ad9e00734126ebfa4a9eeaf73
BRANCH: blackwell-migration
```

The functional commit adds the DirectRLEnv integration, exact tests/comparator,
fixed rollout, and README boundary. The second ordered commit adds only these
six environment records.

## Scope completed

The accepted TASLAB articulation, MID360 RayCaster, M4 history, non-B0 M5
governor/control, Lee/rotor/wrench runtime, and M6 reward/termination/critic are
composed without changing accepted source. The environment enforces float32
normalized `[N,4]` actions, exact M4 policy and separate M6 critic branches,
same-post-step reward/done, direct M5 metrics, and full/selective vector reset.

## Information contract

Actor inputs remain only the accepted onboard MID360 history, validity/
reliability/age, body angular velocity, body gravity direction, commanded body
velocity, and previous issued body velocity. Root pose/state, actual velocity,
clearance/contact truth, critic, rewards, reasons, Lee state, wrenches, and
rotors remain outside policy/action/control. Privilege perturbations change the
critic while the policy branch stays byte-identical.

## Correctness gates

- environment configuration, public API, action, and schemas: PASS;
- actor leakage and source-dependency boundary: PASS;
- frames/units/ray order/direct M5 metric identity: PASS;
- same-state done/reward and accepted reason/timeout semantics: PASS;
- full/selective reset and malformed-input atomicity: PASS;
- two independent complete rollouts: PASS;
- 1,800-tensor/16,834,412-element exact comparison: PASS, max error `0.0`;
- comparator fail-closed mutations: PASS;
- accepted M2-M6 regressions and byte/diff integrity: PASS;
- frozen runtime/formal/external integrity: PASS;
- exact allowlist, history, and diff hygiene: PASS at functional evidence point.

## Package/runtime integrity

```text
Driver changed: NO
Conda changed: NO
PyTorch changed: NO
Isaac changed: NO
Frozen baseline changed: NO
```

The frozen Isaac Lab checkout's previously known kit-file modification was not
created or changed by this Order.

## Deferred findings

None requiring an in-scope implementation change.

## Recommendation and stop declaration

`READY_FOR_REVIEW` after the documentation commit and final clean-history
self-audit.

```text
NO NEXT-PHASE WORK WAS STARTED.
NO PPO, TRAINING, EVALUATION, M7, MERGE, OR PUSH WAS PERFORMED.
AWAITING COMMANDER REVIEW.
```
