# M6-002 Environment Final Report — Revision 1

## Preserved history

Original base is `8295f6622354988ad9e00734126ebfa4a9eeaf73`.
D0001 commits `07746d6e95c218dfe5803c64c86088b976b4cb91` and
`ca793159c6e1ae853b95af554bbb39d0b36cff79` remain unchanged. Revision 1 adds
one correction commit only; no history rewrite is used.

## F1-F4 closure

- F1: step 16 now proves public env0 timeout, terminal reward/reason/extras,
  automatic selected reset, post-reset policy/critic/counter, unaffected env1,
  and sixteen subsequent finite steps.
- F2: subset reset recursively clears only selected batch slices in control,
  Lee, wrench, reward, and task caches. Full reset alone invalidates whole
  cache objects.
- F3: both independent trees persist discriminating behavioral evidence for
  articulation, joints, sensors, rotor/wrench, M4 history/time, M5 pipeline/
  anchor/ICS, counters/actions/commands/flags/policy, and all output caches.
- F4: contact documentation now matches the implemented full vector norm.

## Information flow

Detached pre-reset extras preserve terminal provenance and are never actor or
control input. Returned reward/done/reason refer to the same terminal state;
returned exact-M4 policy and separate M6 critic refer to post-reset state.
Selected reward cache clearing cannot mutate returned reward because the public
return is an independent clone.

## Correctness results

- original API/schema/formulas/defaults: preserved;
- strengthened static suite: 8 PASS;
- required regression passes: 118 tests + 33 subtests, twice;
- M2 static: 10 PASS; accepted M5 runtime: 256 steps PASS;
- independent rollouts: 32 policy/640 rollout physics steps each;
- exact auto timeout/reset: one selected event per process;
- complete comparison: 4,423 tensors, 20,925,086 elements, max error `0.0`;
- actor leakage, integrity, prohibited-scope, and frozen-runtime gates: retained.

## Package/runtime integrity

Driver changed: NO
Conda changed: NO
PyTorch changed: NO
Isaac changed: NO
Frozen baseline changed: NO

## Recommendation and stop declaration

READY_FOR_REVIEW after committed-HEAD formal evidence and final clean audit.

NO NEXT-PHASE WORK WAS STARTED.
NO PPO, TRAINING, EVALUATION, M7, MERGE, PUSH, OR HISTORY REWRITE WAS PERFORMED.
AWAITING COMMANDER REVIEW.
