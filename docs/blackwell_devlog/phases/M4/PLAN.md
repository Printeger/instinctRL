# M4 Observation Boundary Plan

Order: `M4-001`  
Attempt: `M4-001-D0001`  
Base: `99ec3b0099b887a31962b2a5c862e3e73efc6872`

## Goal

Add the pure-tensor actor observation/history boundary and prove that its exact
two outputs derive only from Order-approved MID360, timing, IMU, and body-command inputs.

## Scope and sequence

1. Trace approved M3 and read-only legacy/Handbook authorities.
2. Implement strict tensor validation, four-frame history, age and reliability.
3. Implement recursive exact-schema actor-output audit.
4. Test mapping, ordering, timing, rollover, reset, aliasing, malformed inputs,
   forbidden-field rejection, CUDA determinism, M3 regression, and repository integrity.
5. Commit implementation, then documentation/evidence, using exactly the two ordered commits.

## Guaranteed exclusions

No actor network, controller, governor, anchor, ICS, reward, critic, PPO,
collector, rollout, evaluation, training, M3 modification, simulator launch,
package mutation, merge, or push.
