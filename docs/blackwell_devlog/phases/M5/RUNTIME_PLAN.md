# M5 TASLAB Runtime-Control Plan

Order: `M5-002`, revision 1

Attempt: `M5-002-D0002`

Base: `685498a57f30746d3dd956d143fc416b2e9e1876`

## Goal and source authorities

Connect the accepted M5-001 world velocity output to the exact TASLAB Lee and
rotor contracts and Isaac Lab public wrench API. Authorities are the accepted
local TASLAB data/provenance, frozen MIT controller/rotor sources, approved
M3/M4/M5-001 code, and pinned Isaac Lab 2.3.2 API.

## Expected changes and sequence

1. Strict tensor Lee controller retaining source license and exact parameters.
2. Exact four-rotor mapping, dynamics, thrust/moment and selective reset.
3. Topology-validated Articulation wrench adapter with atomic preflight.
4. Fixed-action one-environment headless CUDA composition and full comparator.
5. Analytic/negative/source/regression/integrity qualification and exact two
   additive commits.

Revision 1 preserves the D0001 implementation and narrows only the obsolete
M5-001 directory-wide source scan to the exact actor-clean/pre-low-level set:
`governor.py`, `anchor.py`, `ics.py`, `command_adapter.py`, and `pipeline.py`.
The forbidden-token set is unchanged. A discriminating negative assertion
proves that adding the Lee module back into that set still detects
`root_state_w`; the actor boundary is scoped, not relaxed.

## Evidence and stop conditions

Preserve byte-pinned evidence tooling, every raw stream, complete traces and
failed development iterations. Stop for an external prerequisite, contract
contradiction, forbidden path, or required runtime/package/system/research
change; do not tune parameters, physics, scenarios, tolerances, or gates.

## Guaranteed exclusions

No actor network, reward, critic, PPO, collector, checkpoint, rollout API,
evaluation, training, M6 work, drag/downwash/randomization, asset edit,
package/runtime change, external repository change, merge, or push.
