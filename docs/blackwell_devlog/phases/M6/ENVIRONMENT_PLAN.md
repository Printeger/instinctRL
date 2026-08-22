# M6-002 Environment Plan

## Understanding of goal

Compose the accepted M2 through M6-001 blocks behind one deterministic Isaac
Lab `DirectRLEnv` boundary and qualify fixed action step/reset behavior. This is
an integration boundary for later work, not a learning implementation.

## Source-of-truth files

- Order `M6-002`, revision 0, attempt `M6-002-D0001`.
- Accepted product base `8295f6622354988ad9e00734126ebfa4a9eeaf73`.
- Frozen Isaac Lab commit `37ddf626871758333d6ed89cf64ad702aef127d0`.
- Accepted M2-M6-001 source, tests, contracts, and runtime smokes.
- Active architecture lock, baseline pointers, formal `CONTEXT.md`, and Handbook.

## Files expected to change

Exactly the twelve product paths in the M6-002 allowlist and the fresh
coordination state/report/evidence paths for `M6-002-D0001`.

## Files guaranteed not to change

Every accepted M1-M6-001 implementation, test, evidence record, legacy path,
formal source, frozen repository/runtime, system/package path, Git
configuration, and every unlisted product path.

## Implementation steps

1. Prove branch/base/clean-state and prior acceptance gates.
2. Record frozen public API provenance.
3. Compose TASLAB, MID360, M4, M5, and M6 without editing accepted modules.
4. Add fail-closed environment, reset, fixed rollout, and exact comparator tests.
5. Run static/unit regressions before bounded headless CUDA execution.
6. Persist formal streams, exits, manifests, hashes, and self-audit evidence.
7. Produce exactly two ordered commits and stop for Commander review.

## Tests

Static/API/schema/leakage and negative-action checks; same-state done/reward and
direct-metric checks; full/selective reset checks; two independent two-env,
32-policy-step CUDA rollouts; exact complete-tree comparison; accepted M2-M6
regressions; source/runtime/external integrity; allowlist/history/diff hygiene.

## Evidence

The immutable attempt tree is
`attempts/development/M6-002/M6-002-D0001/`. Its command ledger records argv,
cwd, environment, UTC timestamps, executable hashes, separate streams,
expected/actual exits, and all output hashes.

## Risks

Frozen `DirectRLEnv` lifecycle compatibility, cloned articulation/contact
topology, RayCaster reset timing, exact body/world transformations, and
cross-process physics determinism.

## Stop conditions

Only the Order's frozen API, unlisted path, actor/research, runtime/system, or
accepted-base incompatibilities block. Ordinary in-scope defects remain in the
attempt and cannot authorize scope expansion.
