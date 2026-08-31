# M7 Evaluation Availability Matrix Plan

## Goal

Represent the accepted nine-baseline by ten-scenario availability truth as one
pure, immutable, canonical and fail-closed control-plane artifact. The matrix
plans no run and contains no performance result.

## Sources of truth

- `instinctlab.evaluation.baselines` at Order base
  `39206507367e0fdd65ed99b6f3113dd5901d43bc`
- `instinctlab.evaluation.scenarios` at the same base
- Order M7-021 revision 0

## Bounded changes

Add the matrix module and pure tests, expose its ordered public API, add one
README section and create the six matrix records. Baseline/scenario registries,
runtime, fixtures, runners, actor/control/training code and prior evidence stay
unchanged.

## Method

For every baseline-major pair, place unavailable nonoptional capability names
and unavailable required-scenario reasons in `required_blockers`; place the B8
capability absence and optional dynamic-fixture absence in
`optional_blockers`. Required blockers select `NOT_READY`, optional-only
blockers select `SKIPPED`, and an empty pair selects `READY`.

## Qualification

Verify exact API and field order, all 90 cells, 7/74/9 totals, every blocker,
canonical JSON/hash, immutable round trip, malformed rejection, import
isolation, accepted M7 pure regressions, M3-M6 regressions, M2 static checks and
repository/evidence integrity.

## Stop conditions

Stop if exact totals cannot be derived, source evidence or privilege drifts,
runtime execution or a locked path becomes necessary, or a blocking diff or
integrity failure remains.
