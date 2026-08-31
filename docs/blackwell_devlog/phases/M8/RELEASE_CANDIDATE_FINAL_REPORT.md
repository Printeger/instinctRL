# M8 release-candidate manifest report

## Outcome

M8-001 adds a pure immutable release-candidate manifest and qualification gate
inventory. The contract binds the exact accepted M1A-M7 chain, frozen
Blackwell/runtime/source identities, hard actor boundary and M7 90/7/74/9
availability truth. It is standard-library-only and has no reverse runtime
import.

The release status is `MANIFESTED_NOT_QUALIFIED`. Qualification execution,
release approval, scale training, merge and push are false. The manifest takes
an exact caller-supplied candidate commit without changing the accepted M7
input commit.

## Product scope

Nominal commit 1 contains the README, qualification package/module and pure
test. Nominal commit 2 contains exactly these six M8 records. The final
coordination Executor report records the resulting self-referentially external
Git commit identities and exact base-to-HEAD ten-path ledger.

## Correctness summary

- Exact accepted chain and approval identities: PASS.
- Frozen runtime/platform/sensor/source identity contract: PASS.
- Actor and privilege boundary classification: PASS.
- Exact M7 matrix and deferred-capability classification: PASS.
- Canonical encode/decode/hash and resigned-mutation rejection: PASS.
- Public API, immutable record and import isolation: PASS.
- Focused and accepted pure/static regressions: PASS.
- Frozen source/package and M7 evidence identity baseline: PASS.

## Boundary

This report makes no release, performance or safety claim. No simulator,
scenario, actor audit execution, adapter implementation, checkpoint creation,
PPO update, training, scale, merge or push work occurred. M8-002 and all
qualification execution remain unauthorized.
