# M8 release-candidate manifest report

## Revision 1 repair outcome

M8-001 Revision 1 corrects the sole D0001 contract defect: the frozen runtime
driver identity is now exactly `580.173.02`, matching accepted evidence. The
obsolete `580.173.02-open` value is absent from the repaired module, focused
expectation and canonical JSON. No schema, field order, enum, accepted chain,
approval hash, other runtime value, actor/privilege boundary, M7 availability
value, qualification gate, release status or authorization boolean changes.

D0001's two nominal commits and coordination evidence remain immutable.
Revision 1 adds one five-path repair commit and fresh D0002 evidence. The final
coordination Executor report binds the resulting three-commit candidate HEAD,
old/new canonical hashes, exact repair ledger and final closure.

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
