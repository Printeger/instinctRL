# M8 release-candidate manifest plan

Order: `M8-001` revision 0, attempt `M8-001-D0001`.

## Goal

Create one pure immutable `m8.blackwell-release-candidate.v1` contract. It
binds the accepted M1A-M7 chain, frozen Blackwell identities, the hard actor
boundary, the accepted M7 availability summary and the qualification gates
still open. It does not execute qualification or approve release activity.

## Scope

The implementation is standard-library-only and isolated in
`instinctlab.qualification`. It takes one caller-supplied exact lowercase
40-hex candidate commit and produces canonical ASCII JSON plus a SHA-256 over
the complete payload. Accepted identities are constants derived from explicit
approval records, never discovered from a moving checkout.

## Steps and gates

1. Bind exact accepted approval and product identities for M1A through M7.
2. Define frozen non-subclassable records and exact enums.
3. Implement deterministic builder, encoder, decoder and validator.
4. Reject malformed structure, aliases, false claims and resigned semantic
   mutations.
5. Prove standard-library-only imports and absence of reverse imports.
6. Run focused and accepted pure/static regressions with frozen identities.
7. Deliver exactly two nominal commits across the ten Order-allowed paths.

## Explicit non-goals

No simulator, scenario, actor audit, adapter, metric, threshold, checkpoint,
PPO, training, scale, performance, release approval, merge or push action is
part of this record.
