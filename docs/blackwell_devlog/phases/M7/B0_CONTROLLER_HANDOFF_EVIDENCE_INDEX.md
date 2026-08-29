# M7-009 Revision 1 B0 Controller Handoff Evidence Index

## Accepted D0002 evidence

Attempt: `M7-009-D0002`

Formal candidate: `formal_candidates/0001`

Coordination root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-009/M7-009-D0002`

| Artifact | SHA-256 | Purpose |
| --- | --- | --- |
| `formal_candidates/0001/SUMMARY.json` | `70012dd921736c56ba82f7053d73c095e8750dc016e53219d5ca04c2769f6d57` | accepted aggregate result and hashes |
| `formal_candidates/0001/COMMAND_LEDGER.json` | `c00db67900a44ef2b2ccae1a2eef3441fa654918b9928f653d2a056de85045ab` | exact 28-command ledger |
| `formal_candidates/0001/REQUIREMENT_MAP.json` | `f1f360693f183e85b187c1a41511bba850758ce170cda4333809e45c43116619` | F1/F2 and regression mapping |
| `formal_candidates/0001/simulator_records/oracle_validation.json` | `57e1a1c9cf1cc6addbc87c96a3e17403241fe5dc138bdba5edd8b3ab38c73db7` | exact zero-oracle/two-sided-thrust validation |
| `formal_candidates/0001/simulator_records/exact_comparison.json` | `cc0aa31f8cbfb02e53f3e002a0548948ad18d46facfcd2138cd64e4af3deda58` | exact two-process complete-tree comparison |

The candidate summary binds both complete `.pt` bundles, runner summaries,
all raw stdout/stderr, and every persisted program by SHA-256. Large tensor
records remain outside product Git.

`D0001_SNAPSHOT.json` binds all 102 files under the prior attempt/evidence
roots. Formal ledger 005 re-verifies that complete snapshot before D0002 tests.
D0001 summary remains
`4f7bce4ff3612489c4f687643269dcd0af9ecf194777bc73798c7e1f765ceb2f`.

The runner source is
`2ef6d903bcc1590a3658e62bf2d836d275797e2a3ac2016791df371c400793fd`;
the unchanged comparator remains
`f56070fa95902063f75b8840054f1e302bc189a87c3b76b7f3fd1669080fa75c`.
