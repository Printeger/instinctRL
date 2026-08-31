# M8 release-candidate evidence index

Attempt root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M8-001/M8-001-D0001/`

Machine evidence is in `COMMAND_INDEX.json` and `COMMAND_LEDGER.json`; raw
streams and metadata are under `commands/<label>/`. `CLOSURE_RESULT.json`
defines the finite coverage boundary and proves directory/ledger/index/raw
stream closure. `EVIDENCE_INDEX.md` is the complete human command table.

## Gate map

- Preflight and source identity: `001`-`002`, `004`, `018`-`019`.
- Focused public/canonical/negative/import tests: `003`, `005`, `011`-`012`.
- First nominal commit and hygiene: `006`-`010`.
- Accepted M7 regressions: `013`-`014`.
- Relevant M3-M6 regressions: `015`-`016`.
- M2 static regression: `017`.
- Later labels in the same immutable attempt index cover the canonical final
  candidate probe, second nominal commit, exact history/path ledger, final
  integrity, clean status and delivery hashes.

All failed attempts remain visible. No index claims recursive self-coverage.
