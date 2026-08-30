# Corridor/Tunnel B0 Evidence Index

Raw root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-017/M7-017-D0001/`.

Every directory under `commands/` contains `metadata.json`, `stdout.log` and
`stderr.log`; metadata records exact argv, cwd, selected environment,
start/end time, exit code and stream hashes.

## Canonical identities

| Identity | SHA-256 |
| --- | --- |
| Catalog | `967091035d7d5f2291e72a65b192fd0b624bdbdd677556a161b27a29fb03df6f` |
| Corridor source | `dcf31feaffe36a7588f9cb2a1affed389fdc32c380e63042a45807b9c2966559` |
| Corridor manifest | `de140ceaae104ea2e4d2f958d41fcf3b7ce6c8f8cdbd1856f17e5d0a6aefbd00` |
| Corridor run plan | `0b894ffc3f955977dce0ec0235d05a0aa5648d165ca472fb68e47daf173d129b` |
| Corridor evaluation record | `9e8b77e9cb70e58911bd533934dc13dd8ba8386a089f636d817ec4dfcab68959` |
| Corridor qualification | `f1082252378757848b539c8af3df6fe154b1d87af93b6757be42346903defcff` |
| Corner v2 source | `ae6f54b2cb1d711418bc44db28784b0a1b92b762a68743b6b88abb26841b5aaf` |
| Flat-wall v3 source | `67a733c67981e543455dec6d12c84ce80f09d7b1d41eadbb851ab9525cda42a0` |
| Open-field v4 source | `3b51174eeac1997786fedc0e8dbc9711ffee14a1f3d33592618e10f79181af0e` |

## Formal artifacts

- `formal/corridor_1.{json,pt}` and `corridor_2.{json,pt}`: the two fresh
  corridor executions.
- `formal/corridor_comparison.json`: complete equality plus 16 mutation
  rejections.
- `formal/corner_v2_*`, `flat_v3_*`, `open_v4_*`: preserved accepted scenario
  executions and comparisons.
- `formal/learned_runtime.pt`, `b0_environment.{json,pt}` and
  `b0_collector.{json,pt}`: bounded CUDA regressions.

Large `.pt` bundles and raw logs remain in coordination storage and are not
committed to product Git. The coordination attempt-level generated index is
the authoritative file-by-file hash ledger.
