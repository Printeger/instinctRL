# M7-019 unsafe-command-toward-wall B0 evidence index

Raw evidence root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-019/M7-019-D0003`

Every `commands/<label>/metadata.json` records exact argv, working directory,
selected environment, timestamps, exit code and SHA256/path for stdout and
stderr. `COMMAND_INDEX.json` and `EVIDENCE_INDEX.md` validate and enumerate
those records; large JSON/PT artifacts remain outside product Git.

## Primary fresh artifacts

- `formal/unsafe_{1,2}.json`: decoded qualification summaries including exact
  normalized direction, wall evidence, matrices, gates and hashes.
- `formal/unsafe_{1,2}.pt`: complete owned bundles compared bit-for-bit.
- `formal/unsafe_compare.json`: exact equality and 17 rejected mutations.
- `formal/{clutter,corridor,corner,flat,open}_{1,2}.{json,pt}`: ten fresh
  retained scenario results.
- `formal/{clutter,corridor,corner,flat,open}_compare.json`: unchanged
  20/16/16/13/11-mutation comparator results.
- `formal/learned_runtime.pt`, `formal/b0_environment.json` and
  `formal/b0_collector.json`: CUDA regression outputs.

## Stable unsafe-wall identities

| Object | SHA256 |
| --- | --- |
| catalog | `42ad16013d4d00deb2daba94a7f5f1764cafcede5f1665c66843da5f5d57a3df` |
| fixture source | `e587f36c06897e3410ffa225b40c3d267ad61d785240194abbd61750333b8329` |
| run plan | `4b4dfb2e7e1facb98c137e2369a1514bee0fe107d13939b725865d0e14936f45` |
| READY manifest | `963237fb3e1e39053bdd922fe4257f53a824c5aa5a908e24d1c7e21ff889f898` |
| evaluation record | `233d3baa3a0bc232d8614601616108c1918d49817f775f29566cf71f14d324ed` |
| qualification | `1608e1793da5fb087cca53b536c8da8b9d407301cbab4c238d5a8460cb83b704` |
| first complete bundle | `e7847955613983df68bdb0ab5b9731e391f4e4891ada10f6e6bc80d915916527` |

D0001, D0002 and M7-018 evidence identity listings are commands `038`, `039`
and `040`; their aggregate stdout identities match the prior accepted/blocked
captures. Runtime/package/frozen identity is command `037`, and the accepted
Kit identity is command `041`.
