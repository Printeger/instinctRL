# M7-018 cluttered-indoor B0 evidence index

Raw evidence root:
`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M7-018/M7-018-D0001`

Every `commands/<label>/metadata.json` records exact argv, cwd, selected
environment, timestamps, exit code and the SHA256/path of stdout and stderr.

## Primary artifacts

- `formal/clutter_1.json` and `formal/clutter_2.json`: decoded qualification
  summaries, geometry/inventory, four hit matrices, normals, gates and hashes.
- `formal/clutter_1.pt` and `formal/clutter_2.pt`: complete owned bundles used
  for bit-exact comparison.
- `formal/clutter_comparison.json`: exact equality plus 20 rejected mutations.
- `formal/{corridor,corner,flat,open}_{1,2}.{json,pt}`: eight fresh accepted
  scenario results.
- `formal/{corridor,corner,flat,open}_comparison.json`: 16/16/13/11 mutation
  comparator results.
- `formal/learned_runtime.pt`, `formal/b0_environment.json` and
  `formal/b0_collector.json`: frozen CUDA regression outputs.

## Stable clutter hashes

| Object | SHA256 |
| --- | --- |
| catalog | `b686f27fe198d45613ea738a29a3f566be45a2222d54e6b81ce71c46c563329a` |
| fixture source | `0e77c42e0fb38b1f1f9531c5c7e3c9f8e9432dede13e579ebfcaffb290db83a4` |
| run plan | `61199b906d09bd0b083017f09f1a89d3001fdfc0325a36d247018f99be9a8149` |
| READY manifest | `591847cdd04ce51ba0553cd2387b286bb26daf1027ed78c3b7f147c05e9007e5` |
| evaluation record | `4512b94b540c352ac81ad3f86d9469e1cc40cf3c210009e08e6cc69a36e814cc` |
| qualification | `0e1538188ef4bb07786eedef6e04a83885bc05aa124c41468c28a071907e901e` |
| complete bundle | `064931015d2a7491639406627936ad340bf9c8b9e4b81d9d51c34baded7e535a` |

M7-017 evidence identity was captured before work at `005`, captured again at
`040`, and compared byte-for-byte at `041`. Integrity output is `039`; the
accepted Kit identity proof is `042`.
