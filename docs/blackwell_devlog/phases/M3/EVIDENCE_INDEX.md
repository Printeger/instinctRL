# M3 Evidence Index

Full raw evidence is stored outside the product repository at:

`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M3-001/M3-001-D0001/`

Key development records:

- `PLAN.md`: pre-implementation plan.
- `dev_runs/0001` and `0002`: preserved failed static subruns.
- `dev_runs/0003`: first passing static suite during repair.
- `dev_runs/0004` and `0005`: initial passing runtime qualification.
- `dev_runs/0006`: final syntax check and two byte-identical static suites.
- `dev_runs/0007/smoke.json`: final independent runtime output 1, SHA256 `81aee366a6d9534045b4c94fae97b84ab9a82801a08ddc801c2c55ac55cac749`.
- `dev_runs/0008/smoke.json`: final independent runtime output 2, same SHA256.
- `dev_runs/0008/comparison.json`: full comparison, SHA256 `366b6268bd0eb41022be3a36d1c4c0d9e73d56cc2eb406d585d9226589f41c7b`.
- `dev_runs/0010`: first commit command streams and exit record.
- `dev_runs/0011`: source/frozen/runtime integrity captures.
- `dev_runs/0012`: M2 regression streams.
- `release/v0001`: frozen exact-tooling final evidence release.
- `final_candidates/0001`: final fail-closed command ledger and self-audit.

The coordination `INDEX.md` and `COMMANDS.md` provide the complete command-to-stream mapping and exact invocations.

## Revision 1 evidence

Fresh correction evidence is stored at:

`/home/mint/rl_dev/instinctRL_coordination/attempts/development/M3-001/M3-001-D0002/`

- `PLAN.md`: correction plan created before implementation.
- `dev_runs/0001`: first passing 8-test transform/static suite.
- `dev_runs/0002`–`0007`: preserved analytic/geometry development failures.
- `dev_runs/0008/smoke.json`: first final corrected CUDA result, SHA256 `8ccb34e7e67fda11d3d2a89c69c11c4b47824f189522808930ce602521c86f78`.
- `dev_runs/0009/smoke.json`: second byte-identical corrected result, same SHA256.
- `dev_runs/0009/comparison.json`: unchanged full-tree comparator PASS.
- `dev_runs/0010`: final syntax and two byte-identical complete M3 static suites.
- `dev_runs/0011`: complete M2 static/runtime regression.
- `release/v0001`: frozen D0002 evidence tooling and declarative manifest.
- `final_candidates/0001`: passing final D0002 ledger and self-audit.

D0001 remains immutable and is not overwritten by these records.
