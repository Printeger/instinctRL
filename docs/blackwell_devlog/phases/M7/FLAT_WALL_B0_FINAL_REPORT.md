# M7-015-D0002 Flat-Wall Identity Rework Report

## Outcome

Commander finding F1 is closed at functional commit
`cbd51657bdecfb76408bc5f4531f77635449b7e8`, a direct child of reviewed D0001
head `cc2128cb65b29c74fd1b61ff9c3c13762501bc0a`. The sole fixture ID is now
exactly `m7-015-flat-wall-b0-two-env-eight-step-v1`, and an independent-literal
regression checks the product constant, fixture evaluation provenance,
qualification fixture section and embedded B0 provenance.

Exactly three functional paths changed. The only source semantic change is the
fixture literal and dependent committed flat source/catalog identity. The
catalog READY/executable set remains exactly `(flat_wall,7015)` and
`(open_field,7014)`; the other eight rows remain NOT_READY.

## Exact evidence

- Flat source / catalog / qualification:
  `900a9147cd74cd1750d831afbba5319acef62c30f748e9599053266d83c87e6e` /
  `0ff2a806ffa5452c784b60dd7a65d7f9ba4b23a2d669af8f990274cc37612769` /
  `8de863aaeb2b369fe9c0038fc33f20bca234851b600ef173143c708653f3163c`.
- Flat plan / manifest / receipt / record:
  `19017a0c378cf43af8a7b239c1f9c7290809d4d434505ca31ffc92c664bb8218` /
  `5a48153a4621e3ddeb67189e12b588ac68c5d8b5913de179c7ce4d5bd28e83c3` /
  `3f3176f4f6a91a8c7dd0a66bb6ba7a2694bbc6bf3892f222e337cac2888eed0f` /
  `2948ddc64371832eb472492c323c080ae38d5f7e11b7554df441fa34c2413f9c`.
- Open-field v2 plan / manifest / receipt / record / qualification:
  `344e1311f7961c52f5b02e1048d4073d4f3429cfff1e09411f04b29fb308aec0` /
  `10b63205f9ed7d78dc053d5224789194466c146a67396b5f0ca38ee6efaac67c` /
  `4c1ea869f74e729a42c4ea7cf1db9bbaf09f4688af2f8f43490a0073852e0f77` /
  `13d2802319d10ab3fe2b86d9a3a0caf5f41008b7efddd105975d014b3142db48` /
  `afa353106b02d9a2e243383d343f9a6477540cce17c1f58eaaa0cf9693e0858f`.
- Pure/static tests: 2,076 passed with two disclosed accepted M6
  deselections. Both comparators passed with 13/11 mutation rejections; three
  accepted CUDA regressions also passed.

Both fresh flat-wall processes retained the exact 26-entry/zero-forbidden
topology and the D0001 runtime-derived hit matrix, with positive bounded x=6 m
hits and acquired -x normals in both environments on all eight rows. Both
open-field processes retained exact 25-entry/zero-forbidden M7-014 semantics.
Hit counts remain live evidence and were not made a product-source oracle.

Locked hashes, package/driver identities, formal source and frozen checkout
heads match the accepted baseline. Both runners, both comparators and
`scenario_open_field.py` are byte-identical to D0001. The complete D0001
evidence namespace is hash-identical to its pre-rework snapshot.

No runtime, actor/control/sensor/task/reward/PPO/training source, other
scenario, B1–B8, M8, scale run, merge, push or Git configuration change
occurred.

## Recommendation

`READY_FOR_REVIEW`. This is not self-approval. No next-phase work was started;
Commander review is required before any further milestone or scenario work.
