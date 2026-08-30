# M7-015 Flat-Wall B0 Final Report

## Outcome

Order M7-015 is implemented at functional commit
`fdc81997ac5055a626571f244d68890cf1a2a425`, a direct child of accepted base
`eb4f265f181169fcd75ac16fb4bc4a49be441a87`. Exactly nine functional paths
changed in that commit. The catalog READY/executable set is exactly
`(flat_wall,7015)` and `(open_field,7014)` in canonical order; all other rows
retain NOT_READY and actor-privacy semantics.

The dedicated runner authored the exact x=6 m CollisionAPI triangle wall before
environment/RayCaster construction. Both formal processes traversed the exact
accepted 25-entry inventory plus only `FlatWallMesh`, found zero forbidden
paths, preserved the z=0 plane and recorded the same positive wall-hit counts
for both environments on all eight rows. Acquired wall normals were exactly -x
within `1e-5`; every valid static hit classified as wall or plane.

## Exact evidence

- Flat-wall source / qualification:
  `ed170f337abf9cd2aff288b54473b9404824be7885b4c3fa61e75484de88bdcf` /
  `2592b2e6dddd0a27fccc126ab98acee644eb3d22fa1f79d6eac4004579cb8497`.
- Open-field v2 source / qualification:
  `04c743b6c3ef31ff74b5ca9e5d4c18a808f5a171fb832e3932aa12c82950d976` /
  `15368f7e8feabb61bf39ac1d4251cb97a3ecc65ff47cce1271e3c812b68474e2`.
- Each scenario process: 2 environments, 8 policy steps, 16 metric samples,
  160 physics/lifecycle/wrench calls, 320 environment samples and 339,840 rays;
  zero reset/done/actor/training calls and one receipt/B0/qualification record.
- Pure/static tests: 2,072 passed, two disclosed accepted M6 deselections.
- Flat-wall/open-field comparators: complete equality and 13/11 rejected
  mutation classes. Three accepted CUDA regressions passed.

Locked hashes, package/driver identities, formal source and frozen checkout
heads match the accepted baseline. IsaacLab retains only its accepted known kit
override at exact hash `0a0dee5f…`. M7-014 product history and evidence were not
modified. No aggregate export, callback, actor/control/sensor/task/reward/PPO/
training change, other scenario, B1–B8, M8, scale run, merge or push occurred.

## Recommendation

`READY_FOR_REVIEW`. This is not self-approval. No next-phase work was started;
Commander review is required before any further milestone or scenario work.
