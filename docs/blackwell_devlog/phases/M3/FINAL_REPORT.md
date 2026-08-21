# M3 Final Report

## Scope completed

M3-001 adds the fixed 360-by-59 sensor-local pattern, the approved TASLAB `base_link` mount, the 0.1-to-40 m fail-closed validity contract, the 0.1 s update configuration, independent static tests, and a minimal deterministic CUDA ray-caster smoke.

## Qualified contract

- 21,240 float32 unit rays, elevation outer and azimuth fastest.
- Axes `+X` forward, `+Y` left, `+Z` up.
- Body mount position `(0,0,0.05) m`, attitude `(w,x,y,z)=(0.9659258262890683,0,0.25881904510252074,0)`, and `ray_alignment="base"`.
- Raw invalid values remain distinguishable; serialized invalid values are exactly 40 m with a separate false mask.
- CUDA output shape `[1,21240]`, with both valid and invalid returns.

## Runtime result

Each independent process used one SimulationContext, one TASLAB Articulation, one static ground plane, and one RayCaster. It performed one public reset, one public sensor update, zero explicit physics steps, and zero control calls. Both outputs were byte-identical; the semantic comparator covered every range and mask entry and reported maximum absolute numeric error `0.0`.

## Correctness gates

- Fixed pattern/frame/mount/range/timing semantics: PASS.
- Public CUDA ray-caster initialization and both return classes: PASS.
- Finite serialization and operation audit: PASS.
- Full two-process comparison: PASS.
- M1B/M2 unchanged and allowed-path hygiene: PASS.
- Frozen runtime identity: PASS, with the previously recorded Isaac Lab checkout modification unchanged.
- Prohibited later-phase work: PASS.

## Recommendation

`READY_FOR_REVIEW`

NO M4/ACTOR/HISTORY/CONTROLLER/PPO/TRAINING WORK STARTED.

AWAITING COMMANDER REVIEW.

## Revision 1 correction

Commander review found that D0001 passed the tracked parent position directly
to range conversion. Attempt `M3-001-D0002` preserves both historical commits
and corrects the wiring with one validated helper:

`sensor_origin_w = parent_pos_w + quat_apply(parent_quat_w, mount_pos)` and
`sensor_quat_w = quat_mul(parent_quat_w, mount_quat)`.

The smoke records parent and mounted poses separately and passes only the
mounted origin to `ranges_and_validity`. Its measured translation separation is
`0.0499999523 m` (error `4.842877388e-8 m`) and quaternion composition maximum
error is `2.030979829e-9`.

Every ray is checked independently against a persisted 84 m by 84 m tiled
world-z=0 mesh. The complete runtime mask has zero mismatches and all 4,290
analytic valid ranges have maximum error `7.62939453125e-6 m`. Substituting the
uncombined parent origin yields 18 mask mismatches and maximum range discrepancy
`0.9721717834 m`, so the D0001 wiring fails the new gate.

Two corrected CUDA processes are byte-identical. The unchanged comparator
covers all 21,240 ranges and all 21,240 validity bits with process error `0.0`.
The final M3 static suite passes 8/8 twice, complete M2 regression passes, and
zero control or explicit physics steps are introduced.

Revision-1 recommendation: `READY_FOR_REVIEW`.

NO M4/ACTOR/HISTORY/CONTROLLER/PPO/TRAINING WORK STARTED.
