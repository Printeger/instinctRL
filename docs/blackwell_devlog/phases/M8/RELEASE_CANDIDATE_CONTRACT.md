# Immutable Blackwell release-candidate contract

## Fixed identity

```text
schema: m8.blackwell-release-candidate.v1
platform: TASLAB_UAV
sensor: Livox MID360
baseline: BLACKWELL_BASELINE_V2_R580
formal source: 6f6dee3da72d0b0202846b330adcaeeec8ee7bc3
M7 release input: e2d46fc3ad909052182200b7cc958012000a454a
status: MANIFESTED_NOT_QUALIFIED
qualification executed: false
release approved: false
scale training authorized: false
merge authorized: false
push authorized: false
```

The caller supplies the M8 candidate commit. It is provenance only and cannot
replace or alter the accepted M7 release-input identity.

## Accepted chain

| Milestone | Order/revision | Attempt | Product commit | Evidence classes |
| --- | --- | --- | --- | --- |
| M1A | M1A-Q-001/0 | M1A-001-Q0005 | `6eb521f2174a64030e30ea5e8ce8db1f89006619` | STATIC |
| M1B | M1B-001/3 | M1B-001-D0004 | `0f4345e979707f62d9e5ae8ca399bd0029f005b3` | STATIC, PURE |
| M2 | M2-001/0 | M2-001-D0001 | `ecdb482bb05c08cfad141fb290a3d6c2e96298de` | STATIC, PURE, BOUNDED_CUDA, BOUNDED_SIMULATOR |
| M3 | M3-001/1 | M3-001-D0002 | `99ec3b0099b887a31962b2a5c862e3e73efc6872` | STATIC, PURE, BOUNDED_CUDA, BOUNDED_SIMULATOR |
| M4 | M4-001/1 | M4-001-D0002 | `bff6f853eb6d95b12d4de7b95cbfa5a7f2283bbc` | STATIC, PURE, BOUNDED_CUDA |
| M5 | M5-002/1 | M5-002-D0002 | `acda9b1b30ca8aceaf6cf6eaca9d2c1d4bdaaa73` | STATIC, PURE, BOUNDED_CUDA, BOUNDED_SIMULATOR |
| M6 | M6-004/0 | M6-004-D0001 | `033514d8fa7d8d2bb2ece5faa1521d528e32b0a4` | STATIC, PURE, BOUNDED_CUDA, BOUNDED_SIMULATOR |
| M7 | M7-021/1 | M7-021-D0002 | `e2d46fc3ad909052182200b7cc958012000a454a` | STATIC, PURE, BOUNDED_CUDA |

Every row also binds its exact approval path and SHA-256 in the code contract.

## Availability and actor boundary

The M7 matrix identity is
`a0d1c1cdb8cedf18705a7ab4144a682b75ba259ef5e83809d85947546f540b65`:
90 cells, 7 `READY`, 74 `NOT_READY`, 9 `SKIPPED`, and no execution performed.
B1-B7 remain required-unavailable, B8 and dynamic obstacle remain
optional-unavailable, and both real-robot scenarios remain
required-unavailable. `READY` is not relabelled as `EXECUTED`.

The actor contract permits only MID360 measurement history,
validity/reliability, timestamps/age, approved IMU-derived fields and commanded
velocity. Pose, odometry, explicit/reconstructed velocity, maps, SLAM,
goal-relative privilege, obstacle truth, simulator-private state and future
information remain forbidden. Privilege remains confined to reward, critic,
evaluation, logging and labelled upper bounds.

## Qualification gate inventory

| Order | Gate | Classification |
| ---: | --- | --- |
| 1 | accepted milestone-chain integrity | EVIDENCE_BOUND |
| 2 | frozen runtime/package/driver/source identity | EVIDENCE_BOUND |
| 3 | clean candidate tree and reproducible import surface | REQUIRES_BOUNDED_EXECUTION |
| 4 | TASLAB/MID360 topology and frame contract | EVIDENCE_BOUND |
| 5 | deployed actor positive/negative input audit | EVIDENCE_BOUND |
| 6 | reward/critic/evaluation privilege isolation | EVIDENCE_BOUND |
| 7 | bounded environment/controller/rollout/checkpoint reload | REQUIRES_BOUNDED_EXECUTION |
| 8 | B0 seven-scenario evidence consistency | REQUIRES_BOUNDED_EXECUTION |
| 9 | B1-B7 required-unavailable disclosure | REQUIRES_BOUNDED_EXECUTION |
| 10 | B8/dynamic optional-unavailable disclosure | UNAVAILABLE_OPTIONAL |
| 11 | real-robot required-unavailable disclosure | REQUIRES_BOUNDED_EXECUTION |
| 12 | release evidence/index completeness | REQUIRES_BOUNDED_EXECUTION |
| 13 | scale/merge/push user authorization | USER_AUTHORIZATION_REQUIRED |

## Canonical rules

Encoding is deterministic finite ASCII JSON with exact field order and no
timestamp, moving ref, absolute coordination path, metric, threshold or
checkpoint value. The payload SHA-256 excludes only its own hash field.
Decoding rejects duplicates, missing/extra/reordered fields, aliases,
noncanonical or nonfinite JSON and any semantic mutation even when resigned.
