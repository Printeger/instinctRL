# M7-006 Baseline Isolation Final Report

## Scope completed

The functional commit `93602e05238cf207bfefea235cbaadef35c957f6`
(parent/base `dbc05a58ca98958c9100fa5e002fe6b9083e40ec`) adds a pure frozen B0-B8
registry, strict exact-ID/config resolution, evidence-bound capability
validation, and immutable canonical per-run manifests. B1 is visibly
nondeployable and privileged; every other row is deployable and limited to the
closed onboard actor-field allowlist. B8 is present and can only be explicitly
skipped when unavailable.

## Registry outcome

| ID | Config | Family | Privilege/deployability | Capability state |
| --- | --- | --- | --- | --- |
| B0 | `direct_velocity` | direct control | actor-clean/deployable | READY, accepted M5 provider/hash |
| B1 | `state_velocity_upper_bound` | learned governor | privileged upper bound/nondeployable | NOT_READY |
| B2 | `no_history` | learned governor | actor-clean/deployable | NOT_READY |
| B3 | `no_anchor` | learned governor | actor-clean/deployable | NOT_READY |
| B4 | `no_imu` | learned governor | actor-clean/deployable | NOT_READY |
| B5 | `no_ics` | learned governor | actor-clean/deployable | NOT_READY |
| B6 | `distance_only_filter` | deterministic filter | actor-clean/deployable | NOT_READY |
| B7 | `braking_heuristic` | heuristic filter | actor-clean/deployable | NOT_READY |
| B8 | `simplified_cbf_vo_optional` | model-based filter | actor-clean/deployable/optional | NOT_READY; explicit SKIPPED rule |

No unknown, alias, incomplete, unsupported, wrong-hash, privileged, or
unavailable selection can fall back to B0 or the full method.

## Change ledger

| File | Change | Requirement | Semantic effect |
| --- | --- | --- | --- |
| `isaaclab-training/README.md` | Precise non-executing layer note | README scope | Documentation only |
| `instinctlab/evaluation/__init__.py` | Explicit public baseline exports | Package API | Additive exports; existing exports retained |
| `instinctlab/evaluation/baselines.py` | Frozen registry, validators, manifest | Core goal | New evaluation-only pure API |
| `tests/test_m7_baseline_isolation_contract.py` | Positive/negative/static/hash tests | Required tests | Test only |
| `BASELINE_ISOLATION_PLAN.md` | Bounded implementation plan | Required record | Documentation only |
| `BASELINE_ISOLATION_CONTRACT.md` | Exact row/schema/status contract | Required record | Documentation only |
| `BASELINE_ISOLATION_TEST_REPORT.md` | Results and requirement map | Required record | Documentation only |
| `BASELINE_ISOLATION_EVIDENCE_INDEX.md` | Raw evidence/hashes | Required record | Documentation only |
| `BASELINE_ISOLATION_KNOWN_LIMITATIONS.md` | Factual limitations | Required record | Documentation only |
| `BASELINE_ISOLATION_FINAL_REPORT.md` | Scope/gates/stop declaration | Required record | Documentation only |

## Correctness gates

- Exact nine-row identity/order/schemas/toggles/readiness: **PASS**.
- B1 privilege and all deployable actor schemas: **PASS**.
- Capability evidence, no alias/fallback/overclaim, B8 explicit skip: **PASS**.
- Canonical byte/order/hash/round-trip/mutation/finite/audit binding: **PASS**.
- Standard-library implementation and reverse-import isolation: **PASS**.
- Focused tests twice, accepted M7 twice, M3-M6 regressions twice, M2 static:
  **PASS** (1,046 pytest passes, zero skips; 66 additional subtests).
- Functional commit exact allowlist, clean state, diff hygiene, frozen package/
  formal/external integrity: **PASS**.

## Provenance and limitations

Handbook, paper, accepted M4 actor schema, M5 B0 source, and M7 source-manifest
files were read-only inputs. B0 capability evidence is the unchanged
`governor.py` SHA-256
`cd5cf6c73578bc69dc5f696ac6d294d9d4c8db0baaf12265e6715646554b7197`.
Known limitations are recorded separately and are not performance claims.

## Stop declaration

No baseline behavior or scenario was executed. No actor/control/environment/
task/reward/training/runtime/observability/source-manifest semantics changed.
No CUDA, simulator, plot, hardware proxy, real robot, PPO/training, M8,
128k/1M/scale work, package mutation, frozen-tree mutation, Git configuration
change, history rewrite, merge, or push was performed.

`NO NEXT-PHASE WORK WAS STARTED.`

`AWAITING COMMANDER REVIEW.`
