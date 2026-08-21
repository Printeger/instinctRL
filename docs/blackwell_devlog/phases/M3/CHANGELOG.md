# M3 Changelog

| File | Change | Reason | Order requirement | Semantic impact |
| --- | --- | --- | --- | --- |
| `isaaclab-training/README.md` | Documented the bounded M3 surface. | Make scope visible at the runtime root. | Persisted M3 records. | Documentation only. |
| `isaaclab-training/instinctlab/sensors/__init__.py` | Exported the MID360 contract. | Provide the modern package surface. | Package export. | No runtime side effect. |
| `isaaclab-training/instinctlab/sensors/mid360.py` | Added fixed pattern/config and validity conversion. | Migrate the approved static sensor contract. | Geometry, mount, range, validity, and timing. | Adds deterministic sensor configuration only. |
| `isaaclab-training/tests/test_m3_static.py` | Added analytic, negative, determinism, and scope tests. | Prove the fixed contract independently. | Required static tests. | Test only. |
| `isaaclab-training/tests/run_m3_mid360_smoke.py` | Added one-sensor CUDA static-scene smoke. | Qualify public Isaac Lab runtime behavior. | Required runtime smoke. | Test-only simulator execution; no control. |
| `isaaclab-training/tests/compare_m3_smoke.py` | Added full-output comparator. | Prove two-process repeatability. | Compare all 21,240 ranges/mask values. | Test only. |
| `docs/blackwell_devlog/phases/M3/PLAN.md` | Recorded the bounded execution plan. | Governance traceability. | Plan before implementation. | Documentation only. |
| `docs/blackwell_devlog/phases/M3/CHANGELOG.md` | Recorded file-level changes. | Phase traceability. | Required phase record. | Documentation only. |
| `docs/blackwell_devlog/phases/M3/TEST_REPORT.md` | Recorded staged results. | Separate executed evidence from claims. | Required phase record. | Documentation only. |
| `docs/blackwell_devlog/phases/M3/EVIDENCE_INDEX.md` | Indexed coordination evidence. | Locate full raw outputs outside Git. | Required phase record. | Documentation only. |
| `docs/blackwell_devlog/phases/M3/KNOWN_LIMITATIONS.md` | Recorded bounded limitations. | Prevent scope overclaim. | Required phase record. | Documentation only. |
| `docs/blackwell_devlog/phases/M3/FINAL_REPORT.md` | Summarized M3 delivery and gates. | Executor handoff. | Required phase record. | Documentation only. |

## Revision 1 mounted-origin correction

| File | Change | Reason | Order requirement | Semantic impact |
| --- | --- | --- | --- | --- |
| `isaaclab-training/README.md` | Documented mounted pose and full-plane oracle. | Correct the M3 contract description. | Revision-1 truthful record. | Documentation only. |
| `isaaclab-training/instinctlab/sensors/__init__.py` | Exported the mounted-pose helper. | Provide one reusable transform boundary. | Explicit testable helper. | Adds one public sensor utility. |
| `isaaclab-training/instinctlab/sensors/mid360.py` | Added validated parent/mount pose composition. | Prevent base-link origin from being used as ray origin. | Mounted-origin correction. | Ranges can use the actual mounted sensor pose. |
| `isaaclab-training/tests/test_m3_static.py` | Added nontrivial analytic transform and malformed-input fixtures plus runtime wiring assertions. | Reject D0001 wiring and fail closed. | Required regression tests. | Test only. |
| `isaaclab-training/tests/run_m3_mid360_smoke.py` | Uses mounted origin; records both poses; constructs world directions; checks every ray against a tiled z=0 plane and rejects the parent-origin substitute. | Correct and prove runtime geometry. | Full-plane analytic oracle. | Test-only corrected range qualification. |
| `docs/blackwell_devlog/phases/M3/*.md` | Added revision-1 plan/results/evidence/limitations/report. | Preserve D0001 while reporting D0002 truthfully. | Required fresh records. | Documentation only. |

`isaaclab-training/tests/compare_m3_smoke.py` is unchanged at SHA256
`5f01966fea83d6eeb0ccdd464834a72743886f605ceb32afb4da1c8774edf1b6`.
