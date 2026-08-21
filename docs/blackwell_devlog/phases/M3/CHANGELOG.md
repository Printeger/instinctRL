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
