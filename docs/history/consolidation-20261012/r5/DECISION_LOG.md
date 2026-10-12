# Decision log

## 2026-08-17 — Bootstrap Blackwell migration governance

- Accepted the formal source only at clean `main@6f6dee3da72d0b0202846b330adcaeeec8ee7bc3`, equal to `origin/main` at bootstrap.
- Created `blackwell-migration` in the dedicated worktree from that exact commit; the formal source checkout remains the witness checkout.
- Set governance commit `a0619aa4e60ecc23a016d339f5c32fa4a98ad967` as the M1A base.
- Froze baseline identity as `BLACKWELL_BASELINE_V2_R580`, with modern comparison pins OmniDrones `5045c71` and NavRL `3725bcc2e7c1be4ecf1455d922299ae85042603a`.
- Classified TASLAB_UAV as AechPro Isaac Sim 5.1 fork-derived plus local feasibility patches, not an official modern OmniDrones implementation.
- Required current Git object/hash/size/type evidence to supersede the stale historical approximately-158-KB USD statement.
- Issued M1A as provenance-only, docs-only, one-commit work. No M1B, runtime migration, asset copy/conversion, simulation, controller/MID360 execution, learning change, training, merge, or push is authorized.
