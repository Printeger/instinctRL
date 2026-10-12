# D7 Control Adoption and Validation Report

Status: CPU CANDIDATE PREPARED FOR INDEPENDENT REVIEW; NO LIVE AUTHORITY OR EXECUTION.

## Candidate binding

Repository: `/home/mint/rl_dev/instinctRL_orchestrator/implementation_records/instinctrl-first-use-20260915/state/workspaces/instinctrl-e0-d7-r5-20260923`  
Base commit: `896ec3d21c64babffe6f3d9c7286a2e7ed206fc2`  
Exact candidate representation: base commit plus `r5_delivery/CANDIDATE_SOURCE_MANIFEST.json` SHA256 `e4ea96311752b35dfea485da1fc994d14fb7381485cfc06daf061d5d4207b27c`. The manifest binds the updated workload path, candidate config, control adapter, writer/analyzer, backend, and tests by SHA256. A control-plane adoption must derive and record the clean commit/tree; these files do not claim one.

Current key file SHA256:
- `training_workloads/unified_h1/candidate_config.json`: `78e31113d78cb88719828aa9cf13bafef7d7f22c1bf35c0417432f206d76e714`
- `training_workloads/unified_h1/control_diagnostic.py`: `bdbc6197d65fa5bfbbf4115a6793c112d73ccb2591b8d4bffd1e08a41e85008d`
- `training_workloads/unified_h1/backend.py`: `47f22f3f32b99ca1a8782ab94c20dc47d78482a00776deb1ce76e550fd734a41`

Current handoff preimage SHA256:
- `r5_delivery/D7_R_SPEC_PREIMAGE.json`: `af9899a594456f64e9d3af2c9f8419be6946388f1fb861f1430d5da5a46ae3f8`
- `r5_delivery/D7_R_AUTHORIZATION_PREIMAGE.json`: `1ce854eccfea011682d191a2ee1ea58f3fb7d48d7bc474c67fb4aec6d568c179`
- `r5_delivery/B0_SPEC_PREIMAGE.json`: `f0b3b59223ceb3ef02f3bb2c1555dffe051a0290177f2f7f45f46f6a93e216fa`
- `r5_delivery/B0_AUTHORIZATION_PREIMAGE.json`: `719386fc3acc766a86f6f1686b6c335a9e19e11ec84579e790425230813ea343`


Source E0 JSON: `4210375fac4a16dd83a023e5fc4c3750474bfe80355634b969088ec411b6ea1f` (`PASS_WITH_NOT_PROVEN`; probability replay remains `NOT_PROVEN`).  
Source D7 proposal: `62992b2242e6aca4a9d4d999aab59f57a3c2e650f5208c132c8cebcd3825d703`.

## Validation scope

The isolated adapter binds `control-diagnostic` to evaluation seed `20260909`; evaluation and training bindings remain distinct. The writer reserves serialized bytes in a shared, locked, fsync-durable ledger before every chunk/manifest write. Reservations are cumulative across D7-R and B0 and are never refunded after failure, enforcing the 16 GiB aggregate ceiling fail-closed.

The analyzer requires manifest identity/hash/size, exact Z/T/S order, exact global ordinal and 630 x 32 coverage. It enforces exact field sets and tensor shapes for `pre_reset.task`, `pre_reset.control`, Lee, rotor/wrench, reset masks, episode/time identity, distinct pre-step and post-physics boundaries, post-physics task/root identity, nested-versus-outer terminated/truncated equality, `done = terminated | truncated`, producer mask semantics and collision > below > above > timeout reason precedence, finite tensors, B0 action constancy, and cache validity. Missing, unidentified, wrong-shaped, wrong-time, or invalid evidence raises; no evidence is zero-filled or marked PASS.

Time boundaries remain distinct: 1,800 aggregate action GPU seconds versus 1,820 allocated-GPU and outer-wall seconds including two ten-second cleanup margins. The separate D7-R/B0 spec and authorization preimages bind 40,320 transitions, two starts/initializations, 16 GiB aggregate disk, zero retries, source hashes, attempts, mode, method/config/product/workload fields, exact parameters, and per-condition budgets. Their status is explicitly not authorized/not materialized. A CPU contract test confirms the preimage is rejected as non-authority, then validates an explicitly materialized fixture through the actual `load_r3_authorization` loader. Before any later authorization, the control plane must substitute the adopted clean workload commit/tree and validate the payload under separate authority.

## Immutable boundaries

No product/runtime/frozen evidence/threshold/reward/actor-input/physics/reset semantics were changed. No attempt, experiment, approval, job, simulator, GPU, training, or evaluation was started. Existing R3 HOLD, budgets, and historical findings remain unchanged. Independent review is required.
