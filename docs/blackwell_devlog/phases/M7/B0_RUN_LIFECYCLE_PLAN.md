# M7-010 Exact B0 Run Lifecycle Plan

## Goal

Add one pure environment-independent state machine that owns a canonical B0
READY manifest, performs exactly the configured positive number of accepted
controller-handoff steps, commits each step atomically, and finalizes once to a
privacy-preserving deterministic receipt.

## Source locks

| Source | SHA-256 | Use |
| --- | --- | --- |
| `control/baseline_b0.py` | `5fa2b8d746eb231a1bb12f331fe1a76e8f988005e9a315e1dbf622e0bf1ed3d1` | exact immutable direct config |
| `evaluation/baseline_execution.py` | `db7b73ee7e9c4b39689814530363bafb6a1ef48c2513a0208c0ff29c7073c108` | accepted READY/EXECUTED semantics |
| `evaluation/baseline_controller_execution.py` | `72d104850f8c8fede59998f14661b5b7ea7cd2c113ec11218fe232337732ab52` | exact per-step handoff |
| `evaluation/baseline_run.py` | `5210566e987b737657d30ba44075a4dc835e379eae438726449dbc76002ab847` | sixth B0 capability bytes |

## Implementation

1. Reparse and own one exact self-verifying B0 READY manifest with six ordered
   bindings and one exact positive run configuration.
2. Delegate every attempted data step exactly once to the accepted handoff.
3. Validate one invariant EXECUTED successor; stage the digest, successor, and
   counter locally; commit them only after all checks and framing complete.
4. Frame schema, configured count/limit, manifest hashes, step index, exact
   field names, dtype, rank/shape, and contiguous little-endian tensor bytes.
5. Retain no input/output tensor and expose only scalar lifecycle properties.
6. Finalize exactly once after all steps to independent manifests and a
   lowercase SHA-256 receipt; add pure independent process/hash evidence.

## Boundaries and stop conditions

Only the 14 M7-010 product paths are writable. Stop on actor/receipt leakage,
non-atomic mutation, tensor retention, successor drift, non-injective framing,
need for accepted source/environment/runtime changes, package/frozen mutation,
CUDA/simulator use, training/M8/scale work, merge, push, or history rewrite.
