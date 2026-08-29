# M7-010 Exact B0 Run Lifecycle Contract

## Public API

```text
B0RunConfig(expected_steps: int, direct_config: B0DirectConfig)
begin_ready_b0_run(manifest: BaselineRunManifest, config: B0RunConfig) -> B0RunSession
B0RunSession.step(commanded_body_velocity_m_s, controller_private_root_state_w)
B0RunSession.finalize() -> B0RunReceipt
```

There are no defaults, aliases, fallback, implicit finalization, retry, abort,
resume, reset, concurrency, provider injection, or environment hooks. The
session is exact-type and non-subclassable; construction is factory-only.

## State machine and atomicity

The factory reparses and owns one canonical B0 READY manifest under the current
six-binding registry. A session begins `ACTIVE` at step zero. Each permitted
step calls the accepted controller handoff exactly once and returns that exact
envelope. It stages an owned EXECUTED successor, digest copy, and next counter,
then commits only after successor consistency and complete hashing. Any input,
handoff, successor, or framing exception leaves state and digest unchanged.

Overrun and post-finalization step calls reject before handoff. Early and
repeated finalization reject atomically. Successful finalization occurs once
after `expected_steps` and changes the state token to `FINALIZED`.

## Trajectory framing

Schema `m7.b0-run-trajectory.v1` uses unsigned little-endian 64-bit length/
dimension/index framing. The header binds the READY manifest hash, configured
step count, and exact direct velocity limit. Every ordered step binds its
index, READY/EXECUTED hashes, exact ASCII field names, float32/float64 token,
rank, dimensions, and contiguous little-endian bytes for:

- actor-approved commanded body velocity;
- controller-private root input;
- B0 issued body/world velocity;
- all five `LeeControllerResult` tensors.

Field name, bit, dtype, shape, order, manifest, configuration, or step-count
changes therefore alter the SHA-256. No tensor value is copied into a manifest
or receipt.

## Receipt and privacy

The frozen receipt schema `m7.b0-run-receipt.v1` contains only schema version,
independent valid READY and EXECUTED manifests, exact step count, and lowercase
trajectory SHA-256. The session retains no command, root, B0, or Lee tensor.
Public introspection exposes only state token, next step, expected steps, and
finalized boolean.

## Capability six

| Name | Provider | Source SHA-256 |
| --- | --- | --- |
| `m7.b0_run_lifecycle` | `instinctlab.evaluation.baseline_run.begin_ready_b0_run` | `5210566e987b737657d30ba44075a4dc835e379eae438726449dbc76002ab847` |

The preceding five bindings remain ordered and exact. B1-B8 remain unchanged
and non-executable. This boundary does not integrate an environment, collector,
scenario, simulator, metrics, performance evaluation, or training.
