# Runtime Evaluation Contract

`RuntimeEvaluationCollector` accepts only ordered M4 `policy`, detached M6
`pre_reset`, returned public `policy`/critic state, and returned episode lengths.
It imports neither Isaac Lab, the environment, nor training code. Sources are
shape/dtype/device/finite/detachment checked before record-owned clones; extra,
reordered, malformed, inconsistent, invalid-domain, or aliased sources fail
closed.

## Structured source metadata

`SOURCE_MANIFEST` has exactly the accepted 23 `EVALUATION_INPUT_KEYS`, in the
same order. Every value is an ordered record with exactly these nine keys:

```text
source
source_shape
record_shape
dtype
units
frame
validity
ownership
privilege
```

The exact complete metadata serialization has SHA256
`08c62ddb3a6ce6ce6df0f673f14b2826e2de85107d59eb79d4eeed2530f64af0`.
Tests assert that digest plus every field/metadata-key order. The collector
validates every entry against an immutable internal specification before
emitting a deep copy. Every emitted runtime tree owns all nested manifest
containers independently; mutation of one returned tree cannot mutate module
contract data or another tree.

The producer mapping remains unchanged from D0001:

| M7 input group | Exact accepted producer |
| --- | --- |
| commanded velocity | pre-action `policy.state_vec[:, -7:-4]`, body m/s |
| actual / issued velocity | same-state `pre_reset.task.actual_velocity_b` / `pre_reset.control.v_final_b`, body m/s |
| position / station reference | same-state task root world position / collector-owned reset-scoped world position |
| anchor | accepted MAE and exact active/usable validity semantics |
| clearance / collision | exact task clearance, exact-producer validity, and task collision |
| ICS | accepted beta, emergency, and worst-margin diagnostics |
| termination | accepted independent masks and collision > below > above > timeout reason precedence |
| observability | finite zero placeholder, invalid, exact `UNAVAILABLE`; excluded from aggregation |

## Actor and reset binding

Every pre-action mapping is audited before actor inference and contains only
`lidar_grid`, `state_vec`. `record_transition` validates and independently owns
the returned public post-reset policy. The next `prepare_policy` call must match
that returned mapping in exact ordered keys, shapes, dtypes, devices, tensor
values, and per-field hashes before actor inference; a hash-only match is not
sufficient.

For the selective timeout, the smoke tree retains both complete post-reset
policy tensors and the complete next audited actor tensors, their ordered field
hashes, metadata comparisons, `torch.equal` result, and exact step-4-to-step-5
binding. These are evidence only and never enter an evaluation field. The actor
still receives only the exact M4 policy.

Each public step adds one `[N]` evaluation row. A done row uses only that
transition's same-state detached `pre_reset`; its retained position is bound to
the terminal row. Afterward only reset environments receive a new station
reference from returned critic root state, while non-reset references remain
byte-identical. Actor inference uses only deterministic mean action under
`torch.no_grad()`.
