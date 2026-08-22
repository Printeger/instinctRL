# M7-001 Evaluation Metrics Contract

## Public information boundary

The public mapping order is exactly `policy`, then `evaluation`. `policy` is
passed directly to accepted M4 `audit_actor_observation` before any metric is
computed and remains exactly:

- `lidar_grid [N,12,59,360]`, float32/float64;
- `state_vec [N,52]`, same dtype/device.

No evaluation value is copied, reconstructed, or forwarded into that mapping.
The evaluation branch is privileged evaluation/logging input only.

At the public boundary, accepted M4 audit runs first. Before provenance,
metric, hashing, or record work, every evaluation tensor's underlying untyped
storage identity is compared with both policy tensor storages. Direct aliases
and differently shaped, strided, or nonzero-offset views fail closed. Equal-
valued tensors with independent storage remain valid. This cross-branch check
does not clone an input to hide aliasing and does not change the retained
evaluation-internal alias check.

## Exact evaluation input order

All vector tensors are `[T,N,3]`; all scalar/mask/class tensors are `[T,N]`.
Floating tensors share one float32/float64 dtype and device. Masks are bool;
reason/source classes are `torch.long` on the same device. Every tensor is
finite where applicable, and pairwise storage aliasing is rejected.

| Order | Field | Units/frame | Ownership/classification |
| ---: | --- | --- | --- |
| 1 | `commanded_velocity_b_m_s` | m/s, body | evaluation copy of approved command |
| 2 | `actual_velocity_b_m_s` | m/s, body | privileged simulator/evaluator |
| 3 | `issued_velocity_b_m_s` | m/s, body | evaluation copy of M5 final command |
| 4 | `position_w_m` | m, world | privileged evaluator |
| 5 | `station_reference_position_w_m` | m, world | privileged evaluator reference |
| 6-7 | `anchor_error_m`, `anchor_valid` | m, measurement anchor | evaluation-only value/validity |
| 8-9 | `minimum_clearance_m`, `clearance_valid` | m | privileged evaluation value/validity |
| 10 | `collision` | bool | privileged label |
| 11-13 | `ics_beta`, `ics_emergency`, `ics_worst_margin_m_s` | ratio, bool, m/s | accepted M5 diagnostics |
| 14-20 | `terminated`, `truncated`, `termination_reason`, four cause masks | bool/enum | accepted M6 semantics |
| 21-23 | `observability_value`, `observability_valid`, `observability_source` | source-defined/bool/enum | evaluation only |

Source codes are exactly `UNAVAILABLE=0`, `PROXY=1`, `EXACT=2`; validity is
exactly `source != UNAVAILABLE`. A PROXY is never relabelled exact. This Order
defines no surface-normal, raycast, finite-difference, or Jacobian algorithm.

## Equations and gates

For sample set `S` with `|S|=T*N`:

- tracking: `sqrt(sum(||v_actual_b-v_cmd_b||^2)/|S|)`;
- null gate: `||v_cmd_b|| <= 1e-3 m/s`;
- null drift: mean `||p_w-p_reference_w||` under the null gate;
- null speed: mean `||v_actual_b||` under the null gate;
- anchor error: mean over `anchor_valid`;
- clearance: minimum over `clearance_valid`;
- collision rate: boolean mean over all samples;
- ICS violation: mean of `ics_worst_margin_m_s < 0` over all samples;
- intervention: `ics_beta < 0.999 or ics_emergency`;
- preservation: mean `||v_issued_b||/||v_cmd_b||` only for non-null,
  non-intervened samples; the denominator is strictly positive by the gate;
- observability: mean only where the source is PROXY or EXACT.

Mean/rate/RMSE results expose the selected-value sum as numerator and selected
sample count as denominator. Minimum exposes the selected minimum as numerator,
denominator one, and separately reports every selected sample as valid count.
Every result reports total count. Empty selections report `available=false`,
`value=null`, `numerator=null`, denominator/valid count zero; no zero, NaN,
epsilon divisor, clamp, or default PASS is fabricated.

## Termination and record semantics

The four cause masks remain independent. `terminated=collision|below|above`,
`truncated=timeout`, and reason precedence is collision, below, above, timeout.
The record separately counts every cause, terminated, truncated, done,
simultaneous terminated/truncated, and each precedence reason.

The canonical JSON order is schema, platform `TASLAB_UAV`, sensor `Livox
MID360`, bounded claim, actor audit manifest and tensor hashes, dimensions,
input order, config, observability source counts, metrics, termination counts,
and provenance. JSON uses `allow_nan=false`; the immutable record owns its
string, while step outputs own detached clones.
