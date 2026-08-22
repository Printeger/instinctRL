# Runtime Evaluation Contract

`RuntimeEvaluationCollector` accepts only ordered M4 `policy`, detached M6 `pre_reset`,
returned public critic state, and returned episode lengths. It imports neither Isaac Lab,
the environment, nor training code. Sources are shape/dtype/device/finite/detachment
checked before record-owned clones; all source aliases fail closed.

| M7 input | Exact accepted producer |
| --- | --- |
| commanded_velocity_b_m_s | pre-action `policy.state_vec[:, -7:-4]`, body m/s |
| actual_velocity_b_m_s | `pre_reset.task.actual_velocity_b`, body m/s |
| issued_velocity_b_m_s | `pre_reset.control.v_final_b`, body m/s |
| position_w_m | `pre_reset.task.root_state_w[:, :3]`, world m |
| station_reference_position_w_m | collector-owned reset-scoped world position, m |
| anchor_error_m / anchor_valid | anchor MAE; accepted active/usable semantics |
| clearance / validity / collision | exact task clearance, exact-producer validity, task collision |
| three ICS fields | accepted beta, emergency, and worst-margin diagnostics |
| terminated / truncated / reason | accepted same-state termination result |
| four termination masks | collision, below-height, above-height, timeout unchanged |
| three observability fields | zero placeholder, invalid, `UNAVAILABLE`; excluded |

The serialized source manifest declares all 23 ordered fields, including units and
ownership. Each step adds one `[N]` row. A done row uses terminal `pre_reset`; afterward
only reset environments receive a new reference from returned critic root state. Actor
input is the same audited policy object and the runner calls only deterministic mean
action under `torch.no_grad()`.
