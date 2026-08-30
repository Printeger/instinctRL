# M7-019 unsafe-command-toward-wall B0 contract

## Exact identity

```text
scenario: unsafe_command_toward_wall
baseline: B0
seed: 7019
environments: 2
policy rows: 8
physics/controller calls: 160
ray samples: 339840
fixture source SHA256: e587f36c06897e3410ffa225b40c3d267ad61d785240194abbd61750333b8329
```

The body-frame schedule is positive x only with row magnitudes
`0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05` m/s, duplicated across both
environments. The collector-owned staged command, environment command,
controller input and retained `v_final_b` must be bit-identical.

## Independent direction witness

For every row, the runner retains exactly 20 controller-private root-state
snapshots and selects the twentieth snapshot associated with the retained
`pre_reset.control` output. With `q_raw = root_state_w[:,3:7]` and staged
`v_b`, the independent oracle is exactly:

```text
q_norm = q_raw / torch.linalg.vector_norm(q_raw, dim=-1, keepdim=True)
v_independent_w = isaaclab.utils.math.quat_apply(q_norm, v_b)
```

Raw norms must be finite and within `1e-5` of one. Accepted `v_final_w` must be
bit-exact to `v_independent_w`; tolerance equality and importing the accepted
rotation adapter are forbidden. Every world-x component and
`6.0 - root_state_w[:,0]` must be strictly positive.

## Geometry and observability

The selected scenario obstacle is exactly one four-point/two-triangle/six-index
collision Mesh at `/World/MID360StaticScene/UnsafeCommandTowardWallMesh`,
bounded at x=6 m with authored -x normal. The RayCaster plane remains a
separate `PlaneMesh`. The complete permitted inventory contains 26 entries and
no forbidden geometry. Every valid selected-mesh hit must belong to the wall;
all 16 cells require at least one hit and acquired -x normals.

## Comparator and claim boundary

Two complete decoded bundles must compare exactly. The comparator retains 17
mutation classes; its `command` mutation is the exact type-preserving one-row
left rotation `schedule[1:] + schedule[:1]`, not the no-op full reversal.

The qualification proves exact command-direction, geometry/observability and
migration wiring identity only. It makes no safety, avoidance, effectiveness,
performance or cross-baseline comparison claim. Actor calls remain zero and
private pose/geometry evidence is evaluation-only.
