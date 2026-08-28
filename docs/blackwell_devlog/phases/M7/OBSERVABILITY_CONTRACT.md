# Exact Observability Contract

## Boundary and API

`compute_exact_observability(surface_normals_w, beam_valid, config)` is pure
PyTorch/Python and is exported only by `instinctlab.evaluation`.
`ExactObservabilityConfig` and `ExactObservabilityResult` are frozen records.
The default finite positive unit-normal tolerance is `1e-5`.

Inputs are exactly:

| Input | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `surface_normals_w` | `[T,N,R,3]` | float32 or float64 | caller-supplied unit surface normals in world axes |
| `beam_valid` | `[T,N,R]` | bool | exact valid-return mask |

`T`, `N`, and `R` are positive. Both tensors use the same device, own their
exact storage, are not views or aliases, and are finite. Invalid beams contain
exact zero normal placeholders. Valid normals have length one within the
configured inclusive tolerance. Validation does not mutate, normalize, repair,
clone-and-accept, or discard invalid inputs.

## Mathematics

Every valid normal contributes one row

```text
J_i = -n_i^T
```

in world coordinates. Singular values are returned in ascending order and
zero-padded to exactly three values. With `sigma_max` the largest value,

```text
rank_tolerance = max(R, 3) * eps(input_dtype) * sigma_max
rank = count(sigma_i > rank_tolerance)
```

The condition number is `sigma_max / sigma_min` only at numerical rank three.
Otherwise its value is finite zero and `condition_valid` is false.

The weak direction is the right singular vector for `sigma_min` only when the
smallest singular subspace is one-dimensional under the same rank tolerance.
Its sign is canonical: the largest-absolute component is nonnegative, with
absolute ties resolved by the lowest component index. A repeated smallest
singular value produces an exact zero vector and false validity.

Zero valid beams produce exact rank zero, three zero singular values, zero
`sigma_min`, unavailable zero condition and weak direction, zero count, and
zero fraction. Open-field unobservability is therefore retained explicitly.

## Results

| Field | Shape | Dtype |
| --- | --- | --- |
| `numerical_rank` | `[T,N]` | int64 |
| `singular_values` | `[T,N,3]` | input float dtype |
| `sigma_min` | `[T,N]` | input float dtype |
| `condition_number`, `condition_valid` | `[T,N]` | input float dtype, bool |
| `weak_direction_w`, `weak_direction_valid` | `[T,N,3]`, `[T,N]` | input float dtype, bool |
| `valid_ray_count`, `total_ray_count` | `[T,N]` | int64 |
| `valid_beam_fraction` | `[T,N]` | input float dtype |

Every result tensor is finite, detached, independently owned, mutually
non-aliased, and on the input device. Ray permutations and normal sign flips
preserve scalar/spectrum results and every uniquely defined canonical weak
direction.

## Isolation

Surface normals and exact results are evaluation-only. They are absent from
actor observations, actor history, governor, anchor, ICS, controller,
environment public observations, reward, critic, PPO, rollout, checkpoint,
deployment, and training paths. This core does not change M7-002 records;
runtime observability remains exactly `UNAVAILABLE`.
