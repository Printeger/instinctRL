# M7-012 Exact B0 Runtime Evaluation Contract

## Public direct-module API

```text
B0RuntimeEvaluationProvenance(...)
B0RuntimeEvaluationConfig(expected_policy_steps, num_envs,
                          rays_per_environment, metric_config, provenance)
begin_ready_b0_runtime_evaluation(manifest, initial_critic, config)
B0RuntimeEvaluationCollector.prepare_command(commanded_body_velocity_m_s)
B0RuntimeEvaluationCollector.commit_transition(staged_command, pre_reset,
                                                exact_observability,
                                                returned_critic,
                                                returned_episode_length)
B0RuntimeEvaluationCollector.finalize(environment_result)
    -> B0RuntimeEvaluationResult
```

The collector is factory-owned, non-subclassable and available only by direct
module import. There are no defaults for manifest, baseline, command, horizon,
provenance, metric configuration, receipt or fallback. Construction owns an
exact current eight-capability B0 READY manifest, exact `[2,18]` initial critic
and an exact 8-step/2-environment/21,240-rays-per-environment configuration.

## Command and transition boundary

`prepare_command` accepts only an exact detached, finite, non-view `[2,3]`
`torch.float32` command on the configured device. It owns one clone and returns
that exact staged tensor for the caller's sole environment `step`. No second
command, policy mapping, learned actor result or normalized `[N,4]` action is
accepted. A failed operation does not change row coverage or silently consume
the pending command.

One `commit_transition` consumes the exact pending object with the accepted
same-state `extras.pre_reset`, exact returned critic and episode length, plus
one current-step, unique-source, independently owned normal-free
`ExactRuntimeObservability`. Missing, extra or reordered keys; aliases;
malformed control/task/termination trees; stale, duplicate or reordered
observability; and any reset, done or termination reject before row commit.

## Evaluation and ownership contract

Each policy transition creates exactly one row for every accepted
`EVALUATION_INPUT_KEYS` field. The fixed initial root position is the station
reference. Anchor validity is false, ICS beta is one, emergency is false and
the unchanged masks/formulas determine unavailable and zero-valued metrics.
Privileged root, velocity, control, task, clearance, termination and
observability fields are evaluation-only and never flow into command creation
or actuation.

The collector calls accepted `compute_step_metrics` and `aggregate_metrics`
without modification. Its ordered `m7.b0-runtime-evaluation.v1` JSON record
uses finite scalars or explicit unavailable metric records with exact
numerator, denominator, valid and total counts. The complete result owns its
manifest/receipt hashes, command/source/result identities, `[8,2]` evaluation
inputs, normal-free details, metrics, record and gates. Raw rays, surface
normals, mesh objects, actor-private lidar/history and live simulator aliases
are absent. Active row, station-reference and pending-command caches are
cleared after exact-once finalization.

## Receipt and bounded CUDA qualification

Finalization requires the exact M7-011 `B0EnvironmentResult`: its READY
identity equals the collector-owned manifest, its EXECUTED manifest is the
exact successor, lifecycle receipt count is 160 and coverage is exactly eight
policy steps, 160 physics/lifecycle/wrench calls and 320 environment samples.
Early, repeated, forged or mismatched finalization emits no record.

Each of two independent headless processes uses the accepted M7-011 scene,
seed and fixed command schedule with two environments and eight policy rows.
It acquires eight exact observability results totaling 339,840 rays, produces
16 metric samples, performs no reset/done, actor inference, normalized action,
pipeline or training call, and emits one environment receipt and one canonical
record. The runner independently recomputes all accepted metric tensors and
scalar/count aggregates; the comparator requires exact complete equality and
rejects structural, bit, count, receipt, record and private-field mutations.

## Capability eight

| Name | Provider | Source SHA-256 |
| --- | --- | --- |
| `m7.b0_runtime_evaluation` | `instinctlab.evaluation.baseline_runtime.B0RuntimeEvaluationCollector` | `c3f1e6f73f48bdcbf995c6f8178d5b93be2d6107af00d81b0ceac69c4a358e66` |

The preceding seven bindings remain exact and ordered. B1-B8 remain unchanged
and non-executable. This contract contains no learned-runtime weakening, fake
audit branch, scenario matrix, baseline comparison, policy-quality claim,
training, M8 or scale work.
