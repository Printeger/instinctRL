# M7-011 Exact B0 TASLAB Environment Contract

## Public direct-module API

```text
B0EnvironmentConfig(expected_policy_steps, direct_config, environment_config)
begin_ready_b0_environment(manifest, config, render_mode)
TaslabB0EvaluationEnv.step(commanded_body_velocity_m_s [N,3])
TaslabB0EvaluationEnv.finalize() -> B0EnvironmentResult
```

The exact class is factory-owned and non-subclassable. The module is not
re-exported by aggregate `instinctlab.evaluation`. There are no manifest,
baseline, horizon, command, fallback, resume, reset, retry or implicit-finalize
defaults. The accepted environment config must retain two environments,
`cuda:0`, decimation 20 and disabled action noise.

## Execution and state contract

Each policy command is an exact finite detached owned-allocation tensor `[2,3]`
on the environment device/dtype. It is cloned once, retained for exactly 20
inherited physics substeps, then cleared. Every substep snapshots public
`root_state_w`, calls the owned M7-010 session once and passes that exact
envelope's Lee normalized rotor command once to the accepted wrench adapter.
One lifecycle step is batched over both environments; it is not multiplied by
environment count.

The exact B0 body/world outputs populate the inherited control identity.
Disabled-component semantics are alpha one, correction zero, anchor inactive,
ICS beta one and emergency false. No HistoryActor, ControlCorePipeline,
governor decode, anchor, ICS, second Lee call or alternate wrench source runs.
Accepted observations, task, reward and diagnostic formulas are reused.

Finalization succeeds once only after exact policy and physics counts. It
returns the exact M7-010 receipt plus frozen scalar policy/physics/lifecycle/
wrench/sample/shape coverage, clears mode command/control/Lee/wrench caches and
does not serialize actor-private values into the receipt.

## Bounded CUDA qualification

Each independent process uses exactly two environments, eight fixed policy
commands, decimation 20, 160 batched lifecycle/controller/wrench calls and 320
environment-samples. Commands include zero, below-limit and over-limit values.
The complete bundle preserves every per-physics command/root/B0/control/Lee/
wrench record, per-policy task coverage, final receipt and all gates. Exact B0
identity uses no tolerance; physical state uses only finite and predeclared
domain bounds, without a flight-performance claim.

## Capability seven

| Name | Provider | Source SHA-256 |
| --- | --- | --- |
| `m7.b0_taslab_environment` | `instinctlab.evaluation.baseline_environment.TaslabB0EvaluationEnv` | `714080d04dee16e73c408f6c0e8c7c8d4a5d0b5a603e698714c5a330c8509e5d` |

The preceding six exact B0 bindings remain ordered. B1-B8 remain unchanged and
non-executable. No collector, scenario matrix, metric comparison, policy
quality claim, training, M8 or scale work is included.
