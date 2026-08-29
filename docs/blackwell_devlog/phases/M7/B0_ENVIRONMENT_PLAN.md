# M7-011 Exact B0 TASLAB Environment Plan

## Goal

Add one explicit direct-import-only evaluation environment that drives the
accepted TASLAB environment with command-only B0 for an exact bounded horizon:
one batched lifecycle/controller call and one exact wrench application per
physics substep, followed by one privacy-preserving final receipt.

## Source locks

| Source | SHA-256 | Use |
| --- | --- | --- |
| `envs/taslab_direct_env.py` | `e4099b8085e2fe1cf69aac6d249e86d14f0d06ad6dfedc400f717eb14af161b7` | accepted environment hooks/task surfaces |
| `evaluation/baseline_run.py` | `5210566e987b737657d30ba44075a4dc835e379eae438726449dbc76002ab847` | exact bounded lifecycle |
| `evaluation/baseline_controller_execution.py` | `72d104850f8c8fede59998f14661b5b7ea7cd2c113ec11218fe232337732ab52` | exact controller envelope |
| `control/lee_velocity_controller.py` | `5fbd9ea46141a8c9e15a0d67ed04c95ebcbe4a28a7760789f8bdce56f63c54e2` | accepted Lee result |
| `control/runtime_adapter.py` | `b11ca3a0b3ddae6d0d18dda837ee79c7b9c535538d922c3865e4ffebed3c1595` | accepted wrench application |
| `evaluation/baseline_environment.py` | `714080d04dee16e73c408f6c0e8c7c8d4a5d0b5a603e698714c5a330c8509e5d` | seventh B0 capability bytes |

## Implementation and evidence

1. Validate one exact seven-binding READY manifest, positive policy horizon,
   exact B0/env configurations and two-env CUDA/decimation contract before
   constructing motion-capable state.
2. Stage one owned `[N,3]` command per policy step; call the M7-010 session and
   accepted wrench adapter exactly once at every inherited physics substep.
3. Populate same-state environment diagnostics with exact B0 body/world and
   fresh explicit disabled anchor/ICS values; retain no alternate control path.
4. Reject malformed/overrun/finalization/reset boundaries before unauthorized
   simulator calls and clear mode tensor caches at finalization.
5. Prove pure/static behavior twice, full regressions, accepted bounded smokes,
   and two independent exact 2-env/8-policy/160-physics process bundles.

## Stop conditions

Stop on accepted-source/default-environment mutation, any actor/pipeline or
alternate Lee path, non-exact lifecycle/wrench coverage, reset/collision in the
bounded smoke, privacy weakening, unlisted path, frozen/package drift,
collector/scenario/training/M8/scale need, merge, push or history rewrite.
