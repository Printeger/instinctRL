# M7-014 Final Report

The exact open-field B0 scenario qualification is complete and ready for Commander review. Functional commit `c93d1ea8bca7683e7a1a855952e114043caa0ada` is the direct child of base `deeb7ad4221df1f0e74ec0ac24affefbc27b817d`.

Exactly `open_field` became READY/executable through the new committed fixture source. The other nine scenario rows remain NOT_READY and retain their prior identity/domain/requirement/reason/metric/privacy values; `dynamic_obstacle` remains the only optional row. No callback or implicit launch exists.

Two fresh headless CUDA processes independently constructed the exact accepted B0 environment only after binding the READY manifest and run plan. Each observed the ground and exact z=0 MID360 plane mesh, ran the sole seed-7014 eight-row schedule through the one-pending-command collector state machine, and produced 8 policy rows, 16 metric samples, 160 physics/lifecycle/wrench calls, 320 environment samples and 339,840 ray samples. Reset, done, learned actor and training calls were zero. Each produced one environment receipt, one B0 evaluation record and one `m7.open-field-b0-qualification.v1` record.

Independent geometry, command, source-step, metric, coverage, receipt, record, finiteness and privacy oracles passed. The qualification SHA-256 was identical across processes: `bf035ef4ae46200315f58911d77d284baea98f0cf01d94cf539abe041a097133`. Full decoded comparison passed and 11 mutation classes were rejected.

All locked runtime/environment/metric/audit/raycaster/source/Handbook hashes remained unchanged. Driver, Conda, PyTorch, Isaac Sim, Isaac Lab and frozen baselines were not changed. This result qualifies migration wiring and scenario identity only; it makes no performance, comparison, safety or flight-quality claim.

No other scenario, B1-B8 scenario execution, training, evaluation campaign, M8, merge, push or history rewrite was started.
