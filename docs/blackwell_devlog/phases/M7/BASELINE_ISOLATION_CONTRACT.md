# M7-006 Baseline Isolation Contract

## Registry identity

The registry is an ordered tuple of frozen records. ID and config switch are
one-to-one exact identities; case folding, trimming, aliases, `ours`, `full`,
switch-as-ID, and ID-as-switch are rejected.

Validation locks every value in every `BaselineSpec` field, not only the row
ID, switch, privilege, and readiness invariants. A separately constructed
tuple passes only when each row is value-identical to the canonical registry;
any single-field drift in identity, family, descriptive/contribution text,
deployability, privilege, actor/evaluation schema, toggles, capability,
optionality, or readiness is rejected.

| ID | Config switch | Execution family | Deployable | Privilege | H/I/A/ICS | Optional | Current readiness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | `direct_velocity` | `direct_control` | yes | actor-clean | 0/0/0/0 | no | `READY` |
| B1 | `state_velocity_upper_bound` | `learned_governor` | no | evaluation-only privileged upper bound | 1/1/1/1 | no | `NOT_READY` |
| B2 | `no_history` | `learned_governor` | yes | actor-clean | 0/1/1/1 | no | `NOT_READY` |
| B3 | `no_anchor` | `learned_governor` | yes | actor-clean | 1/1/0/1 | no | `NOT_READY` |
| B4 | `no_imu` | `learned_governor` | yes | actor-clean | 1/0/1/1 | no | `NOT_READY` |
| B5 | `no_ics` | `learned_governor` | yes | actor-clean | 1/1/1/0 | no | `NOT_READY` |
| B6 | `distance_only_filter` | `deterministic_filter` | yes | actor-clean | 0/0/0/0 | no | `NOT_READY` |
| B7 | `braking_heuristic` | `heuristic_filter` | yes | actor-clean | 1/0/0/1 | no | `NOT_READY` |
| B8 | `simplified_cbf_vo_optional` | `model_based_filter` | yes | actor-clean | 0/0/0/0 | yes | `NOT_READY` |

`H/I/A/ICS` means history, IMU, anchor, and ICS respectively. It records the
component identity only and does not implement any behavior.

## Policy and evaluation schema boundary

B0 consumes only `commanded_body_velocity_m_s`. B1 uses the full accepted M4
source schema plus `explicit_body_velocity_m_s` and `local_odometry`; this is
why B1 is nondeployable and visibly privileged. B2-B5 are exact removals from
the M4 fields. B6 consumes minimum range and command. B7/B8 consume only
MID360 range/validity/time plus command; B7 additionally declares history for
range-rate use. Every deployable field is checked against the closed onboard
allowlist.

Every row records the same exact accepted 23-field M7 evaluation-only schema.
It includes actual velocity, world position, exact clearance, collision,
termination, ICS, anchor, and observability records. Those fields never become
actor inputs.

## Capability and readiness boundary

B0 requires capability `m5.b0_direct_velocity`, provider
`instinctlab.control.governor.b0_governor`, with accepted source SHA-256
`cd5cf6c73578bc69dc5f696ac6d294d9d4c8db0baaf12265e6715646554b7197`.
Only this row is currently `READY`. B1-B8 name distinct requirements but have
no accepted provider/hash; therefore they cannot form a `READY` or `EXECUTED`
manifest. A missing, reordered, duplicate, aliased, unexpected, or wrong-hash
binding never selects B0 or the full method.

Unavailable nonoptional rows form only explicit `NOT_READY` records with no
claimed binding. Unavailable optional B8 forms only `SKIPPED` with a nonempty
canonical reason. It cannot be omitted or relabelled.

## Canonical manifest

The exact top-level field order is:

```text
schema_version, product_commit, platform, sensor, baseline_id, config_switch,
execution_family, contribution_tested, deployable, privilege_class,
actor_input_fields, evaluation_only_fields, component_toggles,
capability_bindings, scenario_id, seed, actor_audit_status, execution_status,
skip_reason, manifest_sha256
```

`product_commit` is a caller-supplied lowercase 40-hex commit. Platform and
sensor are exactly `TASLAB_UAV` and `Livox MID360`. Scenario is a nonempty
canonical token and seed is a nonnegative exact integer. `READY` and
`EXECUTED` require actor audit `PASS`; `NOT_READY` and `SKIPPED` require
`NOT_RUN`. The SHA-256 covers the canonical ASCII JSON bytes of every preceding
field. Parsing rejects duplicate/missing/extra/reordered keys, noncanonical
bytes, nonfinite values, mismatched row data, and hash tampering.

`deployable` and each `component_toggles` member (`history`, `imu`, `anchor`,
and `ics`) must be an exact JSON boolean before its value is compared with the
canonical row. JSON integers `0`/`1`, floats, strings, nulls, arrays, and
objects are rejected even where Python equality would otherwise equate a
numeric value with `False` or `True`.

The frozen record owns only canonical JSON text. Returned mappings are fresh,
so caller mutation cannot change record bytes or registry state. No candidate
ID, moving HEAD, path, timestamp, or implicit default is serialized.
