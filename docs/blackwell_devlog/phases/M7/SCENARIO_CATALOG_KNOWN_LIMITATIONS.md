# M7-013 Canonical Evaluation Scenario Catalog Known Limitations

- All ten scenario rows are intentionally `NOT_READY`. No geometry, terrain,
  obstacle, asset, generator, command schedule, reset/termination contract or
  accepted scenario evidence exists in this Order.
- `dynamic_obstacle` is optional when available, but its absence must remain an
  explicit reason; it cannot silently disappear or fall back to another row.
- The two real-robot identities do not imply hardware access, repeatability
  evidence or safe execution authorization. `real_robot_repeatability` is only
  a later-unavailable metric-name obligation.
- Run-plan binding proves catalog/manifest identity only. It deliberately
  reports `executable=false` even when the independently accepted baseline
  manifest has READY or EXECUTED baseline status.
- Metric obligations contain names only. They supply no formulas, inputs,
  availability masks, values, sample counts, denominators, thresholds,
  rankings, comparisons or performance claims.
- Privileged generator fields are declared for privacy auditing but no
  generator is implemented. Only commanded body velocity may cross the future
  generator-to-actor boundary.
- The two-process fixture uses one exact catalog selection solely to prove
  deterministic identity serialization; it is not a scenario seed list,
  command fixture, campaign or relabelled M7-012 smoke.
- No CUDA, simulator, GUI, ROS, real robot, scenario execution, baseline
  comparison, training, M8 or scale work is included or authorized.
