# M7-016 corner B0 known limitations

- This is a bounded two-environment, eight-row migration-wiring
  qualification. It is not a flight-performance, safety, policy-quality or
  baseline-ranking claim.
- RayCaster selects only the first descendant Mesh under its one configured
  root. The two walls therefore intentionally share one ordered Mesh. The
  separate accepted PlaneMesh is qualified by USD topology and is not claimed
  as a same-scan corner hit.
- Runtime hit counts are exact for this frozen runtime, seed and command
  schedule; they are evidence, not source thresholds or portable performance
  expectations.
- The frozen IsaacLab checkout retains its accepted pre-existing modification
  to `apps/isaaclab.python.headless.kit`, SHA-256
  `0a0dee5fbbdcbf66f8124a07eced9a87c2f70e5df6ee892ec297444c1d1b01c2`.
  D0002 did not change it.
- No B1-B8, corridor, clutter, unsafe-wall, doorway, dynamic, real-robot,
  training, M8 or scale work was performed.
