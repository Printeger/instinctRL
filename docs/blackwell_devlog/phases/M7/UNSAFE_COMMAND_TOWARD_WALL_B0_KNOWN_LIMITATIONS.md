# M7-019 unsafe-command-toward-wall B0 known limitations

- This is a deterministic two-environment, eight-policy-row, B0 wiring and
  observability qualification, not a learned-policy evaluation.
- Positive command toward the wall is deliberately unsafe as an input
  condition. No avoidance, braking, intervention effectiveness or collision
  prevention behavior is implemented or claimed.
- The runner observes controller-private pose only inside evaluation to prove
  body-to-world direction and signed distance. That state is not an actor
  input; actor and training calls are both zero.
- Results apply to the frozen RTX 5070 / driver 580.173.02 / Python 3.11.15 /
  PyTorch 2.7.0+cu128 / Isaac Sim 5.1 / accepted Isaac Lab environment.
- The test covers one static bounded x-wall and does not cover doorway,
  dynamic-obstacle, real-robot, B1--B8 or scale-training behavior.
- D0001 and D0002 remain immutable blocked evidence. D0003 does not erase or
  reinterpret their failures; it supplies fresh complete evidence after the
  ordered numerical-oracle and comparator-self-test corrections.
