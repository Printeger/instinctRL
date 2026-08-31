# M7 Gap/Doorway B0 Known Limitations

- This is a deterministic B0 centerline direction and sensor-visibility
  witness. It does not execute or qualify doorway traversal.
- Strict centerline aperture intersection does not prove vehicle-body or rotor
  clearance.
- The fixture does not measure avoidance, braking, governor/ICS effectiveness,
  collision prevention, safety, policy quality, performance or ranking.
- The separate PlaneMesh is preserved and audited as topology; it is not a
  claimed same-scan hit source for the selected frame Mesh.
- Only seed 7020, two environments and eight prescribed command rows are in
  scope. No B1-B8, dynamic-obstacle, real-robot, training, M8 or scale result is
  established.
- Normal shutdown emits accepted headless Isaac deprecation/windowing messages
  on stdout; formal command stderr is empty and all formal gates pass.
