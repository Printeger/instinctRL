# M8 release-candidate known limitations

- The manifest is preparation for later bounded release qualification; it is
  not a release approval.
- Qualification execution is false. No B0 scenario was executed by M8-001.
- B1-B7 required capability adapters remain unavailable.
- B8 and dynamic-obstacle work remain optional and unavailable.
- Required real-robot evidence remains unavailable.
- A clean final candidate tree, reproducible import surface, bounded pipeline
  replay, B0 consistency and final release evidence closure remain later M8
  qualification gates.
- Scale training, merge and push require separate user authorization and are
  false in every valid manifest.
- The Isaac Lab project release identity `2.3.2` and installed distribution
  version `0.54.2` are recorded separately; neither was changed.
- The frozen Isaac Lab source checkout retains its pre-existing modified
  `apps/isaaclab.python.headless.kit` status; M8-001 does not alter it.
- No metric, threshold, checkpoint identity, ranking, safety or performance
  conclusion is present.
