# M7-008 Revision 1 Import-Boundary Rework Known Limitations

- The execution boundary remains unintegrated with environment, controller,
  runtime collector, scenario, simulator, and metric aggregation.
- No baseline or scenario was executed; no performance claim is made.
- B1-B8 remain non-executable.
- Tensor inputs/results remain deliberately outside canonical manifest payloads.
- D0002 qualifies only the exact dependency-boundary closure and unchanged pure
  behavior. It does not authorize runtime integration or a baseline run.
- D0001 remains rejected historical evidence and is retained byte-identically.
