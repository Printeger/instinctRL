# M1B Known Limitations

- This phase provides no Isaac Lab articulation configuration, robot
  registration, actuator, controller, spawn, reset, physics, or numerical
  flight parity. Those semantics remain explicitly unresolved.
- Exact-copy vehicle YAML lines 29, 46, 54, 67, 75, 84, 89, and 93 retain
  source-owned trailing whitespace. Revision 1 requires those diagnostics to
  remain visible while all authored paths pass the artifact-excluded check.
- The byte-preserved USD contains original generator documentation metadata
  naming an absolute legacy NavRL source path. Read-only dependency inspection
  proves it is not a root-layer external reference or external asset path; it
  cannot be removed without violating the byte-identity requirement.
- Standalone dependency enumeration reports `OmniPBR.mdl` as unresolved while
  the stage itself opens successfully. This is the asset's built-in material
  identifier, not an absolute legacy/formal/coordination/frozen-repository
  dependency. No asset was downloaded or rewritten.
- USD MassAPI reports a base-link mass of `1.158307433128357 kg` versus the
  active YAML's `1.1583 kg`. Authored diagonal-inertia attributes are zero and
  later physics-derived inertia was not evaluated because physics initialization
  is forbidden in M1B. No new physical authority was selected.
- Headless shutdown uses the Order-permitted forced process exit only after the
  normalized result is written, printed, and both streams are flushed.
