# M2 Known Limitations

- M2 proves only a single articulation can spawn, reset, step, update, and
  replay deterministically under gravity for 32 steps. It does not prove hover,
  rotor thrust, controller, tracking, sensing, or learning behavior.
- The zero-gain implicit actuator only registers the four existing free rotor
  joints. It is not a rotor model and issues no targets.
- The YAML base mass (`1.1583 kg`), USD/PhysX base mass
  (`1.1583074331 kg`), and runtime total articulated mass
  (`1.1815179586 kg`) are different quantities and remain distinct.
- PhysX reports the fifth rotor body effective mass as `0.0133839902 kg`
  although the USD's authored mass field for that prim is zero; this effective
  value is recorded, not retuned or interpreted as new source authority.
- Headless startup emits expected display and IOMMU warnings plus an existing-
  prim warning during USD attachment. They do not prevent initialization or
  affect the deterministic normalized result.
- Shutdown uses `os._exit` only after JSON and streams are flushed, matching
  the baseline-known headless close workaround.
