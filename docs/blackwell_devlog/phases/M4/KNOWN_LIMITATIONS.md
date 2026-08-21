# M4 Known Limitations

- This milestone defines and qualifies a pure-tensor observation boundary; it
  is not connected to an environment or actor network.
- Reliability implements only the ordered deterministic validity/age formula.
  Noise, random dropout, and neighbor consistency are intentionally absent.
- M4 does not estimate translational motion from history. Such reconstruction
  is explicitly forbidden at this actor boundary.
- Determinism is proven for two independent processes on the frozen RTX 5070,
  PyTorch 2.7.0+cu128 runtime; no cross-hardware bitwise claim is made.
- Zero-filled initial/reset slots are distinguishable through zero range and
  validity channels; no additional actor-visible initialization key exists.
- Exact equality to 40 m for invalid beams follows M3's serialized output
  contract and is intentionally stricter than approximate sensor arithmetic.
- No policy quality, control, reward, training, or performance claim is made.
