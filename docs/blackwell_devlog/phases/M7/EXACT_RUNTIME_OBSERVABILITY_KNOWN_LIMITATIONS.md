# M7-005 Exact Runtime Observability Known Limitations

- Qualification is limited to the ordered two-environment, eight-policy-step static-plane smoke. It is not a dynamic/multi-geometry scenario qualification.
- The static plane supplies one independent normal direction, so every qualified row is rank one and exact `sigma_min` is zero. This is expected exact evidence, not a performance failure.
- Maximum observed accepted-distance reconstruction error is `7.62939453125e-06 m`, below the locked `1e-5 m` acquisition tolerance. Maximum accepted-hit error is zero.
- World rays are reconstructed from the frozen RayCaster public local `ray_starts`/`ray_directions` buffers and public pose. The evaluation resolver fails closed if the pinned commit/version/API/source hashes change; compatibility with another Isaac Lab version is not established.
- A finite hit rejected by the accepted MID360 minimum range fails closed. This order did not create a scenario exercising such a hit.
- Source-step identity is process-local and begins at one for the fresh authorized runner. Arbitrary mid-process collector attachment is not qualified.
- No proxy, reliability/staleness method, plots, drift correlation, thresholds, B0-B8 ablations, training, or M8 qualification is included.
