# R5 zero-command attribution

Finite CPU reanalysis of already independently accepted real P0/PLR data; no simulator, optimizer, GPU query or physical rerun. Raw compact logs are rehashed against original inventories. All source/history statuses remain unchanged.

| group/update | issued residual xyz (m/s) | recorded actor-mean residual xyz | old reward mean | proposed extra mean | actual reset lanes |
| --- | --- | --- | --- | --- | --- |
| P0/u1 | +0.005608, -0.027010, +0.064662 | +0.001574, -0.029942, +0.064336 | -0.149519 | -0.159457 | 0 |
| P0/u2 | +0.098602, -0.013430, -0.166357 | +0.093048, -0.015109, -0.169215 | -0.259529 | -0.315221 | 0 |
| P0/u16 | +0.106896, -0.106551, -0.132509 | +0.104014, -0.106922, -0.133827 | -0.712093 | -0.325424 | 0 |
| P0/u64 | +0.076008, -0.095211, -0.022935 | +0.073808, -0.098754, -0.026390 | -3.319888 | -0.384367 | 1 |
| PLR/u1 | +0.005608, -0.027010, +0.064662 | +0.001574, -0.029942, +0.064336 | -0.149519 | -0.159457 | 0 |
| PLR/u2 | +0.010990, -0.020810, +0.075922 | +0.006343, -0.022386, +0.074446 | -0.212119 | -0.143863 | 0 |
| PLR/u16 | +0.056013, -0.010690, +0.047259 | +0.052793, -0.010716, +0.051934 | -1.150883 | -0.139571 | 0 |
| PLR/u64 | -0.013493, -0.055586, +0.005060 | -0.013642, -0.054654, -0.000811 | -0.806160 | -0.123418 | 0 |

PLR u64 real residual y=-0.055586 m/s and actual body velocity y=-0.057399 m/s with ICS beta=1 support a proximal residual-bias hypothesis. XYZ signed means, RMSE, quantiles, world/body velocity, anchor active/usable/loss, all components and within-reset-continuous lane segments are in the JSON. Actor means are recorded conditional means; sampled actions are distinct. Commands are zero in every selected window. Action decode equals the existing correction within6e-8.

Segment displacement starts from the first returned observed row of a finite16-step window, not original Z sensor T0 or a complete episode. No cross-reset difference. Control is staged before integration; response velocity/root is afterstep, dt0.1s. Correlation includes inertial lag and does not establish complete closed-loop causality or prove COM caused horizontal drift. Original physical asset/mass/control acceptance is reused.

Real first-update8minibatch gradients/clip/deltas and actual pre/post conditional BetaKL are retained. Saved u1/u2 advantages/returns are aligned to actual oldBeta/actions. The JSON distinguishes output-space score covariance from a reconstructed parameter counterfactual. u16/u64 actual sampling conditional Beta tensors were not saved: aggregate approximateKL and recorded means remain available, exact conditionalKL unavailable; no invented values. A fixed-input CPU checkpoint probe can measure checkpoint migration separately if needed, never masquerading as those rollout distributions.

The original reward already penalizes actual velocity, ray anchor error and command variation. In zero-command windows the governor/final/preservation gates are off and zero_extra is exactly zero; direct residual cost is the specific objective gap. Cumulative fixed-reference Z drift is not identical to instantaneous velocity/ray-anchor costs.

Decision: supported, proceed with one preregistered candidate w=0.05. Its PLRu64 penalty mean≈-0.123418/step is within meaningful old reward scales; 0.1m/s costs0.05, signed axes are symmetric. This is a supplied candidate scale, not an optimizedweight. C0w0/C1w.05 shareactorLR1e-4. No weight selection after training, no hard override. The direct commandB0 originalZ/T/S32/32 remains accepted historical capability, not a newrun.

Limits: one pair seed; selected finite training windows differ from deterministicZ; actor update drives later distribution and no exact unrecorded counterfactual. No claim that all secondary unknowns or historic R4 causes are resolved.

Machine evidence: R5_ZERO_COMMAND_ATTRIBUTION.json SHA256 2efdc11d645b194b8bfdd3de6c18ad71763c18b45826f7df629e1baf2d83654e; analysis tool 143ee76dd9db5b82571f576c5e9b6d1fc5d2402b05646018f62b81be98f4529a.
