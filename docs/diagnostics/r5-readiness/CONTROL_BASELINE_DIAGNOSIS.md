# Current control baseline diagnosis

Status: PENDING_CURRENT_REAL_DIAGNOSTIC. No new real trajectory exists in this Goal.

Historical B0 has 20,160 verified real transitions from 630 hash-verified chunks. Steady vertical velocity agrees with historical base-vs-whole-mass prediction; horizontal +0.1934/-0.0559m/s is unresolved. It is historical evidence, not acceptance of the current asset/source. See historical-b0-verified-v1.json.

The composed current USD retains base mass1.1583074331kg and three authored rotor masses; rotor_3 mass/inertia requires actual PhysX resolution. Current total plant mass is UNKNOWN. No mass constant or plant mass has been changed. Source asset hash and composed layers: current-usd-mass-cpu-v2.json.

New readonly source643747a and exact v3inputpack1689d3c have independent pre-live PASS. This measures actual PhysX bodies, link response, controller COM feedback, rotor thrust and base reaction direction, reset/episode boundaries and per-lane response. Enhanced condition is ENHANCED_B0_ZERO_TARGET, with no policy checkpoint. It does not exercise nonzero T/S commands.

Required next observations: actual summed mass vs Lee control mass, vertical prediction/response, per-axis startup/steady/root/COM/episode behavior and desired/actual attitude/wrench. Signature ranking is a clue only. Horizontal cause is not presumed to be COM. No physical fix has yet been justified from current live data.
