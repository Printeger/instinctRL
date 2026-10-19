# W temporal diagnosis

Decision: **MIXED**. The retained traces support an insufficient control stopping margin; the accepted center-only training also lacks danger samples. First validate a single engineering margin change on the frozen R1/u512 policy, then train a near-wall distribution on that common control baseline. This is not yet a claim of safety or RL improvement.

All 10 original W files (2 arms × 5 nodes × 32 lanes) were checked against original inventories. All 10 checkpoints were rehashed, including R1/u512 `a3b9f9115268270ec000750035085b03980de2aed31987b7b08ae97bf26d8776`. W_CAUSAL_DIAGNOSIS.json contains every lane/event, exact paths and hashes, action residuals, issued output and world-normal actual velocity. No actor forward, simulation or GPU query was used.

| R1/u512 wall | Minimum geometry mean | β=0 geometry mean | Actual wall-normal velocity at β=0 | Further geometry loss after β=0 | Stop confirmation mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| x, 16 lanes | .713261 m | .768713 m | .164095 m/s | .055452 m | 3.13125 s |
| y, 16 lanes | .713931 m | .768060 m | .160785 m/s | .054129 m | 3.10000 s |

At the preceding observation before first attenuation, geometry is only .802634 m (x mean) / .801621 m (y mean). Every one of the 320 lanes has q before first attenuation and geometry below .8 m at the observation following the first β=0 command. Early u272 lanes have smaller inward velocities (~.05 m/s) and overshoot, but still violate .8 m. Thus failure is not merely a late-node anomaly or one wall coordinate bug.

Clock interpretation: row i contains control computed for step i at t=i*.1 s, and actual state/sensor geometry returned at t=(i+1)*.1 s. q belongs to the reward measurement path, not necessarily the exact ICS input frame. Previous-observation geometry is reported explicitly as a proxy; it is not called the true active-beam range. World-normal velocity comes directly from saved root_state_w[:,7:10]; it is root velocity, not the mounted sensor distance derivative. Body velocity is separately retained. Sensor-origin finite-rectangle geometry is recomputed with the original evaluator; recorded ray_clearance is a distinct quantity. Dense reliable ICS input beams and substep rotor dynamics are UNKNOWN here.

The R1/u512 worst original geometric deficit is .088452 m. Observed maximum inward velocity at u512 is .179071 m/s; one .1 s policy interval is less than .02 m at that measured speed. A proposed **.15 m clearance_margin_m** adds the observed deficit, a .02 m discrete-response allowance and approximately .04 m residual allowance, rounded upward. These are conservative allowances, not independently identified dynamics terms. It leaves d_safe=.8, the original evaluation and all other ICS parameters unchanged. This is an empirically motivated candidate, not a guaranteed bound on future motion. It must first pass fixed-policy real full-task regression, including actual approach motion, before serving as a shared training baseline.

The original evaluator permits the triggered branch without separately enforcing 11 incoming motion samples. The Goal explicitly requires motion; final acceptance therefore also checks the saved world/body approach sequence for all 32 lanes. The evaluator itself will not be changed.

Independent read-only review recomputed all 320 original score metrics and event indices, verified original source identity and accepted this diagnosis and the .15 candidate as conditional on real validation. The review relay is reviews/diagnosis-independent.json; training coverage recheck continues in the same Goal. No E0, mass or initial-hover diagnosis was repeated.
