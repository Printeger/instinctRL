# R5 zero-command residual-neutrality: complete actual pair

**Goal scientifically complete: VALID_EXPERIMENT_POSITIVE, independently accepted Role6 PASS with no MUST_FIX.** The preregistered u64 comparison improves speed, drift and height in this one paired seed. Original Z admission still fails: C1 u64 is 0/32 because every lane exceeds 0.30 m XY drift. No 256-update extension or further live work is authorized.

| group/node | Z successes | maximum speed RMS (m/s) | maximum XY drift (m) | maximum height deviation (m) |
| --- | --- | --- | --- | --- |
| C0/u16 | 0/32 | 0.060848549 | 0.828733861 | 0.817441702 |
| C0/u64 | 0/32 | 0.102369525 | 1.959831357 | 0.471305609 |
| C1/u16 | 0/32 | 0.051088054 | 0.783498526 | 0.566943645 |
| C1/u64 | 0/32 | 0.020412797 | 0.347792268 | 0.095134258 |

Original thresholds remain speed ≤0.08 m/s, XY drift ≤0.30 m, height deviation ≤0.20 m. Each node retains sensor-T0 plus all 200 steps ×32 lanes, original startup/safety checks and all d0–d7 subgroups (each 0/4). All four nodes have 32 valid engineering episodes, zero missing, zero reset interruptions, zero collisions and zero engineering height failures; threshold failures remain failures. No exclusions or reset splicing. `pair_analysis.complete.json` retains every lane, metrics, failure reasons, raw paths/SHA and subgroup denominator.

At u64, maximum speed falls 0.102369525→0.020412797 m/s, drift 1.959831357→0.347792268 m and height 0.471305609→0.095134258 m. All C1 u64 lanes satisfy speed and height, yet drift ranges 0.325606376–0.347792268 m. Hence credible within-pair behavioral improvement does not grant original Z admission. u16 is diagnostic: C1 improves its maxima, but still fails drift and height. Scientific validity is assessed from these preregistered numerical comparisons and original gates, not a heuristic score.

Both groups use the same exact read-only d257f9ca8ec8639627d3a025c9352b9d29883d90 source, config4746f5..., method ecc75..., seed20260909, actor LR1e-4/critic3e-4, PPO/network/actor inputs/controller/ICS/curriculum/termination/evaluation. The only method difference is zero-command residual-neutrality w=0 versus .05. Each completes64 updates/32768 training transitions/7 checkpoints and512 Adam steps per model. Both actual first512 gates pass before either optimizer; actual actor/critic init, policy/privileged inputs, actions, old Beta parameters and logp are bit-exact equal. Returns/advantages differ as allowed by reward. Role3/4 independently accepted training; w0 equivalence and C1 formula all32768 samples are verified (max formula error1.008133e-7).

Actual u64 sampled training correction means (body xyz m/s): C0[+.075160322,+.042557184,+.021312151], C1[-.006825240,+.009512214,-.002580675]; actual training velocity means respectively[+.075863041,+.040929950,+.010332806] and[-.006244662,+.009022830,-.004768509], ICS beta=1, no resets in these512-sample windows. Actual deterministic u64 Z body velocity means are C0[+.081183822,+.053548574,+.023115524], C1[+.002052978,+.016287764,-.004397990]. These are separate trajectories and timings. Original evaluation does not log v_corr; its value is UNKNOWN, never replaced by training actions or inferred from actual velocity. Body/world, signed xyz, sampled/conditional mean distinctions and reset-continuous attribution remain in R5_ZERO_COMMAND_ATTRIBUTION.json and the machine pair result.

First update mb8 analytic Beta KL is C0=.276194468 and C1=.196501259. Per-minibatch parameter shifts, gradient norms, conditional mean residuals and per-update gradients are preserved in the complete pair; reward distributions/advantages/returns and all64 update diagnostics remain in the two training analyses/raw logs. Historical P0/PLR lower actor LR reduced first KL15.012778→.276192; PLR u64 met speed/height but drift1.088824m failed. Historical PLR is context only: new C0 is the strict contemporaneous control, and C0 differs numerically from historical PLR. One paired seed does not establish statistical significance, generalization or final causality.

The evidence supports the proximal residual-bias mechanism and the utility of the soft residual cost in this pair; it does not show residual bias is the sole cause of every drift, prove COM effects, or establish a paper-level innovation. It does not validate full R5 Z/T/S/X/W, tracking or obstacle behavior.

## Execution and retained failures

| entry | job | native / raw / formal | observed transitions |
| --- | --- | --- | --- |
| C0 training | J-08bc17b874c978ea | FAILED/-9 / PASS_PENDING_CLOSE / INCOMPLETE,null | 32768 train |
| C1 training | J-3083020018e55362 | FAILED/-9 / PASS_PENDING_CLOSE / INCOMPLETE,null | 32768 train |
| C0 Z | J-4be233e84359e5c5 | FAILED/-9 / PASS_PENDING_CLOSE / INCOMPLETE,null | 12800 eval |
| C1 original entry | J-c28f721ecfa2d171 | FAILED/1 / FAIL / FINAL,0 | 0; before App |
| C1 extra Z | J-ca72975b201fceb1 | FAILED/-9 / PASS_PENDING_CLOSE / INCOMPLETE,null | 12800 eval |

The old C1 failure (GPU samples4/6/2%, max6>original5%) is preserved, never resumed or relabeled0/32. The user explicitly approved one new entry, starts/init4→5 and output reservations64→80GiB only. New approval A-859f34b1f384dcdb, AUTH9d732b..., input commit f400f95 and pushed input tag precede release. The new original coexistence gate passed samples5/2/3%; no unrelated process killed or threshold relaxed. Actual u16/u64 checkpoints are exactly the Role4-accepted hashes. This extra entry contains zero training/update/checkpoint writes.

All five entries have normal owned ACK, two stable empty scans/no survivors and exactly two bounded postrun audits. New complete inventory contains40 originals/3118695 bytes plus equal archive copies; the previous61+61+40+27 originals remain immutable. app.close timeout/-9 is preserved. Four INCOMPLETE entries retain full65536 training+25600 eval+128 logical updates+14 writes reservations. Only the old FINAL0 entry releases unused eval transitions; its start/init/output reservation remains charged. Old diagnostic and P0/PLR budgets/native FAILED jobs are untouched.

Current charges:5 starts/init, observed65536 training/25600 eval/128 updates/14 writes,1024 Adam steps per model; wall848.172850726s, GPU844.968363575s. All5×16GiB output reservations remain (80GiB); actual bytes and metadata are separately reconciled within them. Role6 final independent review is charged before its call; no automatic retries.

The immutable prior blocked result is pair_analysis.blocked.json; original failed C1 analysis remains C1-evaluation-data-analysis.json. New C1-extra-evaluation-data-analysis.json binds the exact new authorized spec, unique actual job, terminal receipt, cleanup and raw evidence; pair_analysis.complete.json is the complete numerical artifact. Canonical pair_analysis.json links the current interpretation and historical evidence. Final independent review is archived in reviews/review-zero-final-scientific.json; COMPLETION_AUDIT.json records the whole-requirement closure. The numerical artifact retains its original pre-review status; this report and canonical index record final acceptance.

Final independent Role6 explicitly accepted the complete original evaluation, unique method comparison, scientific scope, budget/cleanup boundaries and all five research answers. No further live/GPU probe, training, weight search or256-update extension occurs. Final output is reconciled in OUTPUT_ACCOUNTING_CLOSURE.json, including review/audit metadata within the preallocated64MiB allowance.
