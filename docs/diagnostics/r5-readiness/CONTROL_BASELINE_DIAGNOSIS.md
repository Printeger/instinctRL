# Current control baseline diagnosis

Status: READY_FOR_PAIRED_PILOT_WITH_FIRST_ROLLOUT_GATE. Independent Role15 PASS accepts repaired zero-target physics and separate original command Z/T/S. Current source `ef68ebb5682b85aa83044fdea1c13b7d0edbf325`, job `J-ad3ac7db1fb82d89`. [Final review](reviews/review-entry6-final-readiness.json). Historical pre-repair measurement below retains its original scope and failures.

Historical B0 contains 20,160 verified real transitions; it was not an unrun template. The first and second Goal failures below remain separate immutable evidence. First baseline-v3 failed before its first accepted row. Measurement-r1b (source 79ccb40d52a347f0f17466c7f0bdbac8206354c1, job J-fcbe37739d942e3c) produced 128,640 valid physics rows: complete zero-target Z (200 policy steps, 6,400 completed transitions), plus all 20 integrations of the first attempted T step before reward rejected reliable coverage below 0.1. T/S protocol was not completed. The primary failure was reward coverage, not timeout. Cleanup then required bounded forced termination; lifeguard independently reports quiescence/no survivors. Each failed task retains its full 20,160 reservation. Training transitions, optimizer updates and training checkpoint writes are zero.

## Actual plant and source mechanism

The original immutable raw stream SHA256 is ad9aacb7a659cfc6095844fa224afc2e7ed37622862eb0cdeb86907cdee810b4. Its location and all small evidence hashes are in live-measurement-r1b-failure/INVENTORY.json. Offline analysis validates all 128,640 rows against the actual manifest without granting full protocol coverage. See measurement-r1b-partial-mechanics-analysis.json and tools/analyze_r5_partial_mechanics.py; segments separate experiment, zero-target condition, case, lane and effective reset episode. Initial five seconds remain in the raw and formal denominator. Momentum differences do not cross resets.

Actual per-body PhysX masses are base 1.158307433128357 kg, rotor_0/1/2 each 0.003275499911978841 kg, and rotor_3 0.013383990153670311 kg. The sum is 1.1815179230179638 kg. The current full composed asset and frozen runtime are unchanged. The previously unresolved authored rotor_3 mass is now resolved by actual PhysX, not assumed zero. The pre-repair Lee source uses 1.1583 kg, a base-scale constant rather than actual whole-vehicle mass. Its unchanged velocity gain predicts -0.08938170547990508 m/s vertical equilibrium; the real steady sample is -0.08938680589199066 m/s.

Horizontal bias is supported by the actual wrench geometry, not an assumed COM explanation. The old nominal mixer uses nominal symmetric arms about root. Actual thrust applies along rotor local Z at each rotor link origin; reaction applies at base local Z. Measured rotor application points relative to whole-system COM do not match those nominal columns. Their center differs from root as well as from whole-system COM. At real zero-target steady motion, the body is almost vertical, net horizontal rotor force and net COM torque are almost zero, while controller desired attitude remains tilted. The nominal allocation bias cancels the attitude correction at a nonzero velocity.

Solving the *original unchanged* Lee steady torque-x/y and vertical-force equations using measured rotor geometry, starting from zero velocity with no gain fitting, predicts world velocity [0.19355434828183254, -0.055579059498888195, -0.08938204506107636] m/s. The preserved corresponding actual sample is [0.19352911412715912, -0.05558815598487854, -0.08938680589199066] m/s. Equation residuals are below 6e-17. This quantitative source/plant prediction supports a specific allocation mechanism; it is not a heuristic causal winner and does not establish that all R4 failures have one cause. Generic signature analyzer rankings remain inconclusive evidence.

## Minimal candidate repair and falsification

Production flight control reads actual positive five-body masses and actual link/COM transforms, computes total mass and the rotor wrench allocation about measured whole-system COM, and solves the existing desired torque/collective wrench with this measured matrix. It changes no plant mass, COM, asset, gains, velocity feedback convention, action mapping, actor input, reward, curriculum or algorithm. Unavailable or invalid measurements fail explicitly, with no silent base-mass fallback. Standalone legacy controller fixtures keep their original default for historical calculations; the production environment always supplies both measured inputs.

Independent CPU wrench reconstruction using preserved real geometry verifies that the corrected allocation produces near-zero COM torque for a pure collective wrench; the old mixer produces a substantial roll/pitch torque with the same physical geometry. This is a counterfactual arithmetic test, not live acceptance. The repaired zero-target trajectory and separate fixed-command B0 through governor/ICS/Lee/rotor have now passed independent actual regression; original Z/T/S metric-h1-v1 gates are unchanged. Startup, all 32 lanes, reset and safety failures remain included. Any remaining unacceptable low-speed drift or tracking blocks pilot admission; no new position hold or relaxed thresholds are introduced.

## Diagnostic lifecycle repairs

The actual first-T failure follows a bare _reset_idx that clears policy/history without rebuilding valid sensor observations. Public reset performs scene write/kinematic forward and actual observations with no physics integration. The candidate uses it for all three cases; only Z retains the original seed, later cases are not reseeded. No hidden warmup is added.

Observed runtime cost also warrants a bounded diagnostic implementation repair: old probes copy/hash the full roughly 32.6 MB actor history twice per integration. New private leaf cache reuses SHA256 only after exact GPU byte equality, including signed zero and NaN bit patterns; changed bytes, even through Tensor.data without a version increment, force rehash. Other protected state probes remain strict. Primitive physics serialization batches small transfers instead of synchronizing hundreds of individual scalars. These diagnostic changes do not advance controller, history, RNG, simulator or reset state. CPU faults and regressions are preserved in plant-r1-cpu-final and plant-r1-cpu-final-v2.

The native app.close boundary remains a bounded forced-cleanup failure, not graceful close success. A completed scientific trace, safe ownership settlement and independent evidence judgment must be recorded separately from process status. No old FAILED job is resumed; no next live is released merely by this report.

## Actual repaired physical regression

Fourth source f726c019339494e74fbc6e7c90ec6d0fa7a45ff8/jobJ-afe4bda51f5206e5 provides403200 independently validated physical rows,96reset boundaries and96condition/case/lane/episode segments. No physics startup rows are removed. RawSHA c89e44bbf0039e9c647f4232d4029ab00f446b81ba775790d9097c4157ffb92d. Root analyzer and independent Reviewer fullstream scan agree exactly on coverage/statistics; Reviewer also samples288primitive COM/rotorwrench reconstructions.

| Zero-target case | Maximum lane RMS speed, all physics (m/s) | Max root horizontal displacement from first physics row (m) | Max height change from first physics row (m) | Max absolute after5s mean xyz velocity across lanes (m/s) |
| --- | --- | --- | --- | --- |
| Z | 0.0113864287 | 0.0067021551 | 0.0483250618 | [.00033210,.00018148,.000000877] |
| T | 0.0161006683 | 0.0036493149 | 0.0483250618 | [.00039035,.00034340,.000000840] |
| S | 0.0088686329 | 0.0090503361 | 0.0483250618 | [.00025686,.00024047,.000000887] |

Measured mass remains1.1815179230179638kg; late collective11.5906896–11.5906908N matches actualmass×9.81=11.5906908248N. The previous persistent xyz bias disappears under the measured mass/actualallocation intervention without plant/gain changes. This jointly verifies changed intermediates and physical response, supporting the specific originalsource mechanism. Startup maximum instantaneous speed0.1003273m/s and height drop0.0483251m remain included. These diagnostics use root-first-physics reference, not original mounted-sensor T0; they do not claim formal command gates.

Process verdict remains FAILED/137: expensive CPUanalysis exhausted supervisor deadline, actual910.5478907090146s/GPU909.4390038719866s charged. OriginalPREPAREDmanifest and missingfinalreward counters are immutable; full20160 remainsreserved. Fullphysicalcoverage cannot prove final completedreward policy count. Existing-tool supplemental known-owned-identity closure/twoGPUaudits are separate from missingnormalcloseack. Minimal durability/seed successor ef68ebb preserves allphysicalproductcode; its exact independent final acceptance is recorded in Role15. Generic analyzer hypothesis ranking remains no causal conclusion. Drag/contact/joint/other individualwrenches stay unavailable; their separate attribution and alloldR4failures remainunknown, and do not block the current independently accepted zero-target plus fixed-command admission.

## 真实命令直通回归（ENTRY6）

准确 ef68 基线，J-ad3ac7db1fb82d89；原 Z200/T100/S330、32 lanes、mounted-sensor T0 和所有启动／安全样本保留。原 metric-h1-v1 的 Z/T/S 各 32/32 PASS，d0–d7 各 4/4。Z 最大 speed RMS 0.010481975m/s、drift 0.006700954m、高度变化0.048525572m；T 最大全窗 RMSE0.037160032m/s、稳态0.000572826m/s、vz RMS0.000323263m/s、高度0.053401828m；S 全 lane stop_time1.5s，最大稳态 drift0.007121895m，刹车位移0.095374950m、zero-block总位移0.099505298m保留。完整逐 lane 数值及原 gate 在 entry6-command-original-gates.json，范围在 entry6-command-regression-summary.json。

这证明固定命令经原 governor/ICS/Lee/rotor 路径可执行；不把未恢复先前位移误说成位置外环职责，不向 actor 提供位姿或实际速度。未观察到需要额外 D7-R 来解释的下层跟踪失败；旧 R4 策略仅作旧基线历史，下一次真实 rollout 概率 gate 仍强制。

进程因 app.close 9.5s 超时被终止，FAILED/-9、原 status PASS_PENDING_CLOSE；完整现场计数630/20160不改写成 FINAL。H1 collector INCOMPLETE/null，20160全额预留保留；正常 lifeguard ack 两次稳定空扫描，无所属残留。真实控制 gate 与进程/清理判定分开。最终独立 Role15 审核 PASS，接受科学控制、完整证据及清理／保守计账边界；原生关闭失败仍保留。
