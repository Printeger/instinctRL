# Zero-command baseline diagnosis and validation report

Final decision: `DIAGNOSIS_INCONCLUSIVE`

Date: 2026-09-27 (Asia/Shanghai)

## Plain-language conclusion

The downward drift has a clear deterministic mechanism. The Lee controller computes hover force with the 1.1583 kg base mass, while the simulator plant's frozen runtime evidence says the complete articulated vehicle weighs 1.1815179586 kg. Because the velocity loop is proportional-only and has no integral term, the aircraft must maintain a small downward velocity error to request the missing force. The predicted steady velocity is -0.0893818425 m/s; the three B0 pre-reset windows average -0.0893858326 m/s, only 0.0000039901 m/s apart.

That result does not explain the simultaneous horizontal drift of approximately +0.1934/-0.0559 m/s. The available aggregate policy-step trace has no articulated-COM/per-body transforms and no decomposition of gravity, rotor, drag, contact, joint-constraint, and other plant wrenches. It also does not sample rotor/controller state at the reset instant. A z-only mass edit would therefore be a partial, speculative repair under the task contract. No product code was changed, and the authorized live B0 gate was not entered.

## IMU-noise finding

IMU random noise is not supported as the direct cause. Static inspection shows that this B0 path feeds the Lee controller from Isaac articulation `root_state_w` velocity directly. No IMU value enters this control path. The 32-lane Z reset and the two deterministic 16-lane S reset cohorts are also consistent with a common deterministic plant/control bias, not evidence of independent sensor noise.

## Offline evidence and time series

The frozen B0 evidence was reused; no new transition was generated. The extractor verified all 630 source chunks and produced 630 contiguous policy steps, 32 lanes, and 20,160 historical transitions:

- Z: 200 steps; all 32 lanes cross below-height/reset at step 166. Step 166 has no downstream actuation fields and was retained as missing evidence, not zero-filled. Steps 167-199 exactly replay steps 0-32 over the audited control and plant fields.
- T: 100 steps; no reset.
- S: 330 steps; 16 lanes reset at step 165 and the remaining 16 at step 166. Thus all 32 reset across the adjacent boundary, but the raw trace does not support the earlier statement that all 32 reset at S step 165.

The full machine-readable series is `task_inputs/b0_control_chain_time_series.json`, SHA-256 `d775d2f073f829c1ecb3bcd73d24a1b7626c90ede2bf629e19cd9ccddb75de28`. It contains per-step all-lane mean/min/max/std for velocity, attitude, height/position, desired attitude, collective, normalized rotor commands, target/actual throttle, rotor thrust/moment, applied wrench, held position, resets, and finite-difference acceleration.

Pre-reset steady windows were Z 146-165, T 80-99, and S 145-164. Their mean velocities were respectively:

- Z: `[0.19342518, -0.05584377, -0.08938576]` m/s
- T: `[0.19361145, -0.05546098, -0.08938582]` m/s
- S: `[0.19340206, -0.05591909, -0.08938592]` m/s

In all three windows, requested collective and summed rotor/applied thrust converged to approximately 11.590691 N, the 1.1815179586 kg articulated weight. Attitude and reconstructed acceleration approached zero while the nonzero velocity persisted.

## Control-chain audit

- Zero target to Lee: B0 target z is zero. Lee reads root state directly, uses gravity 9.81 m/s², velocity gain 2.2 s⁻¹, and controller mass 1.1583 kg.
- Desired attitude and collective: the steady vertical error supplies the exact proportional force needed to raise collective from controller weight 11.362923 N to plant weight 11.590691173866 N.
- Mixer, directions, frames, and signs: the static mixer sends positive collective to four local +Z thrust slots; reaction moments alternate consistently. The trace does not reveal a mixer/sign inconsistency.
- Rotor command, throttle, and thrust: all 629 valid downstream steps pass normalized-command/target-throttle, actual-throttle/thrust, rotor-thrust/applied-force, reaction-moment, and collective-allocation checks. Evaluator maxima were 5.60e-05 N for mean-of-square throttle relations, 0 N for rotor-to-applied force, 1.93e-09 N·m for moments, and 4.51e-07 N for allocated collective.
- First-order response: `tau_up=tau_down=0.43`. The first recorded policy sample is already after multiple physics substeps and actual throttle is near target, so the trace cannot observe the instantaneous zero-throttle initialization.
- Reset state: source defines resettable throttle state, and exact Z post-reset replay supports effective reset behavior. Instantaneous pre/post-reset internal state remains unobserved.
- Held position: it is emitted for evidence only and is absent from the Lee feedback equation; this is a velocity hold, not a position loop.
- Integral action: none. The mass mismatch therefore creates a stable nonzero vertical velocity offset.
- Shared versus separate cause: the vertical mechanism is established. It predicts no x/y bias, so a common x/y/z defect is not established and horizontal causality remains separate and unresolved.

## Candidate and changes

Isolated repository: `/home/mint/rl_dev/instinctRL_orchestrator/implementation_records/instinctrl-first-use-20260915/state/workspaces/instinctrl-zero-command-baseline-20260927`

- Prepared/base/head commit: `5c4a5e9dfba0cdebfb3478a3f32dc36962337cdb`
- Git tree at that commit: `a6126cb92b76bc6d5f89a5987d18925863e24233`
- rlflow accepted candidate tree SHA-256 (includes the seven untracked diagnosis/test artifacts): `091af87fd4efcd01f6e99166e793cb4b04473b003a2b814cf0fd4905b0689d4d`
- rlflow candidate diff SHA-256: `4ae286207adcc60ca99b0cff0838dc657d622e7b5dc82d1e5517a86496de976a`
- Product tracked diff from the prepared baseline: empty.

Added diagnosis/test artifacts only:

- `zero_command_delivery/analyze_b0.py`
- `zero_command_delivery/diagnosis.json`
- `zero_command_delivery/REPORT.md`
- `zero_command_delivery/changed_files.json`
- `zero_command_delivery/live_validation_handoff.json`
- `zero_command_delivery/test_record.json`
- `tests/training_workloads/test_zero_command_diagnosis.py`

These exact bytes were independently reviewed as an rlflow candidate. They were deliberately not committed after review, because changing HEAD would change the reviewed candidate identity. No actor input, reward, method, acceptance metric, threshold, seed, curriculum, reset semantics, or frozen history changed; no simulator truth or explicit velocity was added to the actor.

## Tests and independent review

Engineer and independent Evaluator both ran:

`/usr/bin/python3 -m unittest -v tests/training_workloads/test_zero_command_diagnosis.py`

Result: 5/5 passed. A final outer replay also passed 5/5 in 0.044 seconds. Tests cover trace hash/coverage, reset boundaries, mass prediction, force-chain/collective consistency, repair gate, and frozen contracts.

Evaluator session `S-9cb2f28e6bb247da` returned `PASS` at the CPU layer with `missing_evidence=[]`. Its PASS explicitly accepts the terminal diagnosis package, not a code repair or real-run behavior. It retained two `FOLLOW_UP` findings: unresolved horizontal causality and missing instantaneous reset-state observation.

The formal rlflow package `INSTINCTRL-ZERO-COMMAND-BASELINE-20260927` used exactly three role calls (`PI`, `Engineer`, `Evaluator`), then was paused with task `T-30b901b95e0b5eb1` at `CONTRACT_DECISION`. The scientific review passed, but the control plane could not authorize the seven artifact changes because Evaluator suggested the unsupported semantic label `ENGINEERING_DIAGNOSIS` whereas the current classifier accepts only its fixed change categories. No fourth call, rework, or retry was made. This control-plane classification detail does not turn an incomplete diagnosis into a repair or authorize live execution.

## Live validation and budget ledger

No new live validation was run because the mandatory complete-root-cause and minimal-repair gates were not met.

- New live argv: not applicable
- New job/evidence ID: not applicable
- Live process starts: 0 / 1
- Simulator initializations: 0 / 1
- New nontraining transitions: 0 / 20,160
- Historical B0 transitions read offline: 20,160
- Training transitions: 0
- Optimizer updates: 0
- Checkpoint writes: 0
- P0/PLR registrations or submissions: 0
- Z policy evaluations: 0
- Automatic retries: 0
- GPU compute used by this task: 0
- rlflow role calls: 3 / 3
- rlflow summed role elapsed time: 714.855 s
- rlflow summed model-active time: 708.265 s

Cleanup commit `02a45b44fc5ec8facd31ad3c4fbbfb15ece977f1` was present in the prepared baseline but was not revalidated in a real run, because starting that run would have violated the live gate.

## Resource cleanup

At 2026-09-27T16:59:57+08:00, the process audit found no Isaac simulator, `SimulationApp`, InstinctRL live workload, rlflow daemon, worker, or lifeguard process. `nvidia-smi` showed no task compute process; it showed only the unrelated desktop service `/usr/libexec/gnome-remote-desktop-daemon` using 164 MiB. No task GPU allocation remains.

## Decision and unresolved risks

P0/PLR is not allowed. The final decision is `DIAGNOSIS_INCONCLUSIVE` because horizontal drift remains causally unresolved and reset-instant internal state is not observed. The cleanup fix also remains unproven in a new real run.

The minimum next step is one separately authorized, no-retry B0 Z/T/S diagnostic with unchanged seed, initial state, curriculum, actor/control contracts, and frozen safety thresholds. It should add only per-physics-step, per-lane articulated-COM state; per-body mass/COM transforms; gravity/rotor/drag/contact/joint-constraint/other wrench decomposition at the COM; and rotor/controller state immediately before and after each lane reset. That future diagnostic is not authorized by this report.
