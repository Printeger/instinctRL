# B0 enhanced live diagnostic — terminal report

## Final state

`LIVE_DIAGNOSTIC_FAILED`

Stage 1 did not establish the horizontal-drift root cause. Stage 2 and Stage 3 are not authorized and were not started.

## Plain-language conclusion

The one authorized run reached Isaac Sim initialization, but failed before the first B0 policy step. At the first explicit reset boundary the product diagnostic protocol could not provide the required controller/rotor reset state and raised `enhanced B0 controller/rotor reset state is unavailable`. Therefore it produced zero physics rows and zero transitions. There is no force/torque/COM time series from which to establish the horizontal drift cause, and no scientific inference is made from the empty trace.

The failure also exposed a cleanup defect: `app.close` exceeded its 9.5-second hard timeout. The watchdog killed the owner process, but an Isaac telemetry child escaped the process group and had to be terminated explicitly. Final process and GPU audits are clean except for the pre-existing GNOME remote-desktop process.

## Frozen identity and review

- Package: `INSTINCTRL-R5-STAGE1-HANDOFF-SEAL-SUCCESSOR-20260928`
- CPU task: `T-ec9addeea721e9f9`, terminal `DONE`
- Live task: `T-e9988179e9ce892a`
- Product/workload base and HEAD: `a0816149ced88bd46c1e0104413b73050626a7e9`
- Candidate diff: `dd95d3f64d903bae131da7cf7d05be5d3718fef579027eed7d8f6e826185917b`
- Independent CPU Evaluator: `S-5acacc1cefe2475c`, `PASS`, evidence `E-S-5acacc1cefe2475c`
- Current control release: `fd27a8db14ee1200c2ec090e5d8c8bf1345b13596aaa17a0e1a2439333527aed`; isolated self-test: 393 passed
- Attempt: `/home/mint/rl_dev/instinctRL_coordination/attempts/development/UNIFIED-H1-R5/UNIFIED-H1-R5-STAGE1-B0-HANDOFF-SEAL-R1-20260928`
- Proposal: `P-915664a4e3d15561`, SHA-256 `915664a4e3d155613497d7d2b010d9a0430331d3b747ac9dfb28bdf26c09c054`
- Approval: `A-915664a4e3d15561`
- Job/evidence: `J-fe6a5d53ec4177b7` / `E-J-fe6a5d53ec4177b7`
- Final terminal Evaluator: `S-3471ac95c995408a`, verdict `INCONCLUSIVE`, finding `F-LIVE-EVIDENCE-NOT-AVAILABLE`

The final CPU classification was `AUTHORIZED` with all four prelaunch criteria covered. The classification recovery was additionally hardened to bind the exact 14 paths changed from the package baseline, eliminating the prior 23-versus-14 path ambiguity.

The final terminal Evaluator independently reproduced the candidate HEAD/diff identity and reran 60 relevant CPU tests successfully. It could not independently accept either live-task criterion because its frozen sandbox evidence index omitted `E-J-fe6a5d53ec4177b7` and did not expose the external report contents. Its verdict was therefore `INCONCLUSIVE`, not a scientific or cleanup PASS. No further role call is available (aggregate 33/33), and this does not authorize a live retry.

## Exact argv

```text
/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B -m training_workloads.unified_h1.formal_unified_h1 --authorization /home/mint/rl_dev/instinctRL_coordination/attempts/development/UNIFIED-H1-R5/UNIFIED-H1-R5-STAGE1-B0-HANDOFF-SEAL-R1-20260928/commands/AUTHORIZATION.json --attempt /home/mint/rl_dev/instinctRL_coordination/attempts/development/UNIFIED-H1-R5/UNIFIED-H1-R5-STAGE1-B0-HANDOFF-SEAL-R1-20260928 --mode enhanced-b0-diagnostic
```

Seed remained `20260909`; cases remained Z/T/S; the target velocity remained zero. Actor, reward, controller, mass, COM, dynamics, curriculum, reset physics, thresholds and training method were unchanged.

## Failure boundary and evidence

The job receipt records exit status `FAILED`, exit code `-9`, elapsed 27.714272305951454 seconds and allocated GPU time 27.176286880043335 seconds. Initialization consumed one live start, one simulator initialization and two initialization physics ticks. Before failure:

- attempted/completed policy steps: 0 / 0
- nontraining transitions: 0
- training transitions: 0
- optimizer updates: 0
- checkpoint writes: 0
- explicit/automatic resets settled: 0 / 0
- physics diagnostic rows: 0 of 403,200

The status is `FAIL_PENDING_CLOSE`; the primary error is a `RuntimeError` stating that enhanced-B0 controller/rotor reset state is unavailable. The raw physics JSONL exists and is empty (SHA-256 `e3b0c442…b855`). Because there are no rows, force, torque, COM, attitude, momentum, velocity, height, reset-state continuity, x/y/z common-cause, and numerical closure results are all unavailable—not zero.

## Budget settlement

- Live starts: 1 / 1
- Simulator initializations: 1 / 1
- Actual transitions: 0 / 20,160; rlflow retains the 20,160 transition reservation because the terminal counter record is formally incomplete
- Physics rows: 0 / 403,200
- Wall: 27.714272305951454 / 910 seconds
- GPU allocation: 27.176286880043335 / 910 seconds
- Diagnostic output: 406,438 bytes in attempt results plus 32,124 bytes in the rlflow job directory, below 16 GiB
- Automatic retries: 0
- Training transitions / optimizer updates / checkpoint writes / P0 / PLR / policy evaluation: all 0
- rlflow role calls under the standing authorization: 33 / 33 aggregate; final audit used the last call

No second live attempt is permitted.

## Cleanup and final resource audit

`environment.close` reported no cleanup error, but `app.close` hit the 9.5-second hard timeout. The watchdog SIGKILLed owner PID 615502. Isaac telemetry PID 615694 was reparented to PID 1 and escaped the owned process group; it was terminated explicitly (TERM, then KILL after five seconds). Final audit found no Isaac Sim, SimulationApp, formal runner, rlflow experiment worker/lifeguard/daemon, or task GPU process. GPU PID 3950 (`gnome-remote-desktop-daemon`, 164 MiB) predates and is unrelated to this task.

## Unresolved risks and minimum next action

1. The product reset protocol requests a controller/rotor reset snapshot before that state is available at the first explicit reset. The next CPU-only successor must trace the exact call order and either establish the first valid reset-state boundary or fail closed before simulator release.
2. The app-close watchdog/process ownership does not capture the detached Isaac telemetry child.
3. The horizontal drift root cause remains unknown because no physics-step evidence was produced.
4. The rlflow transition reservation remains conservative at 20,160 even though raw counters show zero actual transitions.

The minimum next action is a new additive CPU-only successor that fixes the reset-state availability contract and detached-child cleanup, adds tests for the initial explicit reset before any policy step, and receives independent review. A new live run would require fresh explicit live-start authorization; this report does not authorize one.

## Next prompt

> Create a new additive CPU-only rlflow successor from `20260928-stage1-handoff-seal-live-failed`. Fix only the initial explicit-reset diagnostic state availability and detached Isaac telemetry cleanup demonstrated by job `J-fe6a5d53ec4177b7`. Preserve candidate control semantics and all frozen B0/training contracts. Add tests proving fail-closed pre-release validation of controller/rotor reset state, exact reset accounting, and descendant cleanup. Obtain independent Evaluator PASS and stop with a new proposal for any future live run; do not run simulation or training without fresh authorization.
