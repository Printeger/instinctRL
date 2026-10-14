# Formal first-segment operator entry

Status: PREPARED, NOT_READY. The independent confirmations, actual G6/G7 and final review/manifest/conditional permission are still pending. This document creates no approval and this readiness Goal does not submit formal training.

The prospective formal experiment is `R5-FORMAL-MEAN10-FIRST-SEGMENT-01`. Both arms restore the exact same actual DEV02 R1/u256 checkpoint, including actor/critic/Adam/RNG/clock state, then fresh-reset physics/history/anchor. The exact fresh-reset seed is20261038 (originaltrainingseed20260908 + actualoriginreset_number129 +1), identical for both arms. Current mean-prior coefficients are R0=0 and R1=10; both residual reward weights are.05. The callback is explicitly reinstalled after origin load. Details and exact source/config/budgets are in [the formal plan](FORMAL_R5_TRAINING_PLAN.md) and the committed AUTH/spec templates under `experiments/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/inputs/`.

After the final `formal_r5_readiness.json` binds a PASS independent review, the immutable `formal_r5_launch_manifest.json`, and the separate `formal_conditional_permission.json`, the later explicit start action is:

```bash
cd /home/mint/rl_dev/instinctRL-dev
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/rlflow /home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python tools/r5_formal_commander.py submit R5-FORMAL-MEAN10-FIRST-SEGMENT-01 R0 research-formal-training --start-formal
```

This is the first formal arm only. The command rejects absent/drifted READY, manifest, independent review, permission, exact Store approval, complete reviewed spec, tool release or input pins; the normal STOP/serial owned-cleanup/disk/coexistence/master/stage/quota checks remain. It transfers only that entry's unused protected formal reserve on the same conservative ledger precharge. A command without `--start-formal` rejects before opening the Store. Never run it during this readiness Goal.

After its exact job terminates, reconcile that handle, require normal owned ACK with two stable empty scans and no known owned survivors, then use the existing `settle` and `postrun` actions for that same experiment/group/mode. `postrun` makes exactly the two authorized queries. Preserve native FAILED and INCOMPLETE/full reservations when applicable. Do not rerun the old handle, manufacture a process PASS, or skip cleanup.

The second arm uses the same explicit later action with group `R1`; strict serial checks block it until the prior job is terminal and clean. Each arm has its own real first512 gate before either optimizer. Any gate/nonfinite/STOP/ownership anomaly stops that entry; retain failures and do not train through them.

Each formal evaluation remains generated input. After actual training/checkpoint acceptance independently binds all five real checkpoint paths/SHAs, issue the exact `research-formal-evaluation` spec through the same Commander with the final readiness review and that actual-training acceptance. Commit/tag/normal-push those generated exact inputs before its later explicit `submit ... research-formal-evaluation --start-formal`. Only five checkpoint SHAs and their declared acceptance condition may be filled; all other AUTH/spec fields must equal the frozen templates. The manifest and conditional permission enumerate the exact generated evaluation IDs, and the launch gate checks their complete specs.

Evaluate every predefined node272/288/320/384/512 on original Z/T/S/X/W with full T0/startup/32 lanes/thresholds/failures. Original Z regression blocks further continuation. Nonzero performance remains measured, never inferred from G6 implementation checks. The segment ends at absolute512; no automatic further curriculum stage, extra seed, horizon expansion or retry is authorized by these commands.
