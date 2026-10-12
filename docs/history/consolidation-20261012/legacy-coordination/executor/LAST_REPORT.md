# Last Executor Report

Current delivery: [UNIFIED-H1-R3 A0006 Executor Report](reports/UNIFIED-H1-R3_A0006_EXECUTOR_REPORT.md)

Status: `READY_FOR_REVIEW` / evidence `SEALED` / outcome `MAIN_ENGINEERING_FAILURE`.

The single authorized D0004 main was invoked once with a durably captured PID/PGID/SID chain and no retry. Exact bindings, the three-sample RTX 5070 gate, runtime memory, environment construction and two initialization physics ticks passed. Actual 32-environment Warp/USD geometry matched the translated fixture and the raw-ray source was persisted.

Initial exact ray observability then failed with `RuntimeError: returned ray parameters disagree with forward-reconstructed hits`. No B0 case, transition, PPO update, optimizer step, checkpoint or reload ran. Independent `app.close` exceeded five seconds; the process ultimately ended by signal 11 / exit 139, without an outer-timeout trigger or an owned process remaining. The precise numeric disagreement and native crash causes remain `NOT_PROVEN`.

D0004 consumed one live/AppLauncher/simulator-initialization slot. R3 cumulative live and initialization slots are now three each; cumulative real transitions, updates and checkpoints remain zero. D0002 remains uncreated, unstarted and unreleased. Product, workload, method, actor, runtime and shared tooling were unchanged.

Analysis: [UNIFIED-H1-R3-A0006](../analyses/UNIFIED-H1-R3/UNIFIED-H1-R3-A0006/ANALYSIS.json), SHA-256 `4012e2fbcc8edba1137c7dc22d38dc4a28e235ab224a532c21e2c568d5822f4d`. Seal SHA-256: `5669c302144028c5c7ca41dbb4b36a5dcdfbc761092e4c95f6e088a65db77aad`.

Next step is Commander review. Builder must not retry D0004, start D0002, modify the execution candidate, or begin R4/R5.
