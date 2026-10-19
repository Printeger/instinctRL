# R5 W 最终交付索引（2026-10-19）

当前科学结果：R5_ALL_ORIGINAL_GATES_PASS。最终独审见 reviews/final-independent.json；controller终态仅由 batch-resource-v2/state.json 记录。

- FINAL_R5_W_RESULTS.md/.json：结果、模型、限制；TRAINING_AND_EVALUATION_MANIFEST.json：来源、method、训练与全部任务/方向/墙组。
- FINAL_ORIGINALS_INDEX.json：7实际jobs、286条原件索引，每条原始/归档路径、bytes、SHA；大checkpoint/raw仍在原绝对路径。
- FINAL_RESOURCE_AUDIT.json、goal_budget.json：所有失败、全reserve和5snapshot，不退款。
- INDEPENDENT_RESOURCE_V2_CPU_REPLAY.json：真实seed20261020、完整160case/lane、原primary mismatch、W运动/时序。
- FINAL_METHOD_COMPARISON.json、FINAL_W_TIMING.png：R1/E1/L1/独立seed；工程收益与训练分开。
- FINAL_REQUIREMENTS_AUDIT.json：逐项需求验收；reviews/resource-v2-actual-independent.json：实际运行独审；reviews/final-independent.json：最终结果准确pin。
- 精确资源V2计划 coordination/batches/R5_W_CLOSURE_20261019/plan-resource-v2.json；旧batch/state及历史STOP结果保留，唯一现controller使用batch-resource-v2。

以下为历史逐阶段交接记录，含当时尚未执行/IN_PROGRESS表述，不覆盖上述当前证据。

# R5 W closure evidence index — IN PROGRESS

The complete Goal remains active. E1 engineering and L1 training evidence are accepted; both trained nodes still await full evaluation. This index is not a completion audit.

- User scope, immutable incremental master and caps: USER_GOAL_PROMPT.txt, GOAL_SCOPE_AND_MASTER_AUTHORITY.md, MASTER_AUTHORITY.json, goal_budget.json; accepted registration registrations/W-E1-001.json.
- Original accepted model and source: BASELINE_IDENTITY.json and reviews/baseline-accepted.json. Original R1/u512 SHA a3b9f9115268270ec000750035085b03980de2aed31987b7b08ae97bf26d8776.
- Bounded original W diagnosis: W_CAUSAL_DIAGNOSIS.json/.md, W_TIMING.png, ORIGINAL_TRAINING_COVERAGE.json, reviews/diagnosis-independent.json. 320 lanes, unchanged score, separate input/control/observation clocks; q=0 for all131072 original training samples per arm.
- Engineering/learning separation and prospective nodes: METHOD_DECISION_LOG.md, coordination/contracts/R5_W_SAFETY_CLOSURE_V1.md.
- Fixed-policy E1 source/config/compatible origin/expected AUTH/packet/approved spec: experiments/R5-W-E1-CONTROL-01/. Frozen source0c6363b, configdf96f98d; independent reviews/e1-exact-prelive.json. Exact approval A-9306976b2df3dee8, not yet submitted at this boundary.
- CPU tests and preserved initial fixture failure: w-cpu-tests*.txt/.xml; w-batch-tests*; w-batch-gate-tests-r2.*. Independent findings closed before issue: initial episode labels, STOP chain, ownership helper pin, incomplete counters and resumable settlement, large original archive boundary.
- Controller source tools/r5_w_batch.py; batch input/prompt/schema in coordination/batches/R5_W_CLOSURE_20261019/. Future sole runtime authority batch/state.json, controller.log and calls/*/request/receipt/events/stderr. No model token polling.

Latest terminal-event evidence:

- E1 original five-task replay and W motion: experiments/R5-W-E1-CONTROL-01/E1_CPU_REPLAY.json and reviews/e1-actual-independent.json; fixed-policy engineering gain only.
- First L1 prelaunch FINAL0 failure: L1_RECOVERY_DECISION.json and reviews/l1-failed-actual-independent.json; failed start/init/full8GiB preserved.
- RECOVERY1 exact originals: L1_RECOVERY1_ORIGINALS_AUDIT.json, experiments/R5-W-L1-NEARWALL-RECOVERY1/live/R1-research-formal-training/INVENTORY.json, receipt/ACK, execution summary and postrun0/1. All53 originals and100 file references rehashed. Full budget at event: L1_RECOVERY1_SETTLED_BUDGET_SNAPSHOT.json.
- Actual training/checkpoint independent acceptance: reviews/l1-recovery1-actual-independent.json includes full original/candidate source and method dictionaries and exact u512/u640/u768 pins; q2004 is measured danger coverage, not performance acceptance.
- New two-node evaluation: experiments/R5-W-L1-EVAL-01/ contains card, compatibility, packet, CPU checks, exact issued spec and approval A-f86ae4584e07d817. reviews/l1-evaluation-prelive.json. No new snapshot; stage W-L1-002 already covers cumulative resources. Evaluation NOT_EXECUTED at this handoff.
- Next controller input: coordination/batches/R5_W_CLOSURE_20261019/plan-l1-evaluation.json and after-l1-evaluation.prompt.md; independent controller review binds payload without review key.

Still required: both trained nodes' original five-task32lane replay and full W motion; first passing node independent seed/initial-state full verification; controlled contribution comparison, W time series, full resource audit and final independent review. Ordinary negative performance prompts a bounded evidence-based single-variable correction within the same Goal, not a user approval request. Only full evidence supports final positive or scientifically complete negative status.

## L1 原评价完成及独立验证交接

- `experiments/R5-W-L1-EVAL-01/live/R1-research-formal-evaluation/INVENTORY.json`：58项原件索引，106原件/归档hash核验，raw绝对路径不变。
- `L1_EVALUATION_REPORT.md`、`L1_METHOD_COMPARISON.json`、`L1_W_TIMING.png`：全部任务/子组、工程/学习分离、W时序和当前资源。
- `experiments/R5-W-L1-EVAL-01/L1_U640_CPU_REPLAY.json`、`L1_U768_CPU_REPLAY.json`：两节点独立调用冻结评分与每lane运动/事件。
- `reviews/l1-evaluation-actual-independent.json`：实际两节点独审接受；非Goal完成。
- `L1_EVALUATION_SETTLED_BUDGET_SNAPSHOT.json`：当前评价结算快照，不退款INCOMPLETE和失败。
- `experiments/R5-W-L1-INDEPENDENT-01/`：独立seed协议、readonly source/config、精确packet/发行输入；当前尚未执行。
- `reviews/v1-code-independent.json`、`v1-stage-independent.json`、`v1-prelive-independent.json`：CPU实现、累计授权和精确prelive独审。
