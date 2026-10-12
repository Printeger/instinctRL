# Current Order — UNIFIED-H1-R3（D0005 已封存，A0009 审核待批）

STATUS: HOLD
ORDER_ID: UNIFIED-H1-R3
ORDER_REVISION: 0
STAGE_ID: UNIFIED-H1-R3-A0009-RESOURCE-WINDOW
EXECUTABLE_NOW: NONE
IMPLEMENTATION_AUTHORIZED: NO
EFFECTIVE_LIVE_STARTS_ALLOWED_NOW: 0
TRAINING_AUTHORIZED: NO
CONTROLLED_FAILURE_RELEASED: NO
R4_R5_AUTHORIZED: NO
USER_DECISIONS_PENDING: 1

用户已批准并执行[精确 D0005 申请](drafts/UNIFIED-H1-R3_POST_A0007_MAIN_REQUEST.md)，SHA256 `7cb51f824c8bee54df65acde2f2bc1fc05d8a22d4cd12b1fcb40074cc44268b3`。D0005 在 AppLauncher 前因三样本跨度小于十秒 fail-closed；唯一 start 已消耗且不重试。[A0009 审核](reviews/UNIFIED-H1-R3_A0009_D0005_FAILURE_AND_CPU_REPAIR.md)记录封存证据、诊断与 CPU 修复。

待用户集中决定[rlflow A0009 独立验收与条件 D0006 申请](drafts/UNIFIED-H1-R3_A0009_REVIEW_AND_D0006_REQUEST.md)。未批准前无 executable，不签发 D0006，不查询 GPU。D0002、controlled-failure、R4/R5 和正式训练仍未授权。

[A0009 审核](reviews/UNIFIED-H1-R3_A0009_D0005_FAILURE_AND_CPU_REPAIR.md)，SHA256 `033cda18127d0faa123c446b140411280cdee378aad9e7c6842c5c807fafe97d`：D0005 G1 FAIL，G2/G3/G4 未到达，G5 的 owned-group clean exit PASS 但 AppLauncher/env cleanup 未到达。新 finding F6 等待独立 CPU 验收；D0004 的 ray failure、app.close>5s、exit139/SIG11 与 native root cause NOT_PROVEN 继续保留；F1–F5 不重开。

拟议候选：product `/home/mint/rl_dev/worktrees/instinctRL-unified-h1-r3-a0007-product` commit `2248ebb380260af86536f78b70e1235fcb401a7a` / tree `e44363a4b8504fda9e8f1b58a1fb942ef75c586e`；workload `/home/mint/rl_dev/worktrees/instinctRL-unified-h1-r3-a0009-workload` commit `fd47dd321d571c4d6c983048f77bb3913e7e7b01` / tree `f2ccbe684e9d0fb56a861152b91ea4b4870500ee`；config SHA256 `824feeb87fccb33ddb5b56b401b6591688b042606fda7d45f4d5f92c041f110b`。两仓库 clean，方法、actor、阈值与 runtime 未变。

rlflow A0007 账本保持 used 8 calls / `1850.3746883358108` 秒、unused 5 / `2349.625311664189` 秒且 HOLD，不退款、不重置。[新提案](../research/proposals/P-4d1bffeeda49b4d2.json)单独请求 A0009 Sol 3 calls / 1800 秒 CPU stage；未批准，未消费。

[A0009 方法冻结包](review_artifacts/UNIFIED-H1-A0009/METHOD_FREEZE_PACKAGE.json) SHA256 `513dab7a9d4db25de8d8db2adbcdaae4efb9368226aca0547834fda76dd615df`，`training_authorized=false`。原 [R4 精确启动方案](drafts/UNIFIED-H1-R4_TRAINING_START_PLAN_A0008.md)仍被 R3 gate 阻塞。

历史累计 live starts 4、AppLauncher constructions 3、simulator-initialization slots 4；D0004 实测 ticks 2，D0001/D0003 未知；真实 transitions/PPO/checkpoints 仍为 0。D0002 未创建/释放。
