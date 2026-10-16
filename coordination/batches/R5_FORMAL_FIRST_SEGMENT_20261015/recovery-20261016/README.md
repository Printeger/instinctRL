# 固定正式首段的有限恢复

用户2026-10-16选择优先验收现有训练，再完成原定评估。见[授权记录](USER_AUTHORITY.md)。

这是原本地Batch的单次恢复扩展，复用原Commander、准确job等待和owned模型worker；不是新调度框架。原两训练job不重新进入run_entry，也不再settle/postrun。只有一次额外checkpoint-recovery独立调用，原首次STOP永久保留；最多第三次调用为原final-analysis。整个过程由Python等待，Codex不轮询训练/评估或模型进度。

唯一状态：`coordination/research/batches/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/state.json`。原FAILURE、原call目录不改；恢复故障写入同runtime的`recovery-20261016/FAILURE.json`。原plan和前状态由本目录精确pin保留，新plan只允许CPU复审启动，不代表checkpoint或GPU评价已获接受。

顺序：用户授权精确STOP→arm一次→独立审核恢复代码/原始证据→接受并通过原Commander发行两准确评价→原R0评价→正常清理/settle/2postrun→原R1评价→正常清理/settle/2postrun→CPU全部原函数重放→最终一次独立分析→COMPLETE。任一真实异常STOP，不自动重试。

训练更新/训练样本/新训练checkpoint增量0。原4个live入口中只余2评价入口，各164800nontraining；冻结feb1e96/configd263/source/runtime和五节点Z/T/S/X/W不变。main增加未来formal入口binding记录的修复不部署进这次已冻结评价。旧nativeFAILED/collectorINCOMPLETE/full预留不变。

恢复tmux：`instinctrl-r5-formal-recovery-20261016`。控制Python为plan中的既有orchestrator环境，CPU证据环境为既有navrl环境。正常提交/tag/push后，准确本地入口如下（已ARMED或RUNNING不得重复arm）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/rlflow:isaaclab-training CUDA_VISIBLE_DEVICES='' \
/home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python tools/r5_formal_recovery.py arm
```

后台run由tmux启动同解释器 `tools/r5_formal_recovery.py run`。arm验证原STOP、完整旧状态、无owned旧控制/model、Store静止、外部STOP/pause无、无准确评价许可/attempt和输入已push后原子附加恢复记录。只有RUNNING的崩溃接管可重新run；STOPPED直接拒绝，不能删除ONCE/receipt或手改状态解锁。

看日志：

```bash
tail -f coordination/research/batches/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/controller.log
```

独立调用输出：同runtime的`calls/checkpoint-recovery/`和`calls/final-analysis/`，各有准确prompt pin/argv/owned identity/request、events/stderr、last-message、exit receipt和summary。`checkpoint-recovery.prompt.md`要求原证据等价裁决与恢复代码预审都成立后才issue；否则保存真实STOP，不虚构许可。

成功交付：`docs/experiments/r5-formal-first-segment-20261015/FINAL_ANALYSIS.md`、`EXPERIMENT_RESULTS.json`及原最终审核。新的实际训练接受另存`recovery-20261016/joint-actual-training.json`，旧审核仍为STOP。

验证：CPU测试记录[tests.xml](tests.xml)；两组完整CPU补充分析位于[证据论证目录](../../../../docs/experiments/r5-formal-first-segment-20261015/recovery-20261016/EVIDENCE_ARGUMENT.md)。单纯CPU通过不允许评价，独立复审必须接受精确plan与10checkpoint。
