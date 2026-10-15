# 固定正式第一段：本地无人值守执行

本目录只驱动已冻结的 `R5-FORMAL-MEAN10-FIRST-SEGMENT-01`，不提供通用调度、重试、扩额或方法选择。执行源码/config/runtime/manifest/许可不改。唯一进度状态为：

`coordination/research/batches/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/state.json`

顺序固定：R0训练 → receipt/normal owned ACK/settle/两次postrun/完整计数及checkpoint写入hash → R1训练同样闭合 → **一次独立codex exec实际训练/checkpoint验收及准确评价许可** → R0全部25case评价闭合 → R1全部25case评价闭合 → 原始证据哈希检查 → **一次独立codex exec最终科学审核与分析**。实际第一轮样本概率gate仍由冻结训练入口在两类optimizer之前执行，原阈值不变。只有已知close9.5s、完整科学计数、normal ACK及正常quiescence可容纳nativeFAILED/-9；INCOMPLETE/null与保守预留不变。性能负结果如实分析，不自动进入下一段。

本地Python以15秒间隔读取精确job的Store/receipt/owned ACK，只在receipt/所有权需要时调用原reconcile；不用模型轮询，不额外GPU探测。模型只占两个固定slot，各最多一次，最长3600秒、输出256MiB；非零退出、API/授权异常或不完整输出停止，不自动再调用。CLI原生JSON事件、last-message、stderr、准确argv/prompt hash、退出码、摘要在运行目录的`calls/{checkpoint-review,final-analysis}/`保存。模型调用使用本机已登录的Codex配置，不进行额外付费探测，不保证未来API请求必成功。实际API调用只会在相应证据门槛达成后发生；错误进入FAILURE。

`codex exec`的stdin、JSONL事件与最终消息文件按[官方非交互说明](https://developers.openai.com/codex/noninteractive)使用；以本机0.161.0实际help解析和登录状态为准。采用`codex -a never -s danger-full-access exec --json --output-schema ... --output-last-message ... -`，适用本次明确授权的项目CPU审核/准确许可/正常push；不绕过任何Commander/正式入口守卫。独立调用不继承Builder对话，只读取真实证据和固定prompt。

首次发车由根任务在审查通过、正常commit/tag/push和`init`后，tmux中执行：

```bash
cd /home/mint/rl_dev/instinctRL-dev
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/rlflow /home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python tools/r5_formal_batch.py run
```

固定tmux名`instinctrl-r5-formal-first-20261015`。运行期间同目录只有controller或它等待的独立Codex子进程可写；其他上下文只读，禁止并行实现/提交或手动submit。正常检查用shell读取状态/日志即可，不唤起Codex轮询：

```bash
cat coordination/research/batches/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/state.json
tail -f coordination/research/batches/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/controller.log
tmux attach -t instinctrl-r5-formal-first-20261015
```

可恢复指controller意外中止后，使用同一`run`重接精确已有job/已闭合步骤/已产生的模型receipt，绝不重新`init`或重新submit；flock阻止第二个controller。submit之前已原子写意图；有意图/ledger/admission但无可确认job时，或settle/postrun中断留半成品时，进入`STOPPED`/`FAILURE.json`，必须人工核对真实事实，不猜测重试。模型worker与controller分离，controller中止后worker仍将持久化真实退出码；无receipt且worker已死同样STOP，不再调用模型。STOPPED不会自动恢复或清除STOP。外部STOP交给既有runner清理，controller只保全证据。仅controller收到终止信号时，不杀无关进程或实验；等待当前owned job安全闭合后停止推进。

所有JSON状态/receipt/请求用原子替换+fsync；详细日志仅追加。现有Commander的一次性副作用窗口不被冒充事务：不确定即停机。未来实际checkpoint与评价SHA由第一次独立验收填写，不能用模板或完成标志绕过。

自动分析交付固定为：

- `docs/experiments/r5-formal-first-segment-20261015/FINAL_ANALYSIS.md`
- 同目录`EXPERIMENT_RESULTS.json`和`reviews/final-scientific-review.json`

如资源gate、STOP、ownership、数据或API失败，不会生成伪造完整分析；准确原因在运行目录`FAILURE.json`及唯一state。既有正式启动4入口/资源包络保持固定，本次不授权任何重试或额外入口。
