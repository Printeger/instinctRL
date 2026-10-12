# 当前任务

2026-10-12 新用户 Goal：诊断闭环、最小修复与短程训练准入。
需求：`docs/goals/R5_DIAGNOSTIC_CLOSURE_REQUIREMENTS.md`。
目标：`READY_FOR_PAIRED_PILOT_WITH_FIRST_ROLLOUT_GATE`。
当前：`CONTROL_DIAGNOSTIC_PENDING`，Goal 持续推进，仍未完成。
用户明确批准“扩额8-10次”；累计 start/init 保守预留 cap 采用 8，其他总帽、旧用量和失败预留不变。

唯一开发目录 `/home/mint/rl_dev/instinctRL-dev`，唯一分支 main，同一目录单写者。
实现、CPU 验证、工具兼容、证据处理、提交与正常 push 沿用本次授权；历史目录、原始证据和冻结 runtime 保持只读。
本 Goal 的训练 transitions、optimizer updates 和训练 checkpoint writes 均为零。

原 4 次 live／初始化保守预留已消耗，无活动 job。第四次 J-afe4bda51f5206e5 的完整 403,200 行物理轨迹
已独立验收为零目标物理回归；进程仍为 FAILED/137，最终 reward 计数缺失，20,160 全额预留保留。
累计保守 transitions 预留 60,480，真实 wall/GPU/输出与角色用量见 `docs/diagnostics/r5-readiness/goal_budget.json`。

准确只读候选 ef68ebb5682b85aa83044fdea1c13b7d0edbf325 保留已回归的控制器和 plant，
仅补分析前持久化与 pilot-evaluation 工具 seed 分支。Role12 最终 PASS，两项 finding 关闭。
准确工具为 74a6042b0a1c92bda01efc5cd62da7f59948db4ff2be667eb9e34765c1a76c60。

当前资源决定：`docs/diagnostics/r5-readiness/user_resource_decision_startcap8.json`。已用 4 次，剩余 4 次启动槽，但 nontraining 总额仅剩 20,160，不能视为 4 份完整实验预算。
准确命令直通包：`docs/diagnostics/r5-readiness/inputs/passthrough_r3_input_manifest.json`；
实际 AUTHORIZATION 与 `passthrough-r3-commander-approval.json` 已签发，精确源码/配置/工具/独立条件已绑定。

唯一下一动作：STOP、无活动 job、已记录所有权清理和内部 coexistence 通过后释放 fresh PLANT-R3 命令直通一次；
按原 Z/T/S gate 评估真实结果并独立最终审核，旧任务不重启。
达到技术准入后停止，本 Goal 不开始配对训练。

唯一最新索引：`docs/diagnostics/r5-readiness/STATE_AND_EVIDENCE.md`。
准入 manifest 内 `requirements_completion_audit` 逐项区分已证明范围与剩余工作。
