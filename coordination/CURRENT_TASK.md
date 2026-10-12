# 当前任务

2026-10-12 新用户 Goal：诊断闭环、最小修复与短程训练准入。
需求：`docs/goals/R5_DIAGNOSTIC_CLOSURE_REQUIREMENTS.md`。
目标：`READY_FOR_PAIRED_PILOT_WITH_FIRST_ROLLOUT_GATE`。
当前：`CONTROL_DIAGNOSTIC_PENDING`，用户已明确批准恰好一次 ENTRY6 资源拒绝重试例外；Goal 未完成。
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

当前资源决定：`docs/diagnostics/r5-readiness/user_resource_decision_startcap8.json`。已用 5 次，剩余 3 次启动槽，但 nontraining 总额仅剩 20,160，不能视为 3 份完整实验预算。
准确命令直通包：`docs/diagnostics/r5-readiness/inputs/passthrough_r3_input_manifest.json`；
实际 AUTHORIZATION 与 `passthrough-r3-commander-approval.json` 已执行一次，J-1f1d10a06a0e0632 终态 FAILED；不能重用。

第五次真实窗口利用率 [4,6,1]%/10.030832815s，395MiB，仅既存 GNOME164MiB；原 5% gate 正确拒绝。AppLauncher/physics/transitions 均未到达，H1 FINAL0 正式结算第五次预留。正常 lifeguard ack 两次稳定空扫描，无所属残留；授权两次 postrun 审计已用完。
唯一下一动作：实际 ENTRY6 许可和 Commander A-7b50c8e3ccc99a27 绑定已完成；提交实际输入后检查 STOP／已知所有权，并由原六次内部查询通过 coexistence 后才创建 AppLauncher。
本次用户决定 user_entry6_prelaunch_exception.json 仅授权一次新入口；再次拒绝／失败即保存、结算并停止，无后续重复启动／探测／门槛变更许可。
达到技术准入后停止，本 Goal 不开始配对训练。

唯一最新索引：`docs/diagnostics/r5-readiness/STATE_AND_EVIDENCE.md`。
准入 manifest 内 `requirements_completion_audit` 逐项区分已证明范围与剩余工作。
