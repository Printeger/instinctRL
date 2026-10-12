# 当前任务

2026-10-12 新用户 Goal：诊断闭环、最小修复与短程训练准入。
需求：`docs/goals/R5_DIAGNOSTIC_CLOSURE_REQUIREMENTS.md`。
目标：`READY_FOR_PAIRED_PILOT_WITH_FIRST_ROLLOUT_GATE`。
当前：`BLOCKED_EXTERNAL_AUTHORIZATION`，Goal 未完成。

唯一开发目录 `/home/mint/rl_dev/instinctRL-dev`，唯一分支 main，同一目录单写者。
实现、CPU 验证、工具兼容、证据处理、提交与正常 push 沿用本次授权；历史目录、原始证据和冻结 runtime 保持只读。
本 Goal 的训练 transitions、optimizer updates 和训练 checkpoint writes 均为零。

原 4 次 live／初始化保守预留已消耗，无活动 job。第四次 J-afe4bda51f5206e5 的完整 403,200 行物理轨迹
已独立验收为零目标物理回归；进程仍为 FAILED/137，最终 reward 计数缺失，20,160 全额预留保留。
累计保守 transitions 预留 60,480，真实 wall/GPU/输出与角色用量见 `docs/diagnostics/r5-readiness/goal_budget.json`。

准确只读候选 ef68ebb5682b85aa83044fdea1c13b7d0edbf325 保留已回归的控制器和 plant，
仅补分析前持久化与 pilot-evaluation 工具 seed 分支。Role12 最终 PASS，两项 finding 关闭。
准确工具为 74a6042b0a1c92bda01efc5cd62da7f59948db4ff2be667eb9e34765c1a76c60。

已集中请求用户仅增加一个启动／初始化保守预留（4→5）；尚无明确答复，不重复请求，不把自动续接当批准。
具体申请：`docs/diagnostics/r5-readiness/budget_extension_proposal.json`。
准确命令直通包：`docs/diagnostics/r5-readiness/inputs/passthrough_r3_input_manifest.json`；
proposed Commander binding 只是条件式草案，实际 AUTHORIZATION、审批和 release 均未签发。

唯一恢复动作：取得明确资源决定后更新 cap（旧用量不重置）→现有 Commander 绑定独立证据与准确输入／剩余预算
→通过 STOP、coexistence 和所有权检查→释放 fresh PLANT-R3 命令直通一次→原 Z/T/S gate 及最终独立审核。
达到技术准入后停止，本 Goal 不开始配对训练。

唯一最新索引：`docs/diagnostics/r5-readiness/STATE_AND_EVIDENCE.md`。
准入 manifest 内 `requirements_completion_audit` 逐项区分已证明范围与剩余工作。
