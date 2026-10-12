# 当前任务

2026-10-12 新用户 Goal：诊断闭环、最小修复与短程训练准入。
需求：`docs/goals/R5_DIAGNOSTIC_CLOSURE_REQUIREMENTS.md`。
唯一开发目录和分支不变；本任务允许必要 CPU 实现、测试、工具兼容、证据处理、提交与正常 push，
及证据支持的最小底层实现修复。历史失败任务不恢复，冻结 runtime 和历史目录只读。

目标：READY_FOR_PAIRED_PILOT_WITH_FIRST_ROLLOUT_GATE。当前 IN_PROGRESS，尚未技术准入。
本 Goal 不执行 P0/PLR，不进行训练 transitions、optimizer updates 或训练 checkpoint writes。
新包络与用量见 `docs/diagnostics/r5-readiness/goal_budget.json`；每次 live 仍需现有 Commander
机制绑定准确候选、工具、输入、预算和初始化/reset/probe 细项，经独立预审、STOP、coexistence
及清理检查后 release。包络不是现成 release，Builder 不自我审批。

最新索引：`docs/diagnostics/r5-readiness/STATE_AND_EVIDENCE.md`。
