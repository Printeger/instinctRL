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

当前准确续接：4次live已消耗且无activejob。第四次完整物理轨迹403200行已独立复核并接受零目标物理回归，但进程FAILED/137、最终reward计数缺失、20160全额预留。新只读候选ef68ebb5682b85aa83044fdea1c13b7d0edbf325保留相同控制plant，仅补诊断持久化和pilot-eval工具seed分支；Role12精确预审待结论。下一唯一live是不同条件的命令直通B0，准确包inputs/passthrough_r3_input_manifest.json；需用户仅将start/init保守预留4→5的明确决定，再经Commander批准，其他80640/3640/64GiB/24帽不变。budget_extension_proposal.json已完成具体申请包。训练仍为零，目标未完成。

Role12已最终PASS，durability与pilot-eval seed两项finding关闭，准确命令输入及工具预审通过。已集中请求用户仅增加一个start/init保守预留，等待明确答复；不能把未答复当批准。本Goal未DONE、未开始第五次或训练。得到批准后唯一恢复动作：更新新cap（不重置旧用量）→按Role12证据/准确ef68候选与工具74a6进行现有Commander绑定、STOP/coexistence/所有权检查→释放freshPLANT-R3命令直通一次→原ZTS验收与最终独立审核。
