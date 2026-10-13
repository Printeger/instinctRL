# Goal 启动 Prompt

将 `InstinctRL_R5_Autonomous_Followup_Goal_Requirements.md` 放进本地 `docs/goals/`，在本地 Codex Goal 界面提交下面全部内容。首次运行前按实际仓库和可用资源审查，用户通过下方整段授权的是**明确的总范围**，不代替平台必要的机器级授权。

```text
/goal

目标：在一个 Goal 内连续完成 R5 失败原因的有限离线归因、零命令残差修正的单变量实现、CPU 回归、短程配对训练及 Z 评估、独立审核和科研决策。不要只写报告或做到 CPU_READY 就结束。

先读取 docs/goals/InstinctRL_R5_Autonomous_Followup_Goal_Requirements.md，并核对最新 docs/PROJECT_STATUS.md、PAIR_REPORT.md、pair_results.json、当前 HEAD/现有作业/授权。

已知上一轮 P0/PLR 均完成64updates、真实首次 gate PASS，PLR u64 速度与高度达标但漂移1.088824m，零命令训练平均 v_corr_y约-0.05559m/s、实际 v_y约-0.05740m/s。禁止重复 E0、质量修复、旧配对 Pilot。所有旧 negative 和 native FAILED/-9 均保留。

按需求自动推进 Phase A→B→C→D。默认候选为仅在零命令添加软性的三维 residual-neutrality reward，w=0 vs 预注册一个正权重；共用actor LR1e-4，其余方法不变。若离线证据反驳候选或发现重大实现缺陷，应给出具体反证/修复，不能带病运行；非实质任务自主推进，不每阶段要求我发“继续”。

我一次性同意此 Goal 在完成现有精确身份/资源/Commander 审批后，最多执行两组各64updates及各u16/u64原Z评估；总上限4次live启动、65536训练transitions、25600评估transitions、128updates、14训练checkpoint writes、0自动重试；wall/GPU/disk上限须在新整体资源包中按规范精确填写。这不是绕过机器审批的许可，也不继承旧预算。

Routine CPU改动、分析、既定单变量实现、版本提交与正常push、已有范围内的审查及已审批作业间流转，请自主完成。仅在真实STOP、用户不可代理的外部批准、需要扩大上述预算/改变方法、安全边界、证据不可信等实质阻塞时中断；如必须向我请求操作，集中为一个请求，明确唯一所需决定，不拆成多次“是否继续”。

必须保留真实首次512样本概率gate、全部原Z门槛/分母/失败样本，运行前冻结commit/config/seed/试验唯一变量。不得在第一次负结果后私下多次换权重或自动扩到256updates。

最终交付：归因文档、唯一变量代码diff及CPU测试、真实配对机器结果、证据索引、独立审核、NEXT_RESEARCH_DECISION.md；实际负结果也算Goal科学完成。不要以“还可能需要进一步研究”作为唯一结论。
```
