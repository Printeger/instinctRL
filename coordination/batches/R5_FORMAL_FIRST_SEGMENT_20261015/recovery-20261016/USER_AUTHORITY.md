# 2026-10-16 已停止正式首段的有限恢复授权

用户最新原文：

> **优先争取验收现有训练成果，再完成原定评估**

该指令紧接助手明确说明的恢复路线：先修复证据写入和CPU检查，用现存原始证据接受独立复审；第一次独立模型已STOP，新增一次复审令累计模型调用由2变为最多3；接受后仅衔接原定两组五节点Z/T/S/X/W，最后独立分析。

执行解释：用户选择此路线，授权本次**一次额外独立复审及一次受控恢复**。已有第一次STOP调用永久保留；新的checkpoint-recovery调用最多一次；原尚未使用的final-analysis调用仍最多一次。若本次复审或任一评价/资源/所有权环节失败，停机，不自动再次调用模型、重试作业或扩大预算。

不新增训练、不重训、不新增实验变量、不改reward/PPO/actor边界/物理/评价标准。原4个live入口中两训练已消耗，剩余仅R0/R1原评价各一次、各164800非训练transitions；训练transitions/optimizer/checkpoint writes新增均0。原冻结source/config/manifest/runtime保持。现有训练保守账本不退款、不重置。

CPU重分析允许把原训练留下的实际minibatch记录、保存probe、真实checkpoint和冻结源码组成**新派生证据**；不能写入或伪造原缺失actor_mean_prior_binding.json，不能覆盖原STOP审核。独立Reviewer必须判断原实质要求是否已由原始证据满足，不能以用户希望继续作为PASS依据。若需要豁免或改变契约，STOP并明确缺口；本授权不预先允许降低验收标准。

唯一状态仍为原batch state.json。arm只在精确原STOP状态、原模型退出、训练终态正常清理、无评价已issue/submit、无外部STOP/pause时原子附加恢复历史并进入ARMED。旧plan、state快照、FAILURE、模型调用和证据哈希保留。此文是当前会话指令的项目记录，不是外部签名。
