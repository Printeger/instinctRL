# R6 生效委托

用户在本轮提交 `USER_AUTHORITY.txt`，已触发需求 v1 的启动条件。完整契约为
`docs/goals/InstinctRL_R6_Robustness_Autonomous_Goal_Requirements.md`；原文及哈希见
`STARTUP_PREFLIGHT.json`。需求开头的 DESIGN_ONLY 是编写时状态，不撤销此次启动授权。

目标保持完整：冻结新测试、测量 u640、诊断与最多三个科学候选、必要训练、冻结配方后的
三条共同 u640 起点续训确认、一次性最终测试及基线对照、独审、模型、报告与真实三维视频。
基线足够时走三个实际不同随机流的 BASELINE_SUFFICIENT 分支。预算耗尽和基础设施阻塞
不属于科学阴性；CPU 准备和后台启动都不属于科学完成。

资源硬上限及保护额度在 `goal_budget.json`。这是新包络，不继承或退还任何 R5 额度。
普通工程修复、内部审核、Store 精确许可发行、提交/tag/正常 push 已获授权。正式发车
仍须完成容量证明、完整输入冻结、独审、exact Store 许可及控制器交接验收。
本文件本身不构成某个作业的启动许可。

actor 信息边界、旧 Z/T/S/X/W 全分母与阈值、0.8m 几何底线、真实 first-rollout gate、
冻结 runtime 不变。H2、导航、新算法/网络族、依赖/驱动升级不在范围内。
所有 R5 源码、原件、checkpoint、原生 FAILED 与结算保留。

唯一写者工作区为 `/home/mint/rl_dev/instinctRL-dev` 的 main。控制器是唯一
submit/cancel/settle/运行终态写者；模型只在交接窗口提交方案，独审只读。
CPU 阶段当前由启动会话写入，尚未部署控制器或调用自动模型。

平台交接限制：当前聊天 Goal 具有自动续推理，暴露接口没有 active-without-continuation
状态；已向用户请求仅在交接验证后暂停聊天 Goal 的授权。本地项目状态与聊天状态分开。
未得到该授权或另一项已验证的零模型交接机制前，不声称训练期间零模型等待已成立。

本轮新增 STOP 预定路径为 `docs/experiments/r6-robustness-auto/batch/STOP`，并保留
rlflow 全局 STOP。尚未部署执行器，所以此路径目前不是已验证的运行接口。不得自动清除
任何 STOP；部署后的语义必须是停止新动作、取消并清理准确 owned job、保全和结算。
