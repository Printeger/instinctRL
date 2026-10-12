# 给 rlflow 执行窗口的 Prompt

以下内容可整段发送。它表达完整开发目标和条件推进范围；未确定的资源硬上限必须先绑定真实提案，不能自行推定无限预算。

```text
请通过现有 rlflow 流程推进一个有界开发闭环：
接受 E0 的有限结论 → 实际完成 D7-R/B0 → 定位控制链限制 →
条件满足后实际完成带首次更新诊断的 P0/PLR 短程对照及 Z 评估 →
给出改动是否有效、是否值得扩大训练的 Markdown 报告。

工作区：/home/mint/rl_dev
必读：
1. AGENTS.md、ACTIVE_WORKFLOW.json、公共规则 V3 和现行角色入口。
2. InstinctRL_E0_Review_D7_Run_and_R5_Development_Plan.md。
3. InstinctRL_E0_D7_R5_Code_Review_20260923.md。

本指令将新 plan 第 11 节“仅准备 P0/PLR”的目标，扩展为：
满足科学准入且真实运行许可生效后，实际完成短程配对实验。
不授权长训、完整任务复验、额外 Bz/u0 实验或自动增加预算。

一、执行方式与权限
使用 rlflow 现有调度和角色机制实施与审核，不另建工作流。
负责实现的角色不能批准自己的候选；外部窗口不要与被调度角色同时写同一代码。
真实 GPU/仿真/训练统一通过 rlflow 作业运行，不手工旁路直接调用 workload。
先查当前 release、数据库状态、已有 job/attempt 和进程；已完成则复用证据，
运行中或启动状态不明则对账，不重复提交。保留 STOP、身份和资源门槛。

允许准备本任务必需的隔离 product/workload/control 候选、诊断代码、CPU 测试、
分析器及提案；不修改冻结目录、原始证据、runtime、驱动或依赖。
控制面变更经既有审核/发布流程使用，不直接改写冻结 release。
既有有效授权不重复索要；缺少确切运行预算时集中提供真实 proposal ID/SHA、
范围、预算和准确批准操作，不拆成多轮批准准备动作，不自行审批。
具体方法/控制修复超出以下范围时，只交付缺陷证据及最小修复提案。

二、E0 接受与运行前关闭项
复用 R4_REVIEW_ADDENDUM.md、r4_review_e0.json、脚本和 CPU 记录。
仅定向核验身份及影响决策的检查，记录 E0 接受范围，保留历史概率重放 NOT PROVEN。
不重新做全面审计，不把局部 minibatch 编号误解继续当成训练缺陷。

关闭代码审阅指出的问题：
1. rlflow H1 adapter 对 control-diagnostic 使用 evaluation_seed=20260909，
   与 backend 实际 reset 一致；补拒绝错误 seed 的测试，不伪造 spec seed。
2. 明确 D7 action GPU 1800 秒与外层含清理 allocated-GPU/wall 1820 秒的计量边界，
   使用真实审核后的资源配置；不静默扩大原授权。
3. 对 D7 两条件累计 16 GiB 日志上限提供实际执行保护和计数证据。
   必须改候选时生成新 commit/hash 并更新绑定，不冒用 c7a4d79e 的身份。
4. 完成真实 chunk 分析器的 CPU 测试：manifest/hash/步数/条件/有限值、
   B0 恒定动作、reset-cache mask；缺失不可填零或标 PASS。

三、实际运行 D7-R/B0
来源候选：worktrees/instinctrl-r4-d7-control-diagnostic-20260923
原 commit：c7a4d79e35916ce2711085674b9cd37f1054a1ab
来源提案：R4_D7_CONTROL_DIAGNOSTIC_PROPOSAL.json，位于原 E0002 分析目录；
原文件 SHA256：62992b2242e6aca4a9d4d999aab59f57a3c2e650f5208c132c8cebcd3825d703。
该来源摘要不等于注册后的 proposal SHA，必须准确关联两者。

D7-R 使用原 u256 mean action；B0 固定 [1,.5,.5,.5]，均走 H1 governor/ICS/Lee/rotor。
各执行 Z/T/S=200/100/330 policy steps、32 lanes、20160 nontraining transitions。
两条件总计最多40320 transitions、2次启动/初始化、0优化更新、0训练checkpoint、
0自动重试，其他预算按真实绑定提案。记录全部失败与实际资源，失败不重置额度。
不改变 reward、actor 输入、控制、physics、reset 和正式指标。
必要参考评分另作 DIAGNOSTIC_REFERENCE_ONLY，不写回原验收 verdict。

生成 R5_CONTROL_CHAIN_DIAGNOSIS.md、r5_control_chain_diagnosis.json、
少量代表曲线及 R5_PILOT_READINESS.md。
解释上游期望、下游响应、启动瞬态、重置段和 cache 有效性，核对新旧 u256 可比性。
结论选择 UPSTREAM_RESIDUAL_DOMINANT / DOWNSTREAM_OR_RESET_DOMINANT /
MIXED_CAUSES / INSUFFICIENT_SIGNAL，不预设 B0 必须通过。

四、开发首次更新诊断（CPU 开发可先进行，实际训练须通过准入）
在独立训练候选中接入真实训练路径，不能只加一个未调用的 helper：
- 采样时冻结 obs/action/old logp/old Beta 参数及索引，防 history 原地覆盖。
- 第一个真实 rollout 全部512样本在任何 actor/critic optimizer.step 前重算，
  校验 shape、分布、logp和ratio。容差运行前固定；失败则零optimizer更新退出。
- 首个update的8个minibatch前后，用相同512样本固定探针记录解析Beta KL、
  logp/ratio、mean/std/xyz修正、梯度裁剪前后范数、参数增量、advantage/return/value。
- 完整保留前两个update的真实输入及旧分布，之后只做有界预定日志。
- 保留旧approximate KL口径；诊断不采样、不消耗策略RNG、不改网络可变状态。
- 两组同样启用，不临时只给某组加KL early-stop、回滚或其他方法。
- 补无更新一致性、样本错配、广播错误、history原地修改、同分布KL、
  gate失败零参数/optimizer状态变化、诊断开关不改变学习路径的CPU回归测试。

五、条件满足后实际完成 P0/PLR
准入：D7真实结果可比；无未解释的主导下层限制；无已确认未修复训练缺陷；
诊断代码通过审核；独立训练与评估提案已绑定并获真实授权。
若下层问题主导或关键证据不足，不强行短训；交付具体定位与最小下一步。

P0 actor LR=3e-4，PLR actor LR=1e-4；critic LR均为3e-4。
相同初始actor/critic权重、training seed=20260908、新建一致初态optimizer，
原R4前1024 policy steps零命令课程；32 env、horizon16、epoch1、minibatch64。
每组64 updates/32768 training transitions；总计65536。
其他reward/网络/Beta/观测/归一化/控制/课程/算法参数不变。
记录解析后配置及实际optimizer LR，证明唯一方法变量是actor LR。
新实验冷启动，不从u256续训，不伪装成原R4 strict resume。

实现独立pilot保存节点：init0000/u1/u2/u8/u16/u32/u64，每组7个，共14个。
实现两组u16/u64的Z-only开发评估，总计25600 nontraining transitions。
主节点u64，u16过程观察，不以结果为依据另选最优节点。
训练、评估的启动/初始化/GPU/墙钟/磁盘/模型调用硬上限必须单独提案，
不继承D7或旧R4额度；不得未获批自动重试。首次gate前采样也真实计账。

六、有效性判据与完成交付
pilot运行前固定改善与安全退化判据；不以loss或KL单独决定成功。
报告更新机制、xyz修正偏置、Z全窗口高度/水平漂移/速度、失败率和完整时域比例，
沿用原Z公式与子组规则；保留全部lanes结果，明确单seed证据局限。

最终交付一个入口报告 R5_PILOT_FINAL_REVIEW.md，链接JSON、图、候选diff、
CPU测试、真实job/attempt、配置差异、累计资源和原始证据。
首页逐项写 E0_ACCEPTANCE、D7_LIVE、B0_LIVE、CONTROL_ANALYSIS、PILOT_CODE、
P0_LIVE、PLR_LIVE、Z_EVALUATION、EXECUTION_AUTHORIZATION 的实际状态。
科学结论选择 PROMISING_FOR_BOUNDED_EXPANSION / NO_BEHAVIORAL_BENEFIT /
INCONCLUSIVE_AT_64_UPDATES / BLOCKED_BY_CONTROL_BASELINE / INVALID_EXPERIMENT。
用人话说明有效与否、依据、限制及下一项最小动作。
若建议扩大训练，提出对称预算、完整任务和独立seed验证方案，但本轮不启动。

不要重复交付“CPU_READY”代替已经获准且可执行的实际工作。
不要编造批准、作业、指标或运行时间；遇到权限缺口给出真实可批准对象，
遇到科学分支阻塞给出证据与明确决策，完成其他独立可做部分后交付。
```
