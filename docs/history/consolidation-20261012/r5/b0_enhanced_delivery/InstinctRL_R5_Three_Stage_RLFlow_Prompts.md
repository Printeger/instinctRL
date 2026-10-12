# InstinctRL R5 三阶段 rlflow 执行 Prompts

> 用途：以尽量少的人工介入完成 B0 诊断、底层控制修复以及 P0/PLR 短程训练。
>
> 使用方式：按顺序将每个阶段的完整 prompt 粘贴到一个新的 Codex 窗口。只有上一阶段达到文中指定终态，才发送下一阶段。
>
> 阶段划分：
>
> 1. 修诊断工具 + 独立预审 + 单次增强 B0 诊断。
> 2. 修底层控制 + B0 回归 + 必要时 D7-R 回归。
> 3. P0/PLR 共同基线确认 + 短程训练 + u16/u64 评估。

---

## 第一阶段：修诊断工具并完成一次增强 B0

将下面的完整 prompt 发送给新的 Codex 窗口：

```text
请使用 rlflow 创建一个新的 additive successor package，完成：

修复 B0 physics-step 诊断工具
→ 独立 CPU 预审
→ 若预审 PASS，自动执行一次增强观测 B0
→ 离线定位水平漂移根因
→ 独立审核最终因果结论

不要恢复或修改此前暂停的 package。读取并继承以下记录：

- /home/mint/rl_dev/AGENTS.md
- /home/mint/rl_dev/instinctRL_coordination/ACTIVE_WORKFLOW.json
- /home/mint/rl_dev/instinctRL_coordination/protocols/DUAL_AGENT_WORKFLOW_V3.md
- /home/mint/rl_dev/instinctRL_orchestrator/implementation_records/instinctrl-first-use-20260915/state/analyses/UNIFIED-H1-R4-E0002/J-880cd798ee89d6eb/20260927-b0-horizontal-root-cause-successor/B0_HORIZONTAL_ROOT_CAUSE_REPORT.md
- 同目录的 b0_horizontal_root_cause.json
- 前一阶段 ZERO_COMMAND_BASELINE_FIX_AND_VALIDATION_REPORT.md 及 JSON

本 prompt 是该阶段的明确用户授权。在以下硬边界内，授权 Commander/控制角色创建、批准和 release 准确 proposal，不要再向我请求中间计划批准。Engineer 不得自我审批；必须保留独立 Evaluator。

一、关闭全部四项 MUST_FIX

1. F-OBSERVATION-NOT-INTEGRATED

将 physics-step observer 真正接入正式 B0 control-diagnostic backend 的受支持 post-physics 边界。

必须证明：

- 正式 B0 路径实际调用 observer；
- 每个 physics step、每个 lane 只记录一次；
- 原控制、physics、reset 和 observation 顺序不变；
- observer 不修改控制状态、动力学状态、RNG、reset、actor history 或 simulator；
- 不以合成 hook 冒充正式 backend 集成。

2. F-SCHEMA-FAILS-CLOSED-BOUNDARIES

严格验证：

- case、policy step、physics step、lane、episode 连续性；
- 完整记录数应为 630 × 20 × 32 = 403,200；
- 重复、缺失、越界、乱序记录必须失败；
- 使用 per-body world pose 和 local COM transform 重建 articulated COM；
- rotor wrench 必须从 rotor signals、作用点和方向独立推导；
- momentum-derived residual 必须与相邻状态闭合；
- unavailable wrench 必须显式标记，禁止补零；
- reset 前最后一步及 reset 后多个 physics steps 必须包含完整 controller/rotor 状态。

此前 Evaluator 的七项负向反例必须全部被拒绝并成为回归测试：

- Z 范围外 policy_step；
- 重复 physics-step identity；
- 不足 403,200 行；
- 被篡改的 local COM transform；
- 与 rotor signals 不一致的 rotor wrench；
- 与动量变化不一致的 inferred force；
- 缺少 controller/rotor 状态的 reset payload。

3. F-CAUSAL-ANALYZER-NOT-SUFFICIENT

因果分析器必须直接读取原始逐步证据并自行计算：

- 总质量和 articulated COM；
- 各类 force/torque 在 COM 处的合成；
- rotor application point 和 moment arm；
- 线动量、角动量及其变化；
- known wrench 与 residual wrench；
- 姿态、加速度、速度和位置响应。

分别检验：

- COM/推力作用点不一致；
- 固定横向力；
- 固定力矩；
- gravity/mass；
- rotor/mixer/frame；
- drag/aero；
- contact/joint/constraint；
- reset-state；
- x/y/z 是否同源。

闭合不足必须返回 DIAGNOSIS_INCONCLUSIVE，不能根据调用者传入的汇总标量直接分类。

4. F-HANDOFF-NOT-EXACT

生成正式 unified-H1 runner 可精确验证的 handoff，包含：

- 准确 argv；
- authorization 和 attempt；
- experiment ID；
- checkpoint 路径、SHA 和 source identity；
- product/workload/runtime commit、tree、文件 hash；
- 当前 candidate diff；
- seed、Z/T/S、初始状态和命令课程；
- 输出 manifest；
- cleanup/app.close 绑定；
- 精确资源预算。

二、CPU 开发与复审方式

使用新的 --no-hardlinks 隔离 worktree。

运行：

- 全部正向单元测试；
- 上述七项负向测试；
- 正式 backend 接线测试；
- 只读性和调用顺序测试；
- handoff validate-only；
- 因果分析器的力学闭合测试。

允许最多一次有明确 finding 的 CPU-only Engineer rework 和再次 Evaluator review。它不是 live retry。禁止没有新证据的盲目循环。

本阶段 rlflow role calls 最多 8 次，只使用完成实现、预审、必要的一次 CPU rework、运行后分析和最终审核所需的最少调用。

三、条件式 live 授权

若且仅若独立 Evaluator 对四项 MUST_FIX 全部给出 PASS，本 prompt 授权立即执行一次、且仅一次增强观测 B0：

- live process starts：最多 1
- simulator initializations：最多 1
- seed：20260909
- cases：Z/T/S
- nontraining transitions：最多 20,160
- physics diagnostic rows：403,200
- training transitions：0
- optimizer updates：0
- checkpoint writes：0
- P0/PLR：不得注册或运行
- automatic live retry：0
- wall time：最多 910 秒
- GPU allocation：最多 910 秒
- diagnostic output：最多 16 GiB

保持原 B0 零速度目标、初始状态、控制频率、资产、控制器、质量参数、reset 和冻结安全标准不变。本次运行前不得修复质量或水平控制问题。

Evaluator 未 PASS、身份不匹配、预算不完整或 preflight 失败时不得启动。

四、运行后分析

运行结束后必须从逐 physics-step 数据给出：

- x/y 稳态漂移由哪个 force 或 torque 维持；
- 推力轴是否穿过 articulated COM；
- per-body 质量与 COM 对合力矩的贡献；
- gravity、rotor、constraint、drag/contact 和 residual wrench；
- reset 前后 controller/rotor 状态；
- x/y 与已知 z 质量不一致是否同源；
- 可以被独立复算的数值闭合；
- 明确、最小的产品修复建议。

运行完成不等于根因成立。若力学闭合不足，必须给出 DIAGNOSIS_INCONCLUSIVE。

五、最终状态

只能输出以下之一：

- ROOT_CAUSE_ESTABLISHED
- DIAGNOSIS_INCONCLUSIVE
- PRELAUNCH_REVIEW_FAILED
- LIVE_DIAGNOSTIC_FAILED
- POSTRUN_EVIDENCE_FAILED

不修改产品控制行为，不启动训练。

在新的分析子目录交付：

- B0_ENHANCED_LIVE_ROOT_CAUSE_REPORT.md
- b0_enhanced_live_root_cause.json
- 完整 raw manifest、时间序列和力学闭合结果
- 下一阶段可直接读取的 machine handoff

最终报告必须包含版本、测试、审核、job/evidence ID、预算、app.close、进程/GPU 清理和未解决风险。
```

只有第一阶段得到 `ROOT_CAUSE_ESTABLISHED`，才执行第二阶段。

---

## 第二阶段：修复底层控制并完成 B0/D7-R 回归

```text
请使用 rlflow 创建新的 additive product-repair package。读取最新的：

- B0_ENHANCED_LIVE_ROOT_CAUSE_REPORT.md
- b0_enhanced_live_root_cause.json
- ZERO_COMMAND_BASELINE_FIX_AND_VALIDATION_REPORT.md
- R5_CONTROL_CHAIN_DIAGNOSIS.md
- InstinctRL_E0_Review_D7_Run_and_R5_Development_Plan.md
- 当前 V3 workflow、角色状态和保护边界

前置条件：上一阶段必须为 ROOT_CAUSE_ESTABLISHED，并具有完整、可复算的 force/torque/COM 数值闭合。否则终止为 ROOT_CAUSE_NOT_ESTABLISHED，不猜测修复、不运行仿真。

本 prompt 是该阶段的明确用户授权。在以下边界内，授权 Commander 创建、批准和 release 准确 proposal，不要再次向我请求中间计划批准。

一、实现最小产品修复

必须修复：

1. 垂直质量不一致

- 删除或消除 Lee controller 使用 base mass 1.1583 kg、plant 使用 articulated mass 1.1815179586 kg 的双重来源；
- 使用经验证的完整 articulated mass 或一个唯一、受检查的权威配置；
- 增加质量与资产一致性测试。

2. 水平漂移

- 只修复上一阶段数值证明确立的具体根因；
- 若为 COM/作用点问题，按 articulated COM 正确计算力矩或施力；
- 若为关节、附属部件、frame、rotor、drag/contact 或 constraint 问题，只修改对应缺陷；
- 若 reset state 有问题，完整重置 rotor/controller 状态。

禁止：

- 放宽验收阈值；
- 删除启动段；
- 增加预热来绕过失败；
- 盲目调 gain；
- 用积分项掩盖质量或固定 wrench 错误；
- 修改 actor 输入、reward、课程、seed、网络或训练方法；
- 向 actor 加入 simulator truth；
- 修改历史 attempt 或冻结证据。

二、CPU 验证和独立审核

使用新的隔离 worktree，完成：

- 根因针对性测试；
- mass/COM/wrench 数值闭合；
- rotor/mixer/frame；
- reset state；
- B0 control contract；
- D7 action-to-control contract；
- analyzer 和 cleanup 相关回归；
- 受影响测试集。

必须由独立 Evaluator 审核：

- 修复与已建立根因一致；
- 没有扩大到方法变化；
- 没有通过阈值或日志修改伪造 PASS；
- B0/D7-R 配置可比；
- live proposal 身份和预算完整。

允许最多一次有明确 finding 的 CPU-only rework 和复审；不允许 live 自动重试。rlflow role calls 最多 8 次，并使用最少必要调用。

三、条件式 live 回归授权

若独立 Evaluator PASS，自动执行修复后的 B0：

- starts/initializations：最多 1/1
- Z/T/S
- nontraining transitions：最多 20,160
- training/optimizer/checkpoint：0/0/0
- automatic retry：0
- 保持冻结 seed、初始状态、命令课程和安全阈值

B0 必须证明：

- 零目标速度下不再持续下降；
- 水平漂移满足冻结标准；
- 不发生全 lane below-height reset；
- 无非有限值、控制饱和或证据失效；
- app.close 和 cleanup 正常；
- 无 simulator、GPU、worker、daemon 或 lifeguard 遗留。

若 B0 失败，立即停止，不运行 D7-R。

若 B0 PASS，独立 Evaluator 判断底层控制/plant 语义是否影响旧 D7-R 的严格可比性：

- 若影响，自动运行一次同版本 D7-R；
- 若不影响，必须给出书面、可审核理由，允许不重跑。

为可选 D7-R 预授权：

- 额外 starts/initializations：最多 1/1
- 额外 nontraining transitions：最多 20,160
- automatic retry：0
- 与修复后 B0 使用完全相同的产品、runtime、seed、初始状态和场景

本阶段总 live 上限：

- starts/initializations：最多 2/2
- nontraining transitions：最多 40,320
- training transitions：0
- optimizer/checkpoint：0/0
- automatic retries：0

四、最终分类

只能输出：

- READY_FOR_PAIRED_PILOT
- STILL_BLOCKED_BY_CONTROL_BASELINE
- REPAIR_REVIEW_FAILED
- B0_REGRESSION_FAILED
- CLEANUP_FAILED
- EVIDENCE_INCONCLUSIVE

即使 READY_FOR_PAIRED_PILOT，也不要启动 P0/PLR。

交付：

- LOWER_CONTROL_BASELINE_REPAIR_REPORT.md
- lower_control_baseline_repair.json
- 产品 commit/tree/diff
- CPU 测试和独立审核
- B0 及必要 D7-R job/evidence
- 资源与 cleanup 审计
- 精确的 paired-pilot frozen baseline handoff
```

只有第二阶段得到 `READY_FOR_PAIRED_PILOT`，才执行第三阶段。

---

## 第三阶段：P0/PLR 短程训练、评估和扩大训练决策

```text
请使用 rlflow 创建新的 paired-pilot package，完成：

共同基线确认
→ P0/PLR CPU 和配置审核
→ 条件式短程训练
→ u16/u64 Z 评估
→ 是否值得扩大训练的结论

读取：

- 最新 LOWER_CONTROL_BASELINE_REPAIR_REPORT.md
- lower_control_baseline_repair.json
- paired-pilot frozen baseline handoff
- InstinctRL_E0_Review_D7_Run_and_R5_Development_Plan.md
- E0、D7-R/B0、首次更新诊断相关记录
- 当前 V3 workflow 和角色边界

前置条件必须为 READY_FOR_PAIRED_PILOT，且修复后 B0、cleanup、产品身份和必要 D7-R 比较已经通过。否则终止，不训练。

本 prompt 是该阶段的明确训练与评估授权。在以下边界内，授权 Commander 创建、批准和 release 精确 proposal，不要向我请求中间计划批准。

一、共同基线

P0 与 PLR 必须使用：

- 同一修复后 product commit/tree；
- 同一 workload/runtime；
- 同一初始 actor 和 critic 权重；
- 同一 seed 20260908；
- 同一环境数 32；
- rollout 16；
- epoch 1；
- minibatch 64；
- gamma .99；
- GAE lambda .95；
- PPO clip .2；
- entropy .01；
- grad clip .5；
- 同一 reward、网络、Beta 参数化、课程、控制链和 reset；
- 原始前 1024 policy steps 全零命令。

唯一方法变量：

- P0 actor LR = 3e-4
- PLR actor LR = 1e-4

两组 critic LR 均为 3e-4。不得顺便修改 reward、advantage normalization、网络、课程、控制器、KL early-stop 或其他训练参数。

二、首次 rollout 强制门槛

在每组第一次 optimizer.step 前，对全部 512 个真实 rollout 样本检查：

- 保存的真实 observation 快照；
- sampled action；
- old Beta concentration；
- old logp；
- current actor 对同一 observation/action 重算的 logp；
- delta_logp；
- ratio；
- shape、index、mask、dtype、device 和 model mode。

要求参数未更新时：

- delta_logp 在预注册 float32 容差内接近 0；
- ratio 接近 1；
- Beta 参数一致；
- 没有广播、索引、history 原地覆盖或 action clamp 差异。

任何一组失败都必须在其首次 optimizer.step 前停止。不得降低 LR 或放宽容差后重试，不得把另一组结果宣称为有效配对结论。

三、固定探针诊断

保存首个 rollout 的固定 512 样本，在第一个 update 每个 minibatch 前后记录：

- old/new logp 和 ratio 分位数；
- Beta 解析 KL；
- action mean/std；
- v_corr；
- actor 梯度裁剪前后范数；
- 参数增量；
- raw/normalized advantage；
- return/value；
- minibatch 样本索引；
- critic/actor 更新顺序。

诊断必须 no_grad、不采样、不改变 RNG、optimizer、model mode 或训练数据。

四、CPU 审核

训练前运行：

- 概率自洽正向测试；
- action/index 错配负向测试；
- history mutation；
- 广播 shape；
- 解析 KL；
- P0/PLR 配置 diff；
- checkpoint/reload；
- 预算、身份和 cleanup validate-only。

独立 Evaluator 必须确认：

- 唯一变量确实只有 actor LR；
- 使用修复后的共同底层基线；
- 首次 rollout gate 能在 optimizer 前 fail closed；
- 配置、checkpoint、evaluation 和资源身份完整。

允许一次 CPU-only rework 和复审；Evaluator 未 PASS 不得训练。

五、训练授权

若全部 preflight PASS，授权：

P0：

- 64 updates
- training transitions：32,768

PLR：

- 64 updates
- training transitions：32,768

合计：

- training transitions：最多 65,536
- optimizer updates：按现有每 update 契约精确计账
- automatic retries：0
- 不从 u256 resume；两组均为相同冷启动
- 不允许只给其中一组追加预算

每组保存：

- init0000
- u1
- u2
- u8
- u16
- u32
- u64

checkpoint writes 总上限：14。

使用当前正式 runner 支持的最少启动次数。训练进程最多 2 次；任何 job 失败不自动重试。

wall/GPU/disk 上限由 Codex 根据当前正式 runner、最近同类作业实际用量和上述固定 transitions，在 release 前生成明确机器预算并冻结。不得以此扩大 transitions、updates、checkpoint 或实验数量；若无法从现有证据形成安全上限，停止为 RESOURCE_BUDGET_NOT_PROVABLE，而不是询问多轮方案。

六、评估授权

只有对应训练成功且 checkpoint/hash 完整时，评估每组的：

- u16
- u64
- 原 Z，200 steps × 32 lanes

总评估上限：

- 4 个 checkpoint evaluations
- nontraining transitions：最多 25,600
- 使用 runner 支持的最少进程，最多 4 次评估启动
- automatic retries：0

沿用冻结 Z 指标及公式，不修改阈值、分母或子组：

- 速度；
- 水平漂移；
- 高度偏差；
- 失败和完整时域比例；
- 控制修正偏置；
- 期望与实际速度；
- 首次 update KL/ratio/参数变化。

七、最终结论

只能输出：

- EXPANSION_SUPPORTED
- LR_REDUCES_UPDATE_BUT_NOT_BEHAVIOR
- NO_LR_BENEFIT
- FIRST_ROLLOUT_PROBABILITY_GATE_FAILED
- TRAINING_OR_EVALUATION_FAILED
- PAIRED_COMPARISON_INVALID

只有当 PLR 同时满足：

- 首次更新概率路径正确；
- 早期 KL/clip/参数变化更温和；
- v_corr/control bias 改善；
- Z 真实飞行行为改善；
- 没有共同底层控制失效；

才能给出 EXPANSION_SUPPORTED。

即使 EXPANSION_SUPPORTED，也不得自动扩大到完整训练或更多 seed，只输出下一阶段的精确建议。

交付：

- P0_PLR_PAIRED_PILOT_FINAL_REPORT.md
- p0_plr_paired_pilot_final.json
- 配置 diff
- checkpoint manifest
- 首次 rollout 概率证据
- 固定探针记录
- u16/u64 评估结果
- 完整资源和 cleanup 审计
```

---

## 终态与人工介入规则

- 第一阶段只有 `ROOT_CAUSE_ESTABLISHED` 才进入第二阶段。
- 第二阶段只有 `READY_FOR_PAIRED_PILOT` 才进入第三阶段。
- 第三阶段只有 `EXPANSION_SUPPORTED` 才值得申请完整训练或更多独立种子。
- 阶段内部的普通 CPU 修复、一次有具体 finding 的 rework、复审、机器 proposal 注册和满足门槛后的运行已经由对应 prompt 预授权。
- 只有需要改变研究方法、actor/reward、冻结验收标准、runtime/system、实验数量或资源上限时，才重新请求用户决策。
