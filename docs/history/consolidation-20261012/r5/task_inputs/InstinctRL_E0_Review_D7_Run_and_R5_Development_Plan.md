# InstinctRL：E0 结论、D7-R/B0 实际执行与下一次训练开发任务

> 版本：v1.0 · 2026-09-23  
> 面向：研究负责人、Codex、项目现有 Commander/Executor。  
> 本文是开发与研究决策任务书，不是已经执行的仿真报告，也不是机器运行授权。  
> 当前材料对应状态：**E0_COMPLETE / D7_CPU_PREPARED / LIVE_NOT_AUTHORIZED / R5_NOT_ENTERED**。

## 0. 先读这一页：现在到底完成了什么

**E0 已完成；D7-R/B0 完成的是代码实现与 CPU 测试，不是仿真对照实验。**

`R4_REVIEW_ADDENDUM.md` 首段、`r4_d7_candidate_cpu_results.json.status` 和 `R4_D7_CONTROL_DIAGNOSTIC_PROPOSAL.json.status` 对此一致。因此，本轮不能报告“B0 悬停成功”“策略比 B0 差”或“Lee 控制器正常”。截至上传快照，尚没有这些新轨迹。[A §1、§9–10；C；P]

最短推进路径：

```text
接受已经完成的 E0，不重复全面审计
    → 复用 D7 候选，完成必要分析器和现有运行授权交接
    → 实际执行 D7-R 与 B0 各一次（或读取已经存在的新作业结果）
    → 判断主要问题在策略修正，还是下层控制/初始化
    → 若适合研究学习过程，实施 P0 / PLR 短程配对训练
       训练首个真实 rollout 在任何 optimizer.step 之前检查概率一致性
    → 主比较节点 u64，改善有效后再扩展完整任务
```

**这一轮不要再生成一份覆盖整个仓库的“训练前调研报告”。** 交付应是可执行的有限任务、实际运行证据、明确分支，以及可以提交的下一次训练配置。

---

## 1. 证据范围与来源

### 1.1 来源约定

| 标记 | 文件 | 用途 |
|---|---|---|
| A | `R4_REVIEW_ADDENDUM.md` | E0 结论、候选实现、授权与信号边界 |
| E | `r4_review_e0.json` | 本轮机器检查结果 |
| S | `r4_review_e0.py` | E0 检查的范围、公式与局限 |
| C | `r4_d7_candidate_cpu_results.json` | 8 项 CPU 测试与 validate-only 结果 |
| P | `R4_D7_CONTROL_DIAGNOSTIC_PROPOSAL.json` | 两个诊断条件、身份、预算与前置要求 |
| Plan | `InstinctRL_R4_Review_and_R5_Decision_Plan.md` | 上一轮研究决策、原指标及单变量试验 |
| T | `training-action-diagnostics.json` | R4 首个 rollout 的动作、奖励与终止摘要 |
| R | `R4_TO_R5_RESEARCH_REPORT.md` | R4 方法与实际训练参数的源码调研报告 |

本次复核读取了上传文件、检查了脚本、解析了 JSON 并核对了文件摘要；**没有连接用户机器、重跑依赖原日志的 E0 程序、加载原网络执行推理或运行仿真**。文件一致性不等于独立重新验证全部历史计算。

### 1.2 上传文件的已核对摘要

```text
r4_review_e0.json
4210375fac4a16dd83a023e5fc4c3750474bfe80355634b969088ec411b6ea1f

r4_review_e0.py
3c2e67fdbf27fb52765be1faed2866c816452d621aa51a240aa4d853a9098716

r4_d7_candidate_cpu_results.json
0780651151f24440cb6c2758f4b7f5e65e909116c16538cc2ccacf7ee4c5046f

R4_D7_CONTROL_DIAGNOSTIC_PROPOSAL.json
62992b2242e6aca4a9d4d999aab59f57a3c2e650f5208c132c8cebcd3825d703
```

这里的最后一个 SHA 是**上传提案文件的字节摘要**。若 rlflow 注册时包装成另一种提案对象，必须记录注册对象的真实 SHA 与该来源文件的绑定关系；不能直接假定两个 SHA 相同，也不能使用占位 ID 作为真实批准。

---

## 2. 本轮结果的研究解释

### 2.1 已经解决的问题：minibatch 编号不是训练缺陷

E0 确认 2,048 条 minibatch 记录、256 个 update，每个 update 的局部编号为 0–7；样本 coverage 与两套 optimizer 累计计数符合记录契约。旧检查把局部编号当成全局编号，属于 `DIAGNOSTIC_ASSUMPTION_ERROR`。[A §3；E.e0.minibatch_update_optimizer]

**决策：关闭“因为这个 false 所以训练漏更新”的假设。不修改 PPO 采样或更新次数，不反复重做相同审计。**

### 2.2 动作翻译在已记录范围内正确

131,072 个训练 lane-samples 的复算结果：[E.e0.action_control_mapping]

| 映射 | 最大绝对误差 |
|---|---:|
| `command_keep = action_sample[0]` | 0 |
| `v_corr_b = (2*action_sample[1:]-1)*0.5` | 1.49e-8 |
| `v_gov_b = clip_norm(command_keep*v_cmd_b+v_corr_b, 2)` | 4.88e-8 |
| `v_final_b = clip_norm(ics_beta*v_gov_b, 2)` | 4.88e-8 |

通俗解释：**网络给出的数，按既定公式被正确翻译成了期望速度。** 这降低了简单缩放/索引错误的嫌疑；不代表网络给出的数合理，也不代表下层真正执行正确。

实际速度与期望速度不是代数恒等。E0 的同/前一行速度误差约 0.2373/0.2320 m/s，是混合状态下“每样本最大分量绝对误差”的均值，不是稳态 RMSE，不是已经定位的一步延迟，也不能直接用来宣布下层跟踪失效。[A §4；S.mapping]

### 2.3 最大的新信息：异常在第一次 update 内就出现

以下按局部 minibatch 顺序重排 E0 的 top-10，而不是新的运行结果：[E.e0.early_kl_top10]

| update | local minibatch（从 0 编号） | approximate KL | ratio mean | clip fraction |
|---:|---:|---:|---:|---:|
| 1 | 1 | 3.792447 | 0.334185 | 0.984375 |
| 1 | 2 | 8.097852 | 0.203909 | 1.000000 |
| 1 | 3 | 14.783306 | 0.079928 | 1.000000 |
| 1 | 4 | 17.097183 | 0.015015 | 1.000000 |
| 1 | 5 | 15.290251 | 0.029965 | 0.984375 |
| 1 | 6 | 20.124577 | 0.003833 | 1.000000 |
| 1 | 7 | 20.454645 | 0.000134 | 1.000000 |

由此可确认：**早期最大的 7 个记录异常，全部属于第一次 update 的后续 7 个 minibatch。** 不是训练几十轮以后才出现的现象。[A §5]

其含义是：若保存的概率和样本对应正确，新策略很快就与采集该批数据时的策略明显不同。`clip fraction=1` 只说明该统计下全部样本 ratio 越过裁剪区间，**不是“所有梯度都为零”**。

但历史日志没有 old/new log-probability 与 Beta concentration，E0 没有完成逐样本概率重放。所以暂时应写成：

> **高优先级工作假设：首次更新过大，或更新时的数据/概率处理存在异常。**

不能写成“已经证明 actor LR 太大”。同一 update 的不同 minibatch 使用不同样本，表格也不是同一固定观测集合上的连续测量；下一轮要额外设置固定探针集合。

### 2.4 新的交叉证据：最早的异常发生在终止惩罚之前

旧 T 文件的 `updates.1.zero_command` 覆盖首个 rollout 的全部 512 个样本：reset 均为 0，`reward_failure` 均为 0，below/above/collision/timeout 均为 0。[T]

与 E0 的 update-1 异常结合，得到一个**有边界的推论**：

> 第一次 update 的异常，不能直接解释成“这批数据已经被大量坠落和 -300 终止奖励污染”。

这不排除普通奖励、critic 初始估计、优势计算/尺度、输入处理、Beta 参数敏感性或学习率的影响，也不排除后续终止奖励放大问题。它只是把最早的问题时间点缩小到第一次参数学习。

### 2.5 GAE 相关 fixture 改善了信任范围，但不是整机证明

新增 CPU fixture 覆盖了非零 value、rollout tail、pure timeout、terminated 优先级、reset 递归边界及 terminal/reset observation 区分，并均通过。[A §7；E.e0.cpu_gae_terminal_fixtures]

**决策：这些已测试路径不再作为当前最高优先级；不要宣称整个训练器因此绝对无 bug。**

### 2.6 D7-R/B0 的 CPU 通过意味着什么

8 项测试和 validate-only 证明的是候选入口、固定动作路径、信号写入契约等在所测 CPU 条件下成立。文件明确列出未测试：仿真初始化、GPU 推理、真实环境交互、live Lee/wrench 信号及 rlflow 提交。[C.tests、C.not_tested]

**诊断程序准备好 ≠ 对照实验完成 ≠ 无人机控制通过。**

---

## 3. 本轮开发范围：三件事，不重建工作流

| 工作包 | 本轮应完成的东西 | 不做什么 |
|---|---|---|
| W1：诊断执行交接 | 复用候选；读取最新作业状态；完成实际注册/授权交接；获批后执行两条件各一次 | 不再只返回“CPU_READY”；不绕过 STOP/授权；不重复已有作业 |
| W2：控制链分析器 | 从真实结果自动生成两条件对比和下一步分支 | 不以预算充当实际计数；不以 fixture 充当轨迹 |
| W3：短程训练准备 | 隔离实现首次更新概率检查与 P0/PLR 配对配置 | 不在 D7 下训练；不同时改 reward/network/curriculum |

W2/W3 的 CPU 开发可以与 W1 的批准交接并行准备；**实际 R5 训练仍须等待诊断结果和独立的运行授权。**

---

## 4. W1：把已经写好的 D7-R/B0 真正执行起来

### 4.1 复用现有候选

```text
D7 candidate worktree:
/home/mint/rl_dev/worktrees/instinctrl-r4-d7-control-diagnostic-20260923

candidate commit:
c7a4d79e35916ce2711085674b9cd37f1054a1ab

candidate tree:
1e06553c95d9f34725179919165b28086d81778d

candidate config SHA256:
97e2345cc9cd5637659eea3218b5c68fd914fbf8cea0c24f2fdbc2d8b0fe47c0

frozen product:
/home/mint/rl_dev/worktrees/instinctRL-unified-h1-r3-d0011-product
product commit: 903afc5abd0fbf82297527e5790c2cd40827b0b2

u0256 checkpoint SHA256:
39ad6806332376b655ee2b37bcf573997a804ef6f9324a2fe0c367fcaaa9886c
```

来源：[A §2、§9；P.candidate]。这些是已有来源身份，**不是本文件创建的新版本**。

执行前只需定向检查必要信号是否已实际绑定到输出字段、对应时刻和坐标系。已足够就不改候选。必须补日志字段时使用新提交并更新真实提案绑定；不能对已批准的旧 SHA 静默换代码。分析器和 R5 训练改动放在独立开发位置，不污染该冻结诊断候选。

### 4.2 先查有没有新结果，防止重复跑

读取项目当前状态和这两个 experiment 对应的 job/attempt，而不是把上传快照的 `enabled=false` 当成机器永远的状态：

```text
UNIFIED-H1-R4-D7-R
UNIFIED-H1-R4-B0
```

已有完成作业时直接分析；已有 active/uncertain 作业时只查状态，不重新提交；未运行且无授权时进入现有授权交接。不要因为用户说“已完成”就编造作业，也不要因为附件较旧就重复启动。

### 4.3 两个条件完全沿用提案

| 项目 | D7-R | B0 |
|---|---|---|
| 动作 | u0256 的确定性 Beta mean | `[1.0,0.5,0.5,0.5]` |
| 控制路径 | 原 H1 governor→ICS→Lee→rotor | 相同 H1 路径，不使用 legacy direct-B0 |
| 场景 | 原 Z/T/S，各自 fresh reset | 相同场景、初始化与约定随机状态 |
| policy steps | 200+100+330=630 | 630 |
| nontraining transitions | 20,160 | 20,160 |
| optimizer/checkpoint writes | 0/0 | 0/0 |

B0 的四维数字是 **governor 输入，不是电机指令或油门**。它表示不削弱低速原命令、不附加三维速度修正；没有跳过下层安全和飞行控制。[P.conditions；A §9]

### 4.4 按现有提案办理授权，而不是再设计权限系统

当前缺口是 proposal 注册及批准、两个 machine authorization/fresh attempts、package/stage materialization 与显式 release。[A §10]

执行角色必须遵守项目当前的授权机制：Builder 不替用户/Commander 自行审批；负责控制面的授权角色处理 enable/release；不删除旧失败，不把旧 R3 HOLD 随意改成全项目 RUN。

若确实需要用户输入，集中交付**一个批准包**：真实 proposal ID/SHA、来源文件 SHA、两条件身份、总预算、准确批准入口。不要拆成多轮“请批准我准备下一份批准文件”。已获有效授权后继续执行约定任务，而不是再以“需要授权”重复停止。

不要猜测未确认的 CLI。读取本地实际接口；上传材料中的 `proposal show`、`approve --sha256`、`resume --package` 只是已知操作形式，注册/物化命令须以实际实现为准。直接 workload argv 是被调度任务的内容，不是绕过 rlflow 的手工旁路。

### 4.5 预算采用提案原值，不另加未获批试验

| 聚合资源 | D7-R + B0 上限 |
|---|---:|
| live process starts | 2 |
| simulator initializations | 2 |
| initialization ticks | 20 |
| nontraining transitions | 40,320 |
| training transitions / optimizer updates / checkpoint writes | 0 / 0 / 0 |
| explicit resets | 6 |
| action GPU seconds | 1,800 |
| 含清理 outer wall seconds | 1,820 |
| disk | 16 GiB |
| automatic retries | 0 |

每条件 900 action seconds、910 seconds 含清理。均为**运行硬上限，不是耗时预测，不是已经发生的资源使用量**。[P.aggregate_hard_caps]

不加入 Bz、额外 checkpoint 回放、额外仿真 smoke run、预热飞行或训练。任一中断保留部分证据和实际计数；不得重试到通过。若部分失败但另一条件已有结果，已有结果保留，不能把两条件都填成完成。

---

## 5. W2：必须交付的控制链分析器

建议新增独立 CPU 分析入口 `analyze_d7_control_diagnostics.py`（名称为建议），读取真实 attempt 清单与分块轨迹；不要复制整套训练程序。

### 5.1 输入与时间语义

每条样本使用 `(condition, case, lane, episode_id, policy_step, timestamp)` 关联；明确 action 前、physics 后、reset 前/后的字段。

必需内容：原命令、issued action、actor 分布参数/mean、`v_corr_b`、`v_gov_b`、`v_final_b`、ICS 状态、root 位置/姿态/速度、终止与 reset、可用的 Lee/rotor 信号。分布 `beta` 与 `ics_beta` 使用不同字段名。[P.signal_contract]

将期望速度与实际速度放到**一致坐标系**。若需要 world-frame 期望速度，优先使用下层当时真正采用的值；不能用 physics 后的 quaternion 冒充控制下发时的 quaternion。只能重建近似值时明确误差来源，不能对微小偏置作确定归因。

现有代码仅保证部分 step 后 cache；要确认它对应哪个 decimation 子步。没有子步数据时不把最终 cache 写成整段 physics 的平均值或全时域完整信号。

### 5.2 reset lane 缺失不会自动毁掉整个诊断

`downstream_cache_valid=false` 的样本不能用于 Lee/rotor 定量归因，也不能补成 0。该 lane 的真实终止、失败和可用 pre-reset 状态仍须保留。[A §9]

逐条件/案例报告 valid/invalid 数量。先分析首次连续段和未 reset 的片段；如果它们已经能回答主要问题，不因为个别终止步缺 cache 就再跑全部实验。只有关键结论无法区分时，提交**缺字段的最小补采方案**。

### 5.3 每个案例需要的输出

| 类别 | 输出 |
|---|---|
| 执行完整性 | 实际 job ID、起止/退出状态、步数、transitions、reset、各信号有效数量、来源清单 |
| 策略输入/输出 | D7-R 的动作 mean/issued action、Beta 参数、零命令修正的有符号均值与绝对量；B0 实际 issued action 恒定检查 |
| 控制执行 | `v_final` 与实际速度的分轴偏置、RMSE、完整时间曲线；有姿态/时间限制则标明 |
| 悬停 | 全窗口最大高度偏差、水平漂移、速度、首次失败时刻；按原指标定义另算诊断性分数 |
| 跟踪与停止 | 非零段跟踪、切零后的速度与位置；S 前置跟踪不成立时不得把 fallback 当实测停止时间 |
| 初始化 | t0 root/速度及可用 rotor 初值；启动损失是否停止积累，稳态是否仍持续爬升/下降 |
| 比较单位 | 先逐 lane，后按案例和方向汇总；不把 32 lanes 当 32 个独立训练种子 |

启动/后段切分沿用 Plan 的建议 `0–2.5 s` / `2.5 s 以后`，仅辅助诊断，**不是把前 2.5 s 从原评分删掉**。窗口未达到后段或已 reset 时标无有效后段，不硬填稳态量。

原 Z/T/S 指标可以从新轨迹离线计算作参考，但结果标签必须是 `DIAGNOSTIC_REFERENCE_ONLY`：本次不执行或写回 formal verdict，不以该运行授予 R5/部署资格。若冻结授权连参考指标计算也未涵盖，由后处理权限单独覆盖，不扩大仿真范围。

### 5.4 D7-R 对历史 u256 的复现核对

只读比较新旧 Z/T/S 的初始化、命令、主要轨迹与失败模式；允许记录平台数值差异，但不能隐藏明显不一致。若新回放不再表现出原失败，应先说明控制版本/初态/随机状态/日志干预的差异，不能直接拿新 B0 与不一致的旧结果做因果比较。

不要求逐 bit 物理轨迹复现作为任何分析的前提；要求可比条件成立及偏差透明。无需为了证明 bitwise 一致而反复跑仿真。

### 5.5 输出三个文件即可

```text
R5_CONTROL_CHAIN_DIAGNOSIS.md
r5_control_chain_diagnosis.json
R5_PILOT_READINESS.md
```

图放同目录，少量即可：Z 高度、Z 期望/实际垂直速度、Z 水平漂移，以及 T/S 的代表响应。保留所有 lane 的机器结果，不能只展示最好 lane。

首页必须写清：

```text
execution_status: NOT_STARTED / COMPLETE / PARTIAL / FAILED
control_conclusion: UPSTREAM_RESIDUAL_DOMINANT /
                    DOWNSTREAM_OR_RESET_DOMINANT /
                    MIXED_CAUSES / INSUFFICIENT_SIGNAL
scientific_next_action: <一个明确的下一步>
training_approved: false 或真实授权状态
```

这些是新报告标签，不更改冻结方法的验收契约。

---

## 6. D7 结果如何决定下一步

| 实际结果 | 可以作出的解释 | 下一步 |
|---|---|---|
| B0 能执行约定速度；原策略期望速度带持续偏置，实际运动随之偏离 | 上层学出的修正是主要限制；尚未证明具体学习根因 | 优先准备 P0/PLR，并在首个真实 rollout 核实概率一致性 |
| B0 收到固定/零速度仍持续偏离，或出现控制饱和/非有限值 | 下层、坐标语义、初始化或控制职责存在限制 | 先定位一个确定的下层问题；修复后固定动作回归，再重建可比 RL 基线 |
| B0 只有启动损失，后段速度正常但位置未恢复 | 启动瞬态或速度环不恢复位置偏移；不等同于持续控制偏置 | 明确初始化与位置恢复职责；不能删启动段或偷偷预热来通过 |
| 两条件都存在不同程度偏差 | 多因素，不能只选一个责任模块 | 先使下层误差可解释/可控制，再做学习变量对照 |
| 可用信号不足，确实无法比较期望与实际响应 | 证据不足，不是算法失败 | 只补影响分支的字段；不重写全项目调研 |

**B0 不需要因为叫“基准”就被预设为成功；B0 也不能仅靠“没有碰撞”宣告控制正常。** 完整位置/速度曲线及启动行为必须一起判断。

---

## 7. W3：首次 PPO 更新的可诊断实现（CPU 开发现在就做）

这项开发是本轮针对新证据的重点。放在未来配对训练使用的隔离分支，**不要把优化更新添加进 D7 候选**。

### 7.1 为什么 D7 不能代替这项检查

D7 使用 u256 做确定性推理，没有优化更新。它能定位当前飞行的控制来源，却不能重现“u0 第一次 stochastic rollout 的第一次参数更新”。拿 D7 数据反向训练来推断历史 update-1，也改变了数据分布与初始化，不是忠实重放。

无需永久等待不可恢复的历史输入。下一次获批训练在自身第一个真实 rollout 上完成这一项运行前置检查即可。[A §5–6；Plan §5.2]

### 7.2 首次更新之前必须对同一输入、同一动作自洽

在首次 `optimizer.step()` **之前**，用该真实 rollout 的全部 512 个样本，重新计算：

```text
logp_new_before_update = current_actor.log_prob(saved_real_obs, saved_sampled_action)
delta_logp = logp_new_before_update - saved_old_logp
ratio = exp(delta_logp)
```

要求在预注册数值容差内 `delta_logp≈0`、`ratio≈1`；同时对比采样时与重算时的 Beta 参数。若尚未学习，同样的输入和动作已得出不同概率，应先修正数据/概率路径，而不是降低 LR 掩盖。

实现要求：

1. 保存实际用于采样的输入快照或逐元素等价重建；禁止 history buffer 后续原地修改覆盖旧样本。
2. old concentration、old logp、采样 action 和关联索引冻结并 detach；不得更新时重算 old logp 替代旧值。
3. 与原代码保持相同的动作 clamp、四维 log-density 求和、flatten/环境索引顺序与有效样本 mask。
4. 检查广播形状；不能把 `[N]` 和 `[N,1]` 静默组合成 `[N,N]`。
5. 记录 dtype、device、train/eval 模式、输入归一化/可变网络状态。额外记录不改变网络状态或消耗策略 RNG。
6. 使用分块前向控制显存。容差在运行前根据同路径重复计算确定；Plan 的 `1e-5` 是 float32 候选，不是失败后可任意放宽的门槛。
7. 失败时在任何 optimizer 更新前停止该次学习，保留真实失败样本、配置和计数。

这条 `ratio≈1` 针对**一个 rollout 的任何参数更新之前**，不要求后续第 2–8 个 minibatch 仍等于 1。

### 7.3 第一次 update 使用固定探针集合

保存首个 rollout 的全部 512 个真实 `(obs, sampled_action, old_distribution, old_logp)`。在第一个 update 的每个 minibatch 更新前/后，对同一固定集合额外计算：

| 指标 | 用途 |
|---|---|
| old/new logp、ratio 的分位数与最大偏差 | 判断旧动作密度如何改变 |
| 四维 Beta 的解析 `KL(old || new)` 及合计 | 区分真实分布变化与单个采样 KL 估计 |
| action mean、标准差、`v_corr` 的变化 | 判断大 KL 是否对应不合适的控制偏置 |
| actor 梯度裁剪前后范数、实际参数增量 | 判断参数更新幅度及敏感性 |
| 原始/标准化 advantage、return、value 统计 | 检查学习信号是否退化或被异常值主导 |
| minibatch 样本索引、critic 更新前后阶段 | 确保能定位到哪一步改变 |

解析 KL 仅作为新诊断量，不改变 PPO loss。PyTorch 的分布接口支持 Beta 与 KL 计算；以项目本地安装版本确认接口，不为诊断擅自升级依赖。[M2]

保留原 `mean(old_logp-new_logp)`，不要用新量覆盖历史口径。旧采样估计的小负值不自动等于实现错误；clip fraction 也不是“被完全阻断梯度的样本比例”。

### 7.4 记录预算与观测不干扰原则

建议前两个 update 保存完整 512 样本输入。按现有 shape 计算，单实验这两段 LiDAR float32 原始体积约 0.972 GiB；之后保存低维分布/控制统计和预定探针，避免全训练逐样本重复写巨量张量。

这是未来训练的新日志预算，不占用或继承 D7 的 16 GiB 上限。执行前分别配置。

诊断前向应 `no_grad`、不调用采样、不修改模型模式/归一化状态、不改变 optimizer 或训练数据。新增检查在 P0/PLR 中完全相同。数据错配或非有限值立即停止；**不能仅在某组大 KL 时临时加 early-stop/rollback**，否则已经不是 LR 单变量实验。需要 KL 截断时另立方法变更与授权。[M1]

### 7.5 最小 CPU 回归测试

| 测试 | 预期 |
|---|---|
| 参数不变的同输入/同动作重算 | logp/分布相符 |
| history buffer 后续原地改变 | 已保存快照不变 |
| 刻意错配一个样本 action 或索引 | 检查失败 |
| 刻意制造广播错误 | shape 检查失败 |
| 解析 KL 同分布 | 数值接近 0 |
| B0 分析输入伪造出非固定动作 | 分析器拒绝把它标为 B0 |
| 缺 manifest、少步数、NaN 或 reset-cache 无效 | 完整性失败或按已知局部缺失降级；不填 PASS |

合成输入允许用于这些**单元测试**，但不得标成历史真实轨迹重放。测试完直接进入交接，不扩展成无边界测试项目。

---

## 8. 如何开启下一次训练：条件满足后做 P0 / PLR

### 8.1 科学准入与执行授权分开

可以准备/提交配对训练的条件：

- D7-R/B0 已有真实可比结果；没有尚未解释、会主导该学习比较的下层控制/初始化限制。
- E0 相关检查已接受；没有已确认未修复的数据/更新实现缺陷。
- P0/PLR 仅 actor LR 不同，其余方法一致。
- 首个真实 rollout 的概率自洽检查已经实现，并作为第一次更新前的运行门槛。
- 预算、运行身份、停止规则和独立授权已明确。

**不要求先恢复不存在的 R4 历史概率记录，也不要求先证明所有 H1–H9。** 允许状态为 `READY_FOR_PAIRED_PILOT_WITH_FIRST_ROLLOUT_GATE`；运行后通过该 gate 才实际更新参数。诊断结果本身不充当用户/Commander 的训练批准。

### 8.2 首选实验：只改 actor 学习率

若控制对照指向学习策略，且新的概率检查通过，沿用 Plan 的首选候选：

| 参数 | P0 | PLR |
|---|---:|---:|
| actor LR | 3e-4 | **1e-4** |
| critic LR | 3e-4 | 3e-4 |
| 环境数 | 32 | 32 |
| rollout / epoch / minibatch | 16 / 1 / 64 | 相同 |
| gamma / GAE lambda / clip | .99 / .95 / .2 | 相同 |
| entropy / grad clip | .01 / .5 | 相同 |
| reward、网络、Beta 参数化、控制链 | 原 R4 | 原 R4 |
| 课程 | 原始前 1024 policy steps 全零命令 | 相同 |
| 更新数 | 64 | 64 |
| training transitions | 32,768 | 32,768 |

`1e-4` 是待验证候选，不是已经发现的最优值。[Plan §8；R §6.2]

两个实验使用同一已核对的初始 actor/critic 权重、同一训练 seed `20260908` 和一致的初始 optimizer/环境契约。属于新的冷启动实验，不从 u256 继续，也不伪装为原 R4 的 strict resume。

新日志与检查两边相同；不顺便改 advantage normalization、观测归一化、奖励、Beta 范围、动作初始化、rollout、epoch、课程或控制器。若发现真实实现缺陷，先修复，并把两个组都放在修复后的共同基线上，不能把 bug 修复的收益算给 LR。

### 8.3 节点、预算与指标

建议保存 `init0000`、u1、u2、u8、u16、u32、u64，共 7 个节点/组；u1/u2 是本轮针对首次异常新增的检查点。写盘数量须进入新的训练授权，不能继承 D7 的零 checkpoint 约束。

主比较为 u64；u16 作为过程诊断。两个实验的 u16/u64 分别运行原 Z：

```text
训练：2 × 64 × 16 × 32 = 65,536 training transitions
开发评估：2 × 2 × 32 × 200 = 25,600 nontraining transitions
```

公共 u0 的额外评估不是默认必需；使用时另计 6,400 transitions。wall/GPU/初始化/磁盘/进程预算根据本地入口设置单独绑定，不自动从 D7 预算延用。

指标分三层：

1. **更新**：首轮概率一致性、固定探针 KL、参数变化、clip、advantage/value。
2. **控制**：零命令下 xyz 修正偏置，期望与实际速度偏差。
3. **行为**：Z 全窗口高度最大偏差、水平漂移、速度、失败/完整时域比例。

原 Z 阈值为速度 ≤0.08 m/s、水平漂移 ≤0.30 m、高度偏差 ≤0.20 m；公式、分母与子组要求必须沿用实际原实现，不凭这些三个数另造评分。[Plan §8.7]

### 8.4 结果应怎样推动研究

| 配对结果 | 解释 | 后续 |
|---|---|---|
| PLR 降低早期 KL，修正偏置与 Z 行为也改善 | 支持温和更新有助于基础控制 | 扩展相同单因素到完整预算/任务与独立种子 |
| KL 改善，但控制偏置与位置漂移不改善 | 更新变稳不足以解决任务 | 下一次只研究目标对齐或反馈，不继续盲降 LR |
| 概率检查在首次更新前失败 | 首先是数据/概率路径，不是 LR 优劣 | 修复并重建共同基线 |
| 两组 KL 很大但真实行为不随之恶化 | KL 不能单独充当性能判断 | 检查分维分布/动作映射；按行为决定，不把大 KL 自动等同坠落 |
| PLR 在 64 updates 进展较慢但持续改善 | 只能说该预算内学习速度不同 | 预注册相同追加预算，不单独给候选更多训练后宣称公平优胜 |
| 两组长期悬停已接近 B0，却仍有共同高度损失 | 重新评估基准的初始化/位置职责 | 不通过加训练预算掩盖共同限制 |

暂不默认加入高度奖励。若诊断/配对结果把主要矛盾转向长期高度目标，再做 P0/PZ 单变量比较；不同时合并 LR、reward、curriculum。

---

## 9. Codex 的完成标准与状态输出

### 9.1 不允许含糊的“全部完成”

每次交付首页必须分开写：

```text
E0: COMPLETE（引用已有结果，原则上不重跑）
D7_CODE: CPU_PREPARED / CHANGED_AND_RETESTED
D7_LIVE: NOT_AUTHORIZED / NOT_STARTED / RUNNING / COMPLETE / PARTIAL / FAILED
B0_LIVE: NOT_AUTHORIZED / NOT_STARTED / RUNNING / COMPLETE / PARTIAL / FAILED
CONTROL_ANALYSIS: NOT_AVAILABLE / COMPLETE / LIMITED
PILOT_CODE: NOT_PREPARED / CPU_TESTED
PILOT_SCIENTIFIC_READINESS: <状态与唯一最重要理由>
PILOT_EXECUTION_AUTHORIZATION: <真实状态>
```

没有实际 job/轨迹就不要写 `D7_COMPLETE`。没有权限不等于科研失败；学习任务失败不等于运行程序失败。

### 9.2 应交付的最小内容

W1：真实批准交接或已执行的两个 job/attempt、实际资源与退出状态；若未获批准，清楚列出一个需要批准的对象和准确操作。

W2：`R5_CONTROL_CHAIN_DIAGNOSIS.md/.json`（JSON 名称按 §5.5）、必要曲线和 `R5_PILOT_READINESS.md`。

W3：隔离代码 diff、CPU 测试输出、P0/PLR 两份解析后配置、只存在一个方法变量的配置差异清单。没有训练授权时不启动训练，但应交付可提交配置，而不是“后续可以考虑”。

不能为了状态整齐覆写冻结文档、旧 order、检查点或历史失败；追加结果即可。禁止自编运行结果、批准 ID、运行时间或指标。

---

## 10. 研究负责人现在可以采用的结论

> E0 关闭了 minibatch 编号误解，并验证了记录范围内的动作映射和若干关键 GAE 路径。新的重要线索是，最大的早期概率变化记录集中在第一次 update，且该 rollout 尚未出现终止惩罚，因此下一次学习应重点观察最初几次参数更新。D7-R/B0 尚未提供实际仿真结果，下层控制与策略修正的区分仍待完成。下一步不是再做全面调研，而是执行现成的控制对照、形成分支，再以带首次概率自洽检查的短程配对实验检验学习更新幅度。

这仍不是论文贡献证明。它是从“结果不好”推进到“能够区分根因、验证单个改进”的可执行研究步骤。

---

## 11. 可直接发给 Codex 的任务文字

```text
请执行 InstinctRL_E0_Review_D7_Run_and_R5_Development_Plan.md。

先纠正阶段状态：上传证据只证明 E0 完成、D7-R/B0 CPU 准备完成，
不是 D7-R/B0 仿真已完成。读取现有作业状态；若已有新结果，直接分析，
不要重复启动。如果没有，则复用 c7a4d79e... 的已有诊断候选推进实际运行交接。

本轮开发交付：
1. 复用已有 E0 与诊断实现，不重新做全面仓库调研、不重建工作流。
2. 完成有限的运行前信号绑定检查和 D7-R/B0 CPU 分析器。
3. 通过现有授权角色/接口完成真实提案注册、授权/attempt/package 交接。
   如果还需要用户批准，集中提供一个真实 proposal ID/SHA 与准确批准操作。
   不得自行审批或绕过 STOP；已有有效授权时继续执行，不重复停在准备阶段。
4. 获批后两个诊断条件各运行一次，合计最多 40,320 非训练 transitions，
   不训练、不写训练 checkpoint、不自动重试、不加入额外实验。
5. 根据真实轨迹生成 R5_CONTROL_CHAIN_DIAGNOSIS.md、
   r5_control_chain_diagnosis.json 和 R5_PILOT_READINESS.md。
6. 同时在隔离训练分支实现首次 rollout 的真实输入/概率一致性检查，
   以及固定探针的逐 minibatch KL/动作/参数记录，完成 CPU 测试。
7. 为 P0(actor LR=3e-4) 与 PLR(actor LR=1e-4) 准备同基线配对配置；
   其他方法不变。每组 64 updates，暂不在本轮自动启动训练。

明确区分 CPU 准备、仿真实际执行、科学结论、训练运行授权。
不要只再返回 CPU_READY；必须完成当前权限内的可交付实现与推进步骤。
缺权限时只指出具体批准对象；缺证据时只指出影响决策的必要字段。
```

### 11.1 用户通过现有 Commander 入口批准诊断时的范围示例

以下文字是供用户主动发送的范围表达，不代表本文件已经获得批准；机器层仍需绑定注册对象：

```text
我同意并授权通过项目现有 Commander/rlflow 流程实际执行
R4_D7_CONTROL_DIAGNOSTIC_PROPOSAL.json 所描述的 D7-R 与 B0 两个仿真诊断。
来源提案文件 SHA256：
62992b2242e6aca4a9d4d999aab59f57a3c2e650f5208c132c8cebcd3825d703

仅允许该提案的两条件和聚合上限：2 次启动/初始化，最多 40,320 次非训练交互，
0 训练更新、0 checkpoint 写入、0 自动重试，以及提案中的时间和磁盘上限。
请完成真实注册对象绑定及所需控制面授权，检查前置条件后执行并生成分析结果。
不授权扩大到 R5 训练，不授权修改算法、动力学、验收门槛或清除安全停止条件。
```

---

## 12. 一般方法参考（不替代项目证据）

[M1] OpenAI Spinning Up，Proximal Policy Optimization，Key Equations / You Should Know：PPO clipping 不是策略距离的硬保证，KL stopping 是独立机制。本任务先记录，不在 LR 单变量对照中悄悄添加它。

```text
https://spinningup.openai.com/en/latest/algorithms/ppo.html
```

[M2] PyTorch 官方 Probability distributions 文档，Beta 与 `torch.distributions.kl.kl_divergence`。仅用于实现诊断量；项目本地 PyTorch 版本须单独记录，不因文档版本更新依赖。

```text
https://docs.pytorch.org/docs/stable/distributions.html
```
