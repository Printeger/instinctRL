# InstinctRL 下一条自主 Goal 需求书：零命令残差漂移的归因与单变量验证

版本：2026-10-14；研究状态：`PAIR_GOAL_SCIENTIFICALLY_COMPLETE_NEGATIVE_Z_RESULTS`。

## 0. 目标与范围（一句话）

**在同一个 Goal 中连续完成：旧配对实验的 CPU 证据归因 → 选定一个可证伪机制 → 实现一个预注册的单变量修正 → CPU 回归 → 在单次明确的资源授权内运行配对 64-update Pilot 和原 Z 评价 → 独立审核及研究决策。**

不再把“做完一份离线报告”“准备好 CPU 代码”“等用户说继续”分别设为 Goal 终点。Goal 的结束条件是一次有比较对象、证据可信的科学实验结论，或者明确的无法在所给上限下补齐的实质阻塞项。**即使新方法失败，实验完成也是有效交付。**

这不是让代理无限迭代到通过，也不包含 256-update 正式长训、真机飞行、改动安全门槛或任意扩大预算。

## 1. 已核实的起点，不要重复调查

源代码/状态：`Printeger/instinctRL` 的当前 `main`；本 Goal 开始前应打印、固定实际 `HEAD`，本任务起草时 GitHub `main=436db8033ff4e0bb6e3659daf9265c4d47946023`。之前被验证的统一物理控制基线 `ef68ebb5682b85aa83044fdea1c13b7d0edbf325`，不代表最新 `main` 的所有文档改动都已成为训练输入。

必读：

- `docs/PROJECT_STATUS.md`
- `coordination/CURRENT_TASK.md`
- `coordination/protocols/SINGLE_BRANCH_WORKFLOW.md`
- `docs/experiments/r5-paired-pilot-20261013/PAIR_REPORT.md`
- `docs/experiments/r5-paired-pilot-20261013/pair_results.json`
- `docs/experiments/r5-paired-pilot-20261013/STATE_AND_EVIDENCE.md`
- `isaaclab-training/instinctlab/task/rewards_h1.py`
- `isaaclab-training/instinctlab/control/governor.py`
- `training_workloads/unified_h1/pilot_diagnostic.py`

R5 的 P0/PLR 配对已经通过真实首次 512-sample gate、两组 64 updates、两个 Z 评估节点、独立审核；再次跑它们只是重复消耗。P0/PLR 不再被描述成“没有执行”或“还需授权才开始”。

### 1.1 主要真实结果

| 指标 | P0 u64 | PLR u64 | 原 Z 门槛 |
|---|---:|---:|---:|
| 原 Z 通过数 | 0/32 | 0/32 | 原协议 |
| 最大 lane 速度 RMS | 0.137930 m/s | 0.056532 m/s | ≤0.08 m/s |
| 最大 lane 水平漂移 | 2.629815 m | 1.088824 m | ≤0.30 m |
| 最大 lane 高度偏差 | 0.696651 m | 0.046659 m | ≤0.20 m |

首次更新 8 minibatches 后，对同一固定真实输入的 analytic Beta KL：P0=15.012778、PLR=0.276192。此变化反映本配对降低 LR 显著减少了策略分布迁移，**但未使 Z 合格**，也不能从一个配对 seed 推广。

PLR u64 训练期（本 update 的真实 512 lane-samples）平均 body-frame `issued_v_corr_b≈[-0.01349, -0.05559, +0.00506] m/s`；实际 `velocity_b≈[-0.01557,-0.05740,+0.00105] m/s`，`ICS beta=1`。这支持“在零命令情形出现不必要的横向修正，且实际运动随之偏移”的近端机制假设；训练片段与确定性 Z 评估并非同一轨迹，不应称为逐步完整因果证明。

### 1.2 代码中值得检验的具体机制

`governor.decode_governor`：`v_corr_b=(2*a[1:4]-1)*0.5 m/s`，`v_gov=alpha*v_cmd+v_corr`。

`rewards_h1.compute_reward_h1`：零命令时 `h=1`、`g=0`，`reward_velocity` 惩罚实际速度，`reward_anchor` 惩罚 ray-based anchor loss，`reward_smooth` 惩罚相邻下发速度变化；而 `reward_zero_extra` 恒为 0，`reward_governor_proxy`、`reward_final_proxy`、`reward_preservation` 被 `g` 门控，在零命令下关闭。**目前缺少对零命令下非必要 `v_corr_b` 的独立直接代价。**

这是一个可检验的 objective-design 缺口，不是已经证明的最终根因。不能说当前完全没有悬停奖励或只需盲目加权。

## 2. 自动推进的四个阶段

### Phase A — 最小离线归因（必须做，但不可无限扩大）

使用现有 pair results、实际 u1/u2 首更新 probe、已保存真实 512 输入、梯度和 checkpoint，以及可用的 u16/u64 原始训练记录，不启动模拟器：

1. 按 **P0/PLR × u1/u2/u16/u64** 对齐实际采样动作、actor mean、`v_corr_b`、`v_final_b`、真实速度、重置、条件 Beta KL。保留 body/world 坐标和记录时序，不将采样结果冒充确定性策略。
2. 专门解释 PLR u64 的水平偏置：按 xyz 有符号均值、RMSE/分位数、命令零掩码、`ICS beta`、reset 前连续片段统计。检查方向、惯性响应、anchor 有效率和奖励规模；**不把单一均值当做闭环因果证明**。
3. 利用现有日志和 CPU 输入，分析首更新梯度/优势/return 与三个 residual heads 的迁移，以及是否存在 reward 与评估在“累计漂移”上的错位。不要用没有保存的历史变量推导精确反事实。
4. 对旧证据完整性与物理数据已通过的内容直接复用；不要重新跑 E0、质量修复、403200 物理记录或 P0/PLR 原生仿真。

产物：`R5_ZERO_COMMAND_ATTRIBUTION.md/.json`（证据、假设、排除项、方法可行性、不可证明部分）。到此不需等用户批准继续 Phase B。

**Phase A 的预先决策规则：**
- 若真实数据支持零命令 `v_corr` 持续偏置，且直接命令 B0 已能控制 Z/T/S，默认进入 Phase B 的“零命令残差正则化”实验。
- 若直接证实实现错误（如概率、错位观测、reward 公式 bug），先做最小修复并补 CPU 回归；在重新建立对照前不把修复与方法提升混为一谈。
- 若关键资料缺失、不能辨别方法是否适用，只做一个明确的低成本 CPU 探针；不擅自运行 GPU 补证。报告阻塞原因，禁止创造合成真实数据。

### Phase B — 单变量方法设计与实现

**默认候选：零命令时的速度修正中性偏好（Zero-command Residual Neutrality）。**

保留原 actor 观测/网络、PPO、ICS、Lee、rotor、课程、评估和 termination，不新增世界位置或真值速度到 actor。只修改零命令奖励：

```text
r_neutral = - w * I[||command_b|| <= 1e-3]
                 * (||v_corr_b||_2 / 0.1 m/s)^2
```

实验组 `w=0.05` 是**事先给定的候选量级**（在 0.1 m/s 修正时惩罚 0.05/step），不是最优值；对照组 `w=0`。在 Phase A 使用实际旧奖励分布核对其尺度；若发现量纲/尺度显著不适用，可在任何训练/授权前确定新的唯一一个固定值，并在报告中说明选择依据，不能根据训练结果事后多次挑最优。

**实现约束：**
- 使用已有 `v_corr_b`，需补接口则进行最小前向数据传递；不要从 `actual_velocity` 反推 actor action，也不要把控制修正强制设置为零。
- 两个条件来自同一个新 commit，唯一有效方法差异是 `w`。对照 `w=0` 必须逐样本复现此前 reward-h1 的总奖励与控制行为（只允许合理浮点容差）。新增奖励项/版本/方法哈希与身份明确记录，评估阈值不变。
- 要求新代码继续保留旧 R4/R5 证据的原 SHA，所有新结果进入新的有名目录，不覆盖旧数据。
- 奖励是软偏好，不是 hard override：遇障碍、外部扰动或需要修正时策略仍允许给非零 `v_corr`。此次先测试零命令，后续如有价值，再判断与安全干预的冲突。
- 该奖励调整属于**基础学习机制的对照实验**，不是已经成立的论文创新。

当前仓库的 `training_workloads/unified_h1/pilot_contract.py` 和 `backend.cold_start()` 目前只识别 P0/PLR 两组且分别绑定 actor LR。新增 C0/C1 时必须**显式扩展新的组身份及配置绑定**，让两组均实际使用 actor LR=1e-4，不覆盖原 P0/PLR 映射或历史配置；第一份新 checkpoint 的 method/source identity、实际 reward coefficient 与授权必须一一对应。工作量属于 Phase B 中必要的最小入口适配，不借机重写通用 runner。

先做 CPU tests：开关 `w=0` 等价性、零命令正/负残差、非零命令完全不触发、mask 边界、数值有限性、动态依赖和同种子初始化。须保留实际最初 gate，且对照/实验组的首次 rollout 输入动作在首次更新前必须可比较。

若 Phase A 明确反驳上述机制，不为了完成形式而强行改 reward；交付 `METHOD_CHOICE_BLOCKED_BY_EVIDENCE`，写出唯一必须解决的反证。不要在同一 Goal 偷换成改变初始化、改网络和改课程的组合方法。

### Phase C — 短程配对训练与原 Z 评估（一个新的整体 GPU 预算）

只有在 Phase A/B 检查全部通过、新代码/配置/资源与真实授权绑定后执行。每组完整从共同 init 冷启动，保持 **actor LR=1e-4、critic LR=3e-4**，两组唯一主变量为 `w=0` 与 `w=0.05`（或 Phase B 预先冻结的一个值）：

| 范围 | 控制 C0 | 候选 C1 |
|---|---:|---:|
| 训练 updates | 64 | 64 |
| 训练 transitions | 32,768 | 32,768 |
| env/rollout/minibatch/epoch | 32/16/64/1 | 相同 |
| checkpoint | init0000/u1/u2/u8/u16/u32/u64 | 相同 |
| 评估 | 原 Z，u16/u64，32 lanes × 200 steps | 相同 |
| 真实 first-rollout gate | 512 samples，任何 optimizer 之前 | 相同 |
| restart/retry | 0 | 0 |

本 Goal 训练+评估**建议的新总硬上限**（须在用户/Commander 的单次总体授权中明确绑定）：**4 个 live start / 4 次 simulator init / 65,536 training transitions / 25,600 nontraining transitions / 128 optimizer updates / 14 checkpoint writes / 0 自动重试**。wall/GPU/磁盘等硬上限由 Commander 在同一次 envelope 中依项目现有已验收资源与磁盘情况确定，明确数值后方可 release，不能沿用上次已经耗尽的 budget。可把上次短程 Pilot 的 10,840 秒 wall/GPU 合计上限作为拟议参考，**不是现有授权**。

用户一次性授权这个**有界研究范围**后，Codex/Commander 在同一任务中按既有机制建立 proposal、做 exact SHA/identity 检查、签发实际单项授权、预检并顺序提交四项；常规 CPU 修复、验证、审查和下一子阶段不逐项征询用户。**但不得伪造用户批准、绕过 STOP/coexistence、假装有未取得的机器权限或修改原始结果**。如平台强制需要用户实际点选授权，整理为一次集合式请求；不反复请求日常“继续”确认。

执行中首次真实 rollout 概率 gate 失败、NaN、输入身份错误、未清理的 owner 进程、非有限值或超预算，应按已批准 fail-closed 逻辑停止，不自动重试。原生 `app.close FAILED/-9` 保留；若实际数据独立证明完整、owner 清理完成，可接受科学证据，仍不能更改原进程/collector 为 PASS。

### Phase D — 自主科学复核与结论

独立审核数据完整性和方法唯一差异，预注册主比较在 u64；u16 用于过程诊断；并与现有 PLR 作历史参考（不当成同时代严格对照）。至少输出原 Z 通过数、全部 lane 的速度 RMS、XY 累计漂移、高度最大偏差、reset、实际零命令 v_corr/速度、有符号偏置及相关 reward/gradient/KL，原门槛不变。

**三个终态，不以策略必需通过来判定 Goal 工程成功：**

- `VALID_EXPERIMENT_POSITIVE`: 候选在 Z 行为上有可信改善，明确还欠哪些泛化/后续实验；只有满足原 Z 才能考虑扩大训练，且新的长训需要单独预算。
- `VALID_EXPERIMENT_NEGATIVE`: 两组训练与原 Z 评估完整，未证明方法有效；这是完成的科学实验。不能再悄悄调两个权重继续训练。
- `BLOCKED_BY_SPECIFIC_DEFECT`: 数据不足或可定位的实现/授权问题；返回**唯一必要补充动作**，不重开全面调查。

即使 u64 达标，也只是原 Z 悬停能力达到某个节点；**完整 R5 的 Z/T/S/X/W、命令跟踪/避障能力尚需之后的完整训练和评估**。

## 3. 交付与代码管理

遵循 `coordination/protocols/SINGLE_BRANCH_WORKFLOW.md`：固定的日常目录 `.../instinctRL-dev`、单写者 `main`；提交前/后保存版本与输入哈希；隔离只读执行快照；必要的变更提交与正常 push 可自动完成（不 force push）。每阶段使用目录下的 append-only 证据，不覆盖历史任务；不允许新建一套平行 orchestration 系统。

推荐在 `docs/experiments/r5-zero-correction-20261014/` 下保存（任务启动时可按实际日期命名，但在运行前固定）：

```text
R5_ZERO_COMMAND_ATTRIBUTION.md
R5_ZERO_COMMAND_ATTRIBUTION.json
METHOD_DIFF_AND_CPU_TESTS.md
AUTHORITY_AND_RESOURCE_BINDINGS.md
PAIR_ANALYSIS.md
pair_analysis.json
EVIDENCE_INDEX.md
NEXT_RESEARCH_DECISION.md
```

Goal 最终汇总应直接回答：
1. 降低 LR 的实测进展是什么、失败集中在什么指标？
2. 零命令下实际策略修正与真实水平漂移如何关联？
3. 加入零命令中性偏好后，是否比同时代 `w=0` 对照更好？
4. 这是否值得扩大 256 updates？若不值得，下一次只应改变哪个**新**因素？
5. 哪些作业/预算/原生关闭失败被真实保留？

**严禁把四项历史 FAILED/-9 的 native-close status 改成成功；所有策略、评估与重置的原始失败都必须保留。**
