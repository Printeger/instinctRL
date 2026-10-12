# InstinctRL E0 → D7-R/B0 → P0/PLR 代码审阅与执行建议

日期：2026-09-23。性质：咨询性代码审阅，不代替 Commander 正式验收或机器运行授权。

## 1. 结论

**方案研究方向合适，建议有条件实施；当前不能原样一键执行到训练结束。**

推荐流程：接受 E0 的有限结论 → 关闭运行接线问题 → 实际完成 D7-R/B0 → 分析控制链 → 仅在下层限制已排除或解释、训练实现无已知缺陷时，执行带首次更新诊断的 P0/PLR → 给出是否值得扩大训练的结论。

“定位上层或下层”不是保证一定得到单一根因；允许 MIXED_CAUSES 或 INSUFFICIENT_SIGNAL。若下层有主导问题，本轮正确结果是停止进入学习率对照并提出具体修复，不是为了完成流程硬跑 P0/PLR。

新 plan 第 11 节原 prompt 仅要求准备训练、明确不自动启动。因此，要实现用户希望的完整短程闭环，必须使用本报告配套的新 prompt，并完成独立训练预算绑定。扩大训练仅输出建议，不在本轮自动执行。

## 2. 本次核验范围与实际结果

- 阅读新旧 plan、E0 JSON/分析脚本、D7 提案、CPU 验证记录及相关 product/workload/rlflow 源码。
- 四份 E0/D7 文件 SHA256 与新 plan 第 1.2 节相符。
- D7 工作树干净，HEAD 为 `c7a4d79e35916ce2711085674b9cd37f1054a1ab`，tree 为 `1e06553c95d9f34725179919165b28086d81778d`，与提案一致。
- 独立复跑两份既有 CPU 测试：**8 passed in 0.60s**；禁用字节码与 pytest 缓存写入。未启动仿真、GPU 推理或训练。
- 已查看的 rlflow 导出状态为 `enabled=false`；未在该状态的 jobs/proposals/packages 中找到 D7-R/B0 对应记录。执行前仍须通过当前控制面核对数据库及进程，不能以此快照替代提交前查重。
- 未全量重跑 E0、未重新验证全部历史日志或加载真实 checkpoint 推理，故不声称独立复算了 E0 每个数值。

证据目录：
`instinctRL_orchestrator/implementation_records/instinctrl-first-use-20260915/state/analyses/UNIFIED-H1-R4-E0002/J-880cd798ee89d6eb/`

## 3. E0 可以接受到什么程度

| 结论 | 判断 | 边界 |
|---|---|---|
| minibatch 编号异常是旧诊断假设错误 | 可接受为阶段结论 | 源码确实采用每 update 局部编号；日志覆盖结果支持关闭此疑点，不等于整个 PPO 已验证 |
| 采样动作至 governor/ICS 的映射正确 | 可接受已有记录范围的复算 | 不证明动作合理，也不证明实际飞行跟踪正确 |
| 首次 update 后续 minibatch 出现大幅概率变化 | 值得优先验证 | 旧日志缺完整概率与真实输入，不能直接确认 LR 是根因 |
| GAE/timeout fixture 通过 | 接受被测试路径 | 不替代实际环境与全部终止路径验证 |
| D7 CPU 通过 | 本次复跑确认 | 不意味着 B0 飞行成功或下层控制正常 |

源码对应：

- [PPO 更新与局部编号](worktrees/instinctRL-unified-h1-r3-d0011-product/isaaclab-training/instinctlab/training/ppo.py#L286)：独立 actor/critic 优化器，每个 minibatch 分别 step；旧日志的 approximate KL 来自当前 minibatch，不是固定探针集合。
- [E0 分析脚本](instinctRL_orchestrator/implementation_records/instinctrl-first-use-20260915/state/analyses/UNIFIED-H1-R4-E0002/J-880cd798ee89d6eb/r4_review_e0.py#L50)：读取局部编号、coverage 与累计 optimizer 计数；历史样本索引关联包含按固定种子重建的部分，应与直接保存的历史数据区分。

## 4. 运行前和训练前的具体缺口

### 4.1 MUST_FIX：D7 模式的种子接线不一致

当前发布状态所指向的控制版本 `ad587e02521acb102703713e42933975a62cbc455b1df47432138a346d31382f` 中，[h1.py](instinctRL_orchestrator/implementation_records/instinctrl-first-use-20260915/state/releases/ad587e02521acb102703713e42933975a62cbc455b1df47432138a346d31382f/rlflow/h1.py#L81) 仅对 `mode == evaluation` 选择 evaluation_seed，其余选择 training seed。

但 [D7 backend](worktrees/instinctrl-r4-d7-control-diagnostic-20260923/training_workloads/unified_h1/backend.py#L363) 的 `control-diagnostic` 显式 reset seed 为 `20260909`；训练 seed 是 `20260908`。

影响：按实际评估种子注册可能被拒绝；按旧适配器填训练种子又会造成元数据与执行不符。

关闭条件：在隔离控制候选中为诊断模式显式绑定正确种子，补正确/错误种子测试，按现有发布审核流程启用；不能改成错误 seed 只为通过检查，不能改写冻结 release。

### 4.2 MUST_FIX：首次更新 gate 与固定探针尚未接入

现有 PPO 在 `update()` 中直接计算各 minibatch loss、反传、裁剪并 step。已有 old-logp 不变和 shape 检查，但没有新 plan 所需的全部 512 样本首次更新前概率/分布自洽 gate，也没有逐 minibatch 前后固定探针解析 KL、裁剪后梯度和参数增量证据。

现有 [训练循环](worktrees/instinctrl-r4-d7-control-diagnostic-20260923/training_workloads/unified_h1/driver.py#L119) 可用于定位 rollout 完成到 trainer.update 之间的接入点，但原 before_optimizer 回调仅接收 update 编号，不能把它直接当成已具备完整批次检查。

关闭条件：隔离训练分支实现并测试；首次 gate 失败时 actor 与 critic 均不得发生任何 optimizer step；两组同样启用。完整保存前两个 update 的真实输入/动作/旧分布，首个 update 的 8 个 minibatch 使用固定 512 样本探针比较。

补充边界：[rollout_h1.py](worktrees/instinctRL-unified-h1-r3-d0011-product/isaaclab-training/instinctlab/training/rollout_h1.py#L27) 在 step 后构造 transition 快照，但当前 [环境返回](worktrees/instinctRL-unified-h1-r3-d0011-product/isaaclab-training/instinctlab/envs/taslab_direct_env.py#L403) 已 clone policy，后续 rollout 也 clone next policy；不能仅因 clone 位置就宣布已发现历史 buffer 污染。新诊断仍应明确冻结采样时快照并测试原地修改情形。

### 4.3 MUST_FIX：短训配置、保存节点和 Z-only 评估需真实开发

现有 backend 初始化使用 PPOConfig 默认 actor LR，不能只在报告里写 PLR=1e-4；必须把解析后配置真正传入 optimizer，并记录实际 param_groups 学习率。

[train_to_update](worktrees/instinctrl-r4-d7-control-diagnostic-20260923/training_workloads/unified_h1/backend.py#L296) 的保存间隔固定 64。新计划要求每组 init0000/u1/u2/u8/u16/u32/u64，共 7 个节点，不能默认旧入口已经支持。现有正式检查点评估走完整案例，也不能拿它代替预算仅覆盖的 Z-only 开发评估。

关闭条件：新增明确的 pilot 入口/配置契约，支持 64-update 冷启动、7 个节点和 u16/u64 的 Z-only 开发评估；新身份与授权对应。保留原冻结 R4 行为，不把短训伪装成旧任务 resume。

### 4.4 CONTRACT_DECISION：资源口径与硬上限需落地

D7 提案给出 action GPU 1800 秒、含清理墙钟 1820 秒；rlflow H1 adapter 使用每条件 `wall_seconds + 10` 同时作为外层 wall 与 allocated-GPU 资源，因此外层两条件 GPU 预留也会是 1820 秒。这可以解释为不同计量边界，但必须在注册包中显式写明，不能把它们称为同一个 1800 秒硬上限。

提案的磁盘上限是两条件合计 16 GiB；当前 DiagnosticChunkWriter 记录每块大小，却没有累计磁盘预算参数或写前上限判断。本次未找到该 writer 的聚合硬上限执行证据。运行前应证明外层已有可用限制，或补最小写入保护及跨条件累计计数；仅估算文件小于 16 GiB 不等于执行了硬限制。

P0/PLR 还缺单独的启动、初始化、GPU、墙钟、磁盘、模型调用预算。先生成完整注册提案并集中批准；不得继承 D7 的零更新/零 checkpoint 约束，也不得继承旧 R4 训练额度。

### 4.5 分析器需处理真实信号边界

D7 已正确区分 mean action 与 B0 固定 action，并走相同 H1 控制链。[实现](worktrees/instinctrl-r4-d7-control-diagnostic-20260923/training_workloads/unified_h1/backend.py#L389) 对 reset lane 的 Lee/wrench cache 标记无效，这是合理披露。

分析器仍需验证 manifest/hash、630 步连续性、32 lanes、B0 恒定动作、有限值、时间关联和有效 mask；无效 cache 不能填零。原 Z/T/S 指标只作 DIAGNOSTIC_REFERENCE_ONLY，不写回正式验收。现有 CPU 测试没有覆盖真实仿真信号与完整分析器，不应夸大为端到端已验证。

## 5. 建议的分段闭环

| 阶段 | 必须完成 | 下一步条件 |
|---|---|---|
| E0 接受 | 独立记录接受范围与 NOT PROVEN 项 | 不再重复全量调研 |
| D7 运行准备 | 种子/预算接线、分析器、CPU 测试、真实注册与批准 | 身份一致、无重复作业、STOP 等门槛通过 |
| D7-R/B0 实跑 | 两条件各一次，全部原始证据与实际计数 | 结果可比且信号足以分析 |
| 控制链归因 | 上层、下层、混合或证据不足，给出理由 | 只有适合学习变量比较时进入短训 |
| P0/PLR | 首次 gate、64 updates/组、指定 checkpoint、Z 开发评估 | 独立授权且共同基线成立 |
| 研究结论 | 明确 LR 是否改善控制与行为、是否值得扩展 | 只建议后续预算，不自动扩训 |

## 6. 预算边界

| 阶段 | 训练 transitions | 非训练 transitions | 更新与写盘 |
|---|---:|---:|---|
| D7-R/B0 | 0 | 40,320 | 0 update，0 训练 checkpoint；2 次启动/初始化 |
| P0/PLR | 65,536 | 0 | 128 个 PPO update 合计；actor/critic 各 1,024 optimizer steps；14 个训练 checkpoint 节点合计 |
| 两组 u16/u64 的 Z 评估 | 0 | 25,600 | 0 optimizer 更新 |
| 合计 | 65,536 | 65,920 | 131,456 次环境交互 |

不含 Bz、额外 u0 评估、自动重试、完整五类复验和扩大训练。失败尝试也占预算。首次 gate 失败前已经采集的 512 次交互仍计入已消费训练采样，不能因 optimizer=0 就抹掉。

## 7. “有效、可以扩大训练”怎样判定

运行前注册同一组判据，主比较节点固定 u64、u16 仅过程观察，所有 lanes/节点完整报告：

1. 实验有效性：输入/概率一致，唯一方法差异为 actor LR，资源与证据完整。
2. 机制：固定探针 KL、参数变化和 xyz 修正偏置是否改善。
3. 行为：Z 全窗口成功率、高度偏差、水平漂移、速度、存活时域与安全失效是否改善，保留原指标口径。
4. 跨指标退化与不确定性：不能只依据 loss 或 KL 变小宣告有效；单训练 seed 的配对结果不是跨种子稳健性证明。

建议结论标签：

- `PROMISING_FOR_BOUNDED_EXPANSION`：行为与机制有一致改善，安全没有预注册意义上的退化，值得做同预算扩展/独立种子和完整任务复验；不代表正式合格。
- `NO_BEHAVIORAL_BENEFIT`：KL 或优化统计改善，但行为未改善，不继续盲目降 LR。
- `INCONCLUSIVE_AT_64_UPDATES`：预算内不能区分或候选学习更慢；若续做需两组对称预算。
- `BLOCKED_BY_CONTROL_BASELINE`：下层/初始化限制尚主导，暂不进入学习率比较。
- `INVALID_EXPERIMENT`：概率/样本/配置/证据失效，不能据此评价 LR。

这些是报告标签，不是改写正式 gate。具体改善幅度与“无明显退化”的数值规则须在 pilot 运行前确定，不在看结果后选择。

## 8. 下一步使用方式

将配套 [执行 prompt](InstinctRL_E0_D7_R5_RLFLOW_PROMPT_20260923.md) 整段发送给 rlflow 执行窗口。它要求实际完成授权范围内的闭环，而不是只重复 CPU_READY；发现下层问题时允许有证据地停止短训分支。

Codex 窗口负责发起目标、接线开发和解释结果；实际仿真/训练必须经 rlflow 注册作业、预算检查、证据收集和结算。实现与审核按现行角色分离，同一代码区域保持单写者。

本报告本身不批准旧结果、不创建运行许可、不启动任何作业。下一轮执行前应重新核对实时状态，避免重复提交。
