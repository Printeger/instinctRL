# InstinctRL 下一步 Goal：诊断闭环、最小修复与短程训练准入

> 版本：1.0｜编制日期：2026-10-12  
> 代码审查快照：`Printeger/instinctRL`，`main@ebc27e2e47d4190efccac9a9179d4e291bd933da`。  
> 推荐存放位置：`docs/goals/R5_DIAGNOSTIC_CLOSURE_REQUIREMENTS.md`。  
> 本文是新的开发需求，不是已经完成的诊断报告，也不是原有实验授权的续期。用户实际发送配套启动 prompt 后，才按其中明确的范围建立新任务授权。本文的建议参数、预算和准入状态不冒充历史事实。

## 0. 一句话 Goal 与终点

**在当前统一仓库上，完成尚缺的真实控制诊断，修复有证据支持的实现缺陷并进行回归；复用、修正和验证已有首次更新诊断，使下一轮短程配对训练具备明确的启动条件。**

推荐成功终态：

```text
READY_FOR_PAIRED_PILOT_WITH_FIRST_ROLLOUT_GATE
```

含义是：控制基线已有真实验证；必要修复、训练代码和诊断工具已审核；下一轮配置、输入、预算与入口可准确生成；下一轮第一次参数更新前仍须通过真实 rollout 的概率一致性门槛。

**本 Goal 不运行 P0/PLR，不进行任何训练参数更新，不扩大训练，不要求证明所有研究假设。** 它结束在“可以负责任地启动一次有解释力的训练试验”，不是“无人机已经学会全部任务”。

```text
核对现状并复用证据
  → 打通诊断、审核与清理链
  → 补齐必要的真实控制诊断
  → 对已证实缺陷做最小修复和真实回归
  → 校验现有首次更新诊断及训练入口
  → 交付短程训练准入包，停止在训练启动之前
```

不要把 Goal 写成“修完全部 RL 问题直到收敛”；也不要缩成“再写一份调查报告”或“让 CPU 测试通过”。

---

## 1. 当前事实：从这里继续，不从旧计划的起点重做

### 1.1 本次审查边界

本文依据当前 GitHub 的源码、状态和已归档报告编制。已直接检查的关键源码和文档见第 13 节。编制时没有在用户主机上重跑测试、载入历史大体积原始证据、进行仿真或训练。

下文区分：

- **当前源码事实**：在上述固定 commit 直接读取。
- **历史报告事实**：仓库中记录的历史运行、统计和审查结果；执行者须核对对应输入和原始证据是否仍可用。
- **本任务要求**：现在提出的工作范围、测试、决策与预算。

如果本地 HEAD 比审查快照更新，只检查相关增量，不回退 main；把真正执行版本记入新任务。不用报告日期替代 commit 或真实运行身份。

### 1.2 已有进度与真正缺口

| 对象 | 当前能够确认的状态 | 本 Goal 如何处理 |
|---|---|---|
| E0 | 历史补证已经解释 minibatch 编号误解，验证记录范围内的动作映射和关键 GAE fixture | 接受已有结论；只有涉及新修改的路径才回归，不重做全面 E0 |
| 工作区整理 | 已统一为 `/home/mint/rl_dev/instinctRL-dev` 的 `main` | 沿用单仓单分支，不恢复旧任务分支体系 |
| 历史零命令 B0 | 9 月 27 日报告复用了 20,160 次真实历史交互，发现垂直质量不一致机制，水平漂移未解释 | 找到并复用对应原始证据；不能说所有 B0 都从未运行 |
| 最近 enhanced B0 | 9 月 28 日在首个 reset 失败，0 transitions、0 physics rows | 这是诊断执行失败，不是“零漂移”或“控制正常” |
| Stage 1 修复 | 初始 reset 观测、计账、证据边界和后代进程清理修复已迁入 | 验证现有实现，不默认还没写，也不沿用旧 receipt 判定新版本通过 |
| 独立审核 | 当前状态仍未验收；最近输出因 MUST_FIX 必要字段缺失而被协议拒绝，旧任务预算已耗尽 | 新任务解决审核可读性与输出协议；保留旧失败，不伪造旧任务完成 |
| 工具环境 | 状态记录 Codex CLI 为 0.161.0，旧隔离契约固定 0.153.4 | 先核对实际安装和必要兼容性，不能只改版本字符串变绿 |
| 首次更新诊断 | `pilot_diagnostic.py` 和 P0/PLR 配置、测试已经存在 | 复用并修正具体缺陷，见第 5 节 |
| R5 控制链报告 | `r5_delivery/R5_CONTROL_CHAIN_DIAGNOSIS.md` 当前是 NOT_STARTED 模板 | 模板存在不代表有结果；以真实输入、结果和 receipt 判断 |
| 当前运行权限 | 状态中的新 live、训练授权均为 false | 不借整理授权启动实验；按新 prompt 和现有授权机制建立本任务边界 |

来源：[S01–S08、S11–S13]。

### 1.3 当前最有价值的物理线索

历史零命令 B0 报告给出：[S06]

```text
Lee 使用的 base mass：          1.1583 kg
完整 articulated vehicle mass：1.1815179586 kg
预测稳态垂直速度：             -0.0893818425 m/s
三个历史窗口的观测均值：       -0.0893858326 m/s
水平速度约：                   +0.1934 / -0.0559 m/s
```

当前 `config.py` 仍定义 `TASLAB_UAV.mass=1.1583`；Lee 的 collective 直接使用该 mass。代码中没有积分或恢复参考位置的反馈项；返回的 held position 不是位置保持控制。[S09、S10]

**研究判断：下层基线问题优先于 PPO 学习率试验。** 当前不能直接沿用“B0 正常，所以只降 LR”的条件分支。历史垂直机制很有依据，但不解释水平漂移；更不能用它声称整轮 R4 的全部失败已经解释。

历史完整质量必须与当前实际资产和施加到 plant 的配置重新绑定，不能直接抄一个小数作为新真值。

---

## 2. 范围：一项有终点的工程／研究任务

### 2.1 必须完成

1. 建立简短的现状差异表和证据索引，识别已完成、模板、失败、未执行及本地未上传结果。
2. 验证并关闭阻止诊断的工程问题：初始 reset、观测接线、计账、退出清理、独立审核的证据可读性和必要 CLI 兼容性。
3. 在新授权预算内执行必要的真实诊断，形成对下一步修复足够具体的结论。
4. 在新候选上修复证据明确的控制／数据／诊断实现问题，并做相应真实回归。
5. 修正并验证已有首次更新诊断，尤其是三轴修正字段、真实概率一致性和异常退出后的证据保存。
6. 交付有明确成功条件、共同基线和输入绑定的下一轮配对训练准入包；本次不训练。

### 2.2 明确不做

不开展完整 R5 训练、多种子性能研究、超参数搜索、新网络／SAC／Transformer、奖励与课程的组合改造、部署试飞、新 dashboard 或通用 rlflow 平台重构。

不向 actor 增加位置、实际或显式估计速度、SLAM、地图、障碍真值。不得通过放宽门槛、删掉启动段、隐藏失败样本、偷偷预热或重置来获得通过。

不安装或升级全局依赖，不改冻结 runtime，不恢复旧 SCALE 训练，不用旧报告或旧授权代替本任务的实验许可。

**允许修复已证实的实现错误，不把未知的研究设计选择伪装成 bug。** 例如，纠正重力补偿读取错误质量与引入全新位置外环是两种不同变更；后者需要明确任务／方法决策。

### 2.3 Goal 内职责

- **Supervisor／Commander**：维护一个总目标，依据证据选择下一动作，管理范围、预算和批准；不把“编写者自称通过”当独立验收。
- **Builder**：唯一代码写入者，完成实现、测试、分析和交付。
- **独立 Reviewer／Evaluator**：在独立上下文审查准确候选和实际证据；可并行阅读，不与 Builder 在同一目录同时写代码。

优先复用现有可用角色与控制工具。Goal 提供持续目标，不替代运行权限、过程记账或独立审查；也不应为此增加一个新的常驻 agent 服务。[S01、S03、S17]

---

## 3. 阶段 A：只做必要的现状核对与工具恢复

### A1. 确认实际执行上下文

先读：

```text
AGENTS.md
docs/PROJECT_STATUS.md
coordination/ACTIVE_WORKFLOW.json
coordination/PROJECT_STATE.json
coordination/CURRENT_TASK.md
coordination/protocols/SINGLE_BRANCH_WORKFLOW.md
```

记录当前目录、HEAD、dirty 文件、远端继承关系，以及 Python、Codex CLI 和实际使用的项目工具版本。不要覆写其他人的修改，不自动 reset/clean/stash。

依据现有 provenance 和报告路径寻找后续证据。新结果已存在且身份可复核时直接使用，不重复启动。搜索以当前任务相关路径为范围；不要再次遍历全部历史工作区做一次迁移审计。

### A2. 关闭“审核系统自己不能审”的问题

当前状态记录的两项工具安装检查失败必须分类：是单纯固定版本不兼容，还是输出、隔离、证据访问等真实契约变化。[S02、S04]

要求：

- 对实际安装的 CLI 做必要兼容验证，并保存调用与输出；需要时建立新的、经测试的兼容配置，不篡改历史 0.153.4 契约。
- 仅修项目内与本任务有关的兼容层、路径绑定和解析器。不借此重构整个 orchestrator。
- 独立审核必须实际读得到候选源码、测试结果、最新 job 的原始 evidence 与分析脚本。不得只给文件名、外部路径字符串或摘要。
- 在昂贵审核前本地校验输出 schema。`MUST_FIX` 至少具备 `contract`、`evidence`、`impact`、`closure`。
- 格式不合法与科学判断不通过分开记录；不能把格式修正冒充技术问题已经解决。
- 不要求重复完成与当前修改无关的历史认证。新工具不能冒充旧认证 release；新授权和运行入口须准确绑定工具版本。

如果缺少账户、安装权限或可工作的独立审核入口，这是外部阻塞。继续完成不依赖它的 CPU 工作，最后只提交一次准确的缺口和恢复动作，不循环伪造审核。

### A3. 工作区与版本管理

日常修改直接在当前 `main` 形成 additive commits。每个实际运行之前冻结全部执行输入，记录准确 commit、配置、工具和数据哈希；用只读快照或 detached HEAD 执行。

快照是运行输入，不是新的长期开发分支。旧历史目录、旧 receipt 和冻结 checkpoint 保持只读。

当前 `SELF` 配置标记须按已有机制解析成准确新身份，不能伪装成历史 R4 source identity。更改路径或工具后重新验收相应候选，不能继承旧 receipt。[S01、S03、S14]

**A 阶段完成标准：** 已确定唯一候选、真实证据所在位置、必要工具能执行和审查；没有把旧任务的 BUDGET_EXHAUSTED 或迁移状态改写为成功。

---

## 4. 阶段 B：先让诊断成为可靠的测量工具

已有核心模块应复用：[S07、S13–S15]

```text
isaaclab-training/instinctlab/envs/unified_h1_direct_env.py
training_workloads/unified_h1/b0_observer.py
training_workloads/unified_h1/b0_enhanced.py
training_workloads/unified_h1/b0_causal_analyzer.py
training_workloads/unified_h1/b0_handoff.py
training_workloads/unified_h1/control_diagnostic.py
training_workloads/unified_h1/backend.py
training_workloads/unified_h1/budget.py
```

### B1. 初始 reset 必须按真实生命周期处理

9 月 28 日的失败发生在第一条 policy step 之前。检查已经迁入的修复是否真正解决该边界，而不是照旧再写一遍。[S05]

必须区分：

- 初始 reset 前，从未产生的上一步 Lee 输出：可以按 schema 标识“不存在／尚未计算”，不能要求它必须是有效旧缓存。
- reset 后的真实 controller／rotor 初始状态：应从实际状态读取并完整记录，不能用任意零张量充当测量。
- 正常运行中的 reset：保存最后一个有效 pre-reset 和约定 post-reset 序列，不能把被清空的缓存当有效下游值。

初始边界、普通显式 reset、自动 reset 均要测试。真实 reset 计账必须在实际发生边界结算，即使其后的 observer 写盘报错，也不能把已经发生的 reset 记为零。

### B2. 观测必须接在真实执行路径

复用 `enhanced_b0_read_only_protocol`／`FormalEnhancedB0EnvironmentAdapter`。证明 post-physics、pre-reset 的顺序正确、每子步每 lane 恰好一次，且没有改变控制、动力学、RNG、actor history 或 reset。

测试分层报告：纯函数、mock 接线、真实产品模块集成、真实仿真。AST 或字符串出现测试只能证明接线声明存在，不能替代实际调用验证。

不得为了生成历史状态额外执行一次 physics step、修改 rotor throttle 或悄悄进行热启动。

### B3. 清理与失败证据

先验证已经迁入的进程后代清理：正常结束、环境异常、app.close 超时、脱离进程组并 reparent 的子进程，都应在任务所有权边界内清理。

不能按通用进程名杀全机任务，不能影响既存的 GNOME／远程桌面等服务。使用准确进程身份和所有权证据；清理不确定时禁止启动下一次 live。

失败也应留下真实 attempted/completed counters、主错误、close 结果、部分 manifest、实际行数、退出码与保守资源结算。保留旧失败预算的保守预留，未经正式对账不自动释放。

### B4. 分析器不能用启发式分数冒充因果结论

当前 `b0_causal_analyzer.py` 的 `discriminate_signatures` 使用残差稳定性、某类 wrench 是否活动、是否发生 reset 等启发式分数，再结合领先分数生成 root-cause 标签。[S12]

**本任务要求：**

1. 保留这些分数作为候选线索，但不因 `reset_rows > 0` 就断言 reset 导致漂移，不因固定力分数最高就宣布根因成立。
2. 明确区分数据完整性、数值一致性、机制支持程度、修复验证。四者不是同一个 PASS。
3. `analyze_raw` 中相邻样本目前按 `(case, lane)` 保存；审核其 reset／episode 边界。禁止把重置位置跳变做普通动量差分。
4. 每个 `(condition, case, lane, episode)` 独立计算响应；不能拿全样本第一行与最后一行当同一条轨迹端点，也不能用跨方向均值掩盖相反偏差。
5. 明确 root、sensor 和 articulated COM 的位置、速度及加速度差别；`m*a` 和动量变化必须引用同一个物理对象、坐标系与时间窗。
6. 不可读 wrench 保持 `UNAVAILABLE`，附来源限制；不得补零。是否阻塞某个结论取决于它能否改变该结论，不机械要求所有九类假设同时闭合。
7. 容差依据数值精度、差分方式和数据量级确定，写入新分析配置并用正负样例验证。不得事后调宽来让某次数据通过。
8. “净力接近零”并不证明“速度接近零”；必须解释速度控制为什么没有纠正非零速度。不能把维持匀速本身等同于存在某个非零合力。

分析器修改影响的是诊断口径。保留历史分析原文，新报告对同一 raw 重算时注明新版本和差异，不回填历史 verdict。

### B5. 独立预审完成标准

已迁入的修复均有针对性测试；首次 reset 与清理有可执行验证路径；必需字段缺失、重复行、跨 reset 差分、错误 COM／rotor／时间单位等反例被拒绝；审核者能读取实际证据。

**B 完成仍不等于真实诊断已经完成。** 预审通过后按第 10 节进入获批运行，不以 CPU_READY 结束本 Goal。

---

## 5. 可并行的 CPU 工作：修正已有首次更新诊断

### 5.1 确定的诊断字段错误

当前 `pilot_diagnostic.py::FirstUpdateDiagnostic._evaluate` 中：[S11]

```python
"xyz_correction": self._probe["actions"][:, :3] - mean[:, :3]
```

这不是物理意义上的三轴速度修正。它包含 `a0` 命令保留系数，漏掉 `a3` 垂直头，且计算的是 sample 与 mean 之差。

按已确认 governor 语义，候选正确表达为：

```python
issued_v_corr_b = (2.0 * actions[:, 1:4] - 1.0) * 0.5  # m/s
mean_v_corr_b = (2.0 * mean[:, 1:4] - 1.0) * 0.5       # m/s
sample_minus_mean_action = actions - mean             # dimensionless
```

要求调用／核对当前 governor 的同一参数化，不另造长期漂移的“第二份公式”。旧字段废弃或明确改名，版本化新 schema，并检查所有消费者。

测试至少覆盖：中性修正、仅 a0 变化、仅 a3 变化、三个修正头的正负边界和 batch shape。不能只 `assertIn("xyz_correction", source)`。

**这是诊断输出错误，不代表历史训练时 governor 也按错误公式控制了无人机。不要混淆。**

### 5.2 复用现有 gate，而不是重新实现整套 PPO

现有代码已经捕获 sampling-time Beta 参数、绑定 512 个样本、检查首轮更新前的 log-prob／分布一致性，并记录固定 probe 上八个 minibatch 前后变化。[S11、S13]

需验证：

- collection 时的 old alpha/beta/logp、policy input 和 action 一一对应，不被可变 buffer 覆盖。
- 在第一次 optimizer step 之前，同输入、同 action、未改参数时的重新计算结果符合预先固定容差。
- 失败发生在参数更新之前，actor、critic、两套 optimizer 不前进，RNG 不被诊断额外消费。
- gate、gradient hook、optimizer hook 在正常结束和异常时均正确解绑，部分证据在报错前后可读取。
- probe 的新旧分布比较使用同一组真实输入，记录动作头、量纲、样本索引和参数变化；不把不同 minibatch 样本差异当作策略漂移。
- 诊断关闭时是无副作用空操作；开启时，除检查失败导致停止外，不改变训练数据、损失、梯度裁剪、参数更新及 RNG。
- 采用实际 torch 张量和产品网络／概率函数补数值回归。现有 fake torch 与 mock 接线测试保留，但不替代真实计算测试。

**不在本 Goal 中制造新的训练更新以验证 gate。** 可以进行 CPU 单元测试与可用真实输入的无更新前向；下次训练的首个真实 rollout 必须在第一次 optimizer step 之前执行该 gate。这是合理的运行前条件，不是永久欠缺。

如果 E0 的原始训练概率状态已经丢失，不再要求“恢复所有历史输入”才能进入下一轮。

---

## 6. 阶段 C：完成一次有区分力的真实控制诊断

### 6.1 先统一实验名与动作语义

仓库历史中的 B0 名称并不总代表同一干预，必须在新 manifest 中明确区分：[S06、S08、S15]

| 条件 | 含义 | 能回答的问题 |
|---|---|---|
| `ENHANCED_B0_ZERO_TARGET` | 现有 enhanced adapter 的 Z/T/S 全程目标速度均为零 | 下层零速度响应、质量／COM／力矩／reset 原因 |
| `B0_COMMAND_PASSTHROUGH` | 固定 governor 动作 `[1,.5,.5,.5]`，在 H1 同一链中执行各案例命令 | 无学习修正时的悬停、低速跟踪与停止 |
| `D7_R_FROZEN_POLICY` | u256 原 Beta mean，明确 product 与 workload 身份 | 原学习策略如何修改速度及其实际结果 |
| `D7_R_ON_REPAIRED_PLANT` | 原 u256 权重，但运行在修复后的下层 | 新 plant 上的探针；不是原 R4 严格重放，也不是新训练成果 |

不能从 enhanced 零目标版本的 T/S 文件名推断它完成过非零命令跟踪；不能用旧 direct-B0 旁路替代已约定的 H1 governor／ICS 路径。

### 6.2 运行前必须完成

提交并冻结候选；独立预审认可准确输入；通过现有正式 runner 的 validate-only、身份、预算、coexistence、STOP 与所有权检查；确认 actor／奖励／动力学无未登记变更。

初始证据尚不足时，先采集未作物理修复的 enhanced 零目标 B0。后续物理修复必须用不同候选与不同实验 ID；不能把两个版本拼接成一个“原基线”。

若本地已经有有效且满足当前诊断问题的完整证据，优先离线重分析，跳过重复 live。缺少非关键字段不自动要求重跑全套；缺少决定修复方向的字段才提出最小补齐。

### 6.3 数据与统计

默认沿用现有完整 enhanced 协议：32 lanes、Z/T/S 共 630 policy steps、每 policy step 20 physics steps、共 20,160 nontraining transitions 和 403,200 个 lane-physics rows。[S08、S15]

必需数据包括实际 per-body mass 与 COM transform、各 body pose／速度、rotor 状态和作用点、控制期望、实际施力／力矩及可获得的其他作用、root/COM 响应、时间与 reset 标识。initialization ticks 单独计账，不冒充任务交互。

输出按轴、条件、case、lane、episode 展示，分开启动、稳态、命令切换和 reset 前后。诊断可以另列分段统计，但正式指标不能删去启动误差或失败样本。

对不可直接测量的力或约束，只报告可识别的净残差和限制。用残差定位可能范围，再结合可核实的代码／资产机制或受控回归区分，不硬给每一类力补一个数字。

### 6.4 这次必须产出的研究判断

不能只写“还可能是九种原因”。至少回答：

- 当前资产下，质量不一致的预测是否仍与实际垂直响应吻合？
- 水平漂移在哪个响应或施力／姿态环节出现，哪些解释已被排除？
- 证据支持一个明确实现修复，还是仍需一次特定区分诊断？
- 修复的可证伪预测是什么：改动后哪个中间量和哪个行为应同时变化？

结论可分为“垂直机制已支持／水平未决”，不强求同一个 x/y/z 根因。未知项要说明是否会改变下一次实验选择。

---

## 7. 阶段 D：修复证实的缺陷，不让 RL 学习补偿坏基线

### D1. 质量问题：统一正确的来源，而非全局替换数字

若当前绑定资产复核支持历史机制，修复 Lee 的控制质量来源，保证它与受控 articulated system 的总质量一致。[S06、S09、S10]

**不能直接把所有 `1.1583` 替换成 `1.1815179586`。** 查清每一个使用处：base-link mass、总质量、资产生成配置和控制器补偿是不同概念。若同时把 plant 的 base mass 增大，附属部件仍有质量，可能再次造成总质量错误。

要求：

- 明确 base mass 与 whole-vehicle mass 的角色、单位、资产版本与校验方式。
- 采用一个受检查的权威总质量来源，错误或不可获得时显式失败，不静默回退旧 base mass。
- 不顺带改变真实 plant 质量、重力、惯量、gain 或 rotor 参数，除非另有证据证明它们也错；所有必要变化逐项登记。
- 增加组件求和、控制质量与运行资产一致性、缺失值及重复计数的测试。
- 修复后验证期望推力、稳态垂直速度和高度曲线，不能只验证常量相等。

本任务允许对已证实的垂直缺陷做可独立验证的修复，同时继续处理水平问题；**不能将垂直修复成功写成整个控制基线通过。** 不再要求先证明两者必然同源才允许改确定的缺陷。

### D2. 水平问题：根据数据选择最小修复

只修实际证据支持的 COM／作用点、frame、mixer、joint／constraint、数据取值、初始化或 reset 等实现问题。

不因为理论上“COM 偏了可能会漂”就先移动 COM；不因为 gain 小就先加倍，不通过固定水平补偿抵消未解释偏置。若是根本的控制职责或可观测性缺口，而非实现错误，单独形成 `CONTRACT_DECISION`，说明最小方法选择及影响。

### D3. 固定动作回归与必要的冻结策略探针

在准确新候选上进行同条件 B0 回归；必要时进行原低速命令的 passthrough 测试和 D7-R。

共同约束：不改变原评估定义、初态、seed、分母或安全门槛；不通过让飞行器不动来伪装跟踪成功。完整保存修复前后中间量与结果。

修复下层以后，历史 R4 只能作为旧系统参考，不能作为新 P0/PLR 的同条件对照。新配对训练的两组必须使用**同一个修复后控制基线**。

### D4. 控制基线的准入判断

复用原 Z/T/S 指标与适用的子组规则，逐项报告通过／失败；同时报告连续误差、启动瞬态和稳态行为。零目标 enhanced B0 不能替代非零命令 T/S 准入。

对本次准备的全零 warmup pilot，必须有可靠的零命令控制基线，且不能存在尚未解释的持续下坠或明显水平漂移。T/S 不足能否留到后续训练，要明确它是“策略尚未学习”，还是“下层连期望速度都执行错误”；后者阻塞。

原 Z 的速度、漂移、高度门槛及时间窗应从当前保留的评价实现读取。不得由本需求另造一个“差不多就通过”的阈值。

若固定速度控制在启动后已稳定，但未恢复先前的位置偏差，需明确这是初始状态问题还是缺少位置恢复职责；不得擅自加位置外环、向 actor 输入位置，或删掉开始几秒来满足原门槛。

---

## 8. 阶段 E：准备下一轮，而不是提前开始下一轮

### E1. 首选短程设计是条件式候选

只有控制基线和实现检查支持时，才冻结下面的下一轮方案：[S11、S13、S14]

| 设置 | P0 | PLR |
|---|---:|---:|
| actor LR | 3e-4 | 1e-4 |
| critic LR | 3e-4 | 3e-4 |
| 环境、奖励、网络、控制基线 | 同一修复后候选 | 与 P0 相同 |
| 训练内容 | 原前 1,024 policy steps 的零命令 warmup | 相同 |
| env / rollout / minibatch / epoch | 32 / 16 / 64 / 1 | 相同 |
| 更新 / training transitions | 64 / 32,768 | 64 / 32,768 |
| 已有保存节点 | init0000、u1、u2、u8、u16、u32、u64 | 相同 |
| 开发评估 | u16/u64 的原 Z | 相同 |

两组训练合计 65,536 transitions；上述四次 Z 开发评估合计 25,600 nontraining transitions。这些是**下一轮候选预算，不属于本 Goal 的运行额度**。

不能同时加入高度奖励、KL early-stop、网络重初始化方案、观测归一化、rollout 改长等新变量。降低 LR 是待检验的假设，不是本 Goal 已经证明有效的修复。

如果发现真实 PPO 数据实现错误，先修错误，再定义共享修复基线；不能把修复收益归因给 LR。若诊断明确指向目标／反馈而非更新幅度，可以提出不同的单变量 pilot，但本 Goal 只交付其准确方案和所需方法决策，不顺手执行。

### E2. 不再要求一个逻辑上无法先完成的条件

`FirstUpdateDiagnostic` 的真实 rollout gate 必须在**下次训练的第一次参数更新之前**验证。因此准入包允许它处于：

```text
implemented_and_cpu_verified: true
live_first_rollout_gate: PENDING_NEXT_PILOT
first_optimizer_step_requires_gate: true
```

这不是“假装运行通过”。本 Goal 的完成条件是门槛可靠实现、基线和入口通过验证；下次开始采样后，若门槛失败，必须零参数更新停止。

不得在本 Goal 中偷偷做一次 optimizer step 再称为 nontraining diagnostic。独立进程、不同文件名或“只是 smoke”都不改变它属于训练更新的事实。

### E3. 最终交接内容

下一轮实际可用的命令和 manifest 必须由当前 CLI／runner 契约导出，包括 cwd、真实 argv、mode、group、初始化、seed、配置、源码／工具／输入哈希、输出、资源上限、停止和清理。

未签发的训练 authorization 要明确为未签发，不能放一个看似准确的假批准文件。可以输出 proposal 模板和机械生成它的步骤，但不能把缺少训练批准与技术未就绪混为一谈。

---

## 9. 独立验收与 Goal 完成定义

### 9.1 完成清单

以下全部成立才输出成功终态：

- [ ] 当前代码、工具、数据与任务身份明确，旧的失败／模板／未验收状态未被改写。
- [ ] 与当前候选相关的独立审核有效，所需真实证据可读取；没有未关闭的阻塞性 MUST_FIX。
- [ ] 诊断实际执行并产出足够证据，或确实复用了等价且可核验的已有真实证据；不是只有 CPU 测试。
- [ ] 当前控制基线的持续垂直与水平异常已修复并验证，或被明确区分为不阻塞此次 pilot 的研究问题；不存在靠模糊“影响不大”豁免的物理缺陷。
- [ ] 每一项产品修复均有问题证据、最小差异、验证结果和残余限制。
- [ ] 首次更新诊断的字段、分布、索引、RNG、hook 和无副作用测试成立。
- [ ] P0/PLR 或经说明的替代单变量方案有共享基线、比较指标和完整入口；未来首 rollout gate 强制开启。
- [ ] 实際资源与所有权清理已结算，所有训练 transitions、optimizer updates、训练 checkpoint writes 为零。
- [ ] 当前项目状态和任务入口已同步到准确的准入状态；独立审查没有被 Builder 自评代替。

### 9.2 终态与阻塞

```text
成功：READY_FOR_PAIRED_PILOT_WITH_FIRST_ROLLOUT_GATE
未成功：BLOCKED_EXTERNAL_AUTHORIZATION
        BLOCKED_CONTROL_BASELINE
        BLOCKED_IMPLEMENTATION
        DIAGNOSIS_INCONCLUSIVE
        BUDGET_EXHAUSTED
```

未成功状态也要提交已完成工作和唯一最小下一动作，但不得标记 Goal DONE。`CPU_READY`、`PROPOSAL_PREPARED`、`LIVE_EXITED`、`TESTS_PASS` 只是中间状态。

允许仍有非阻塞问题，不要求所有 H1–H9 均完成因果证明。阻塞判断必须写清“这个未知为什么会使下一轮试验不可信”，而不是遇到任意 UNKNOWN 就终止。

### 9.3 避免再次陷入无限流程修复

每一轮局部修复都应关闭一个具体 finding、增加一份关键证据或排除一个解释。没有这些增量时，不新增 successor 名称、分支、文档或重复角色调用。

完成一个阶段后，Supervisor 自动检查下一阶段前置条件并推进。不等待用户重复输入“继续”。真正需要用户参与的事项限于扩大资源／权限、依赖安装、变更方法或 actor 边界，以及无法自行解决的外部资源缺口。

---

## 10. 建议的新授权与资源包络

**这一节是本任务的新预算建议，不继承历史额度。仅当用户实际发送采纳该范围的启动 prompt 后，才可由当前授权机制登记。**

### 10.1 有界运行预算

| 资源 | 单次 live 上限 | 本 Goal 聚合上限 |
|---|---:|---:|
| live process starts | 1 | 4 |
| simulator initializations | 1 | 4 |
| nontraining transitions | 20,160 | 80,640 |
| 完整 enhanced 的 lane-physics rows | 403,200 | 按实际条件分别核对，不用 policy 行冒充 |
| outer wall（含清理） | 910 s | 3,640 s |
| runner GPU allowance | 910 s | 3,640 s |
| 诊断输出 | 16 GiB | 64 GiB |
| training transitions | 0 | 0 |
| optimizer updates | 0 | 0 |
| 训练 checkpoint writes | 0 | 0 |
| automatic blind retry | 0 | 0 |

以上是预算上限，不是耗时承诺或预分配目标。磁盘和 coexistence 必须在每次启动前检查。初始化 ticks、reset 上限和 GPU 探测次数按实际 `required_mode_budget`／runner 路径在首次 live 前补齐并固定；**这些未明确的字段不能视为无限额度，未齐全不得启动**。

建议至多使用：一份缺失的原基线 enhanced 诊断、一份修复后 B0 回归、一份确有必要的 D7-R／其他已明确对照，以及一个修复后新候选验证预留。已有结果有效则省掉相应运行，不为凑满预算消耗资源。

每次 live 失败后必须先结清资源、保存失败和分析原因。只有确有新修复、提交和独立审核的新候选，才可使用剩余包络申请准确新实验；这不是对同一失败输入自动重试。不确定是否仍运行时绝不重发。

每次的精确 proposal、身份、预算仍由现有授权链核对。旧 job 的无重试约束、STOP 和已耗尽预算不被本文件改写；新任务只是另行分配明确的新额度。

### 10.2 角色与自主推进

建议本 Goal 新角色调用上限 24 次，包含外部 PI／Builder／Evaluator 的实际调用；Supervisor 的普通文件读取和 CPU 测试不冒充角色调用。该数字是新建议，不是旧任务余额。

至少预留 6 次给关键预审、运行后分析和最终独立验收。先修本地 schema、路径和证据索引再调用审阅，避免为同一个格式问题消耗整段预算。不得为了降低计数把同样任务换名字不记账。

任务内普通实现、测试、文档和配置修正可以自主迭代；不设置“每失败一次就重新找用户批准”的机械规则。超出包络或变更保护边界才要求一次明确扩展。

### 10.3 授权与权限并不等于关闭安全检查

用户的启动 prompt 可授权 Commander 在本包络内组织和批准准确任务；Builder 不能自我审批，仍需要独立审核。平台权限、账号能力、runtime 保护和 STOP 不能由文字绕过。

仅修改项目内必要的工具兼容配置；安装、降级或升级系统／全局 CLI 和冻结运行环境不在默认授权内。若确实依赖这种操作，单列最小申请，不停止其他可做工作。

---

## 11. 最少交付物与续接方式

建议新输出位于 `docs/diagnostics/r5-readiness/`。这些是新增路径；可复用项目当前等价目录，但必须只有一个最新索引，不再制造多套平行状态。

| 文件／内容 | 必須回答 |
|---|---|
| `STATE_AND_EVIDENCE.md` | 从哪个 commit 开始，哪些旧工作已完成，使用哪些准确证据，哪些模板不可当结果 |
| `CONTROL_BASELINE_DIAGNOSIS.md` + JSON | 垂直／水平解释、原始统计、可核验机制、反证、结论范围 |
| `FIX_AND_REGRESSION.md` | 改了什么、没有改什么、测试与真实回归、失败和残余风险 |
| `PILOT_START_READINESS.md` + `pilot_launch_manifest.json` | 技术准入结论、共享基线、未来 gate、两组差异、启动过程、尚未签发的训练权限 |
| 独立 review 与运行记录索引 | 审核对象、finding 关闭、job／evidence ID、资源、退出与清理 |

大体积 raw、checkpoint 和内部敏感信息不因写报告而提交到公开 GitHub；保留在现有证据存储并以 manifest 引用。审核 sandbox 要实际能读到被引用证据，不能只存一个公网不可达的主机路径。

同时准确更新：

```text
docs/PROJECT_STATUS.md
coordination/PROJECT_STATE.json
coordination/CURRENT_TASK.md
```

沿用现有 task／ledger 机制，不开发新的调度系统。恢复会话时，先读当前阶段、准确候选、已消耗预算、active／uncertain job 和唯一下一动作；不要重跑整个需求。

最终给用户的简明总结必须包含：

1. 现在真正确认的问题是什么，修复了什么，仍不知道什么。
2. 哪些结果来自实际仿真，哪些只是 CPU／源码检查。
3. 下一轮是否技术可启动，以及为何；未启动训练的事实。
4. 下一次应测试哪个主变量、采用哪一组共同基线。
5. 剩余阻塞及一个明确的恢复动作；没有阻塞时给下一轮启动包的准确位置。

---

## 12. Codex 的执行规则摘要

```text
先续接，再补缺；不是重新做全仓调查。
先修测量工具，再用其结论修物理／训练实现。
先保留未修基线证据，再在新候选上验证修复。
把 enhanced 零目标 B0、passthrough B0 和 D7-R 区分命名。
已证明的轴向问题可以单独修，但不能冒充全部问题已解。
诊断分数是线索；模型／数值／回归证据才支撑机制判断。
普通代码缺陷在本 Goal 内闭合，不每一步找用户索要新 prompt。
方法改变、保护边界和资源扩展要明确决策，不绕过权限。
首 rollout gate 是下次首次更新前的闸门，不要求它在零训练任务里已实跑通过。
完成是可审计的训练准入；不以写完文档或测试通过代替真实验证。
```

---

## 13. 已审查来源索引

所有仓库来源固定到 `ebc27e2e47d4190efccac9a9179d4e291bd933da`；路径后注明主要用途。本文对当前源码的新增判断集中在第 4.4、5.1 节；历史物理数值来自 [S06]，没有在本文编制环境重算原始轨迹。

| 标记 | 仓库路径／官方资料 |
|---|---|
| S01 | `AGENTS.md`：当前工作目录、main、保护与权限边界 |
| S02 | `docs/PROJECT_STATUS.md`：当前阶段、迁移测试、CLI 不一致和剩余问题 |
| S03 | `coordination/protocols/SINGLE_BRANCH_WORKFLOW.md`：单写者、实验快照、角色与验收 |
| S04 | `coordination/PROJECT_STATE.json`、`ACTIVE_WORKFLOW.json`、`CURRENT_TASK.md`：机器状态和当前授权 |
| S05 | `docs/history/consolidation-20261012/r5-live-failure/B0_ENHANCED_LIVE_ROOT_CAUSE_REPORT.md`：9 月 28 日真实失败与 0 transitions |
| S06 | `docs/history/consolidation-20261012/r5/task_inputs/prior/ZERO_COMMAND_BASELINE_FIX_AND_VALIDATION_REPORT.md`：历史 B0、质量预测及水平未决 |
| S07 | `docs/history/consolidation-20261012/r5/r5_delivery/R5_CONTROL_CHAIN_DIAGNOSIS.md`：当前仅为 NOT_STARTED 模板 |
| S08 | `docs/history/consolidation-20261012/r5/b0_enhanced_delivery/InstinctRL_R5_Three_Stage_RLFlow_Prompts.md`：历史三阶段研究设计；其中旧分支／授权规则不覆盖当前流程 |
| S09 | `isaaclab-training/instinctlab/assets/taslab_uav/config.py`：当前 mass 与 gain 配置 |
| S10 | `isaaclab-training/instinctlab/control/lee_velocity_controller.py::taslab_velocity_control`：实际速度控制及质量补偿来源 |
| S11 | `training_workloads/unified_h1/pilot_diagnostic.py`：现有 first-update gate、probe、保存路径及错误 xyz 字段 |
| S12 | `training_workloads/unified_h1/b0_causal_analyzer.py`：机制重建、分段与启发式判别 |
| S13 | `tests/training_workloads/test_paired_pilot_unittest.py`：已有配对设置、7 检查点、测试范围及 mock 限制 |
| S14 | `training_workloads/unified_h1/candidate_config.json`：统一目录、执行 mode、零 runtime 权限、来源绑定 |
| S15 | `training_workloads/unified_h1/b0_observer.py`：支持的 post-physics protocol、计账和 enhanced 零目标语义 |
| S16 | 原附件 `R4_REVIEW_ADDENDUM.md`、`r4_review_e0.json`：E0 的历史结论；不替代当前工作区状态 |
| S17 | OpenAI 官方 Codex CLI slash commands：`/goal` 的设定、查看、暂停／恢复及 4,000 字符限制 |

固定仓库浏览入口：

```text
https://github.com/Printeger/instinctRL/tree/ebc27e2e47d4190efccac9a9179d4e291bd933da
https://developers.openai.com/codex/cli/slash-commands
```

**本文没有修改 GitHub 仓库、签发运行许可、执行仿真或开始训练。**
