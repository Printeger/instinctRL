# InstinctRL — R5-W 安全能力训练闭环 Goal 需求书

**任务类型：** 一个自执行研究／工程 Goal，内部多阶段自动衔接，而非多个需要用户逐次启动的 Goal  
**依据日期：** 2026-10-19  
**仓库：** `Printeger/instinctRL`；2026-10-19 读取远端 `main=f4f759f4c38d0fd1e40d03e46392b05c5415cb71`。启动时复核实际 HEAD 与远端，优先复用更新的有效证据，不覆盖历史。  
**冻结科学成果：** `docs/experiments/r5-formal-first-segment-20261015/FINAL_ANALYSIS.md`；原训练身份 `R5-FORMAL-MEAN10-FIRST-SEGMENT-01`；原产品/方法源码为 `feb1e96f97e44c7d872cec44268fc2d8b352bc93`、config SHA `d263cdefa038ecf044e3679ffe470c5f2229e5768bb322e591aa05f0a8710d24`。  
**本文件为新任务的需求与推荐主授权包，不表示当前会话已经在 Commander/Store 签发机器审批或执行训练。**

## 0. 目标与必须纠正的术语

**正式训练已经进行并完成了第一段**，不是“还没正式训练”。R0/R1 均完成 u257–u512 的新增 256 次更新，各自完成五个节点的 Z/T/S/X/W 评估。所有节点 Z 为32/32；u384、u512两组的 T/S/X均32/32；全部节点 W 0/32。独立 Reviewer PASS 接受实验完整性，**不代表五任务全部性能合格**。

本 Goal 的唯一最终科学问题是：**利用已经合格的悬停、跟踪、停止和反向控制基础，使 W 在原始 0.8 m 安全距离与 3 s 停车确认门槛下通过，且不遗忘 Z/T/S/X。**

不要重新从零开发 PPO／基础控制；不要先做单独的“全面 W 分析 Goal”。所有诊断、方法选择、CPU实现与独立测试、真实训练、五任务复评、必要的单因素迭代、最终审核都置于**本 Goal**，在预委托总范围内自动衔接。普通阶段性失败不是等待用户指令的理由。既存 `STOP`、身份/权限/所有权或硬资源失败仍须 fail-closed。

## 1. 已确认的事实与当前最有价值的假设

### 1.1 正式结果（R0/R1 在 u512）

| 指标 | R0 | R1 | 说明 |
| --- | ---: | ---: | --- |
| Z/T/S/X | 各32/32 | 各32/32 | 原评分完整通过 |
| W | 0/32 | 0/32 | 两墙子组各0/16 |
| W最小几何安全距离：32 lanes 平均 | 0.72514 m | 0.71360 m | 原要求 ≥0.80 m |
| W停车确认时间：32 lanes 平均 | 3.300 s | 3.115625 s | 原要求 ≤3.00 s |
| W高度／碰撞 | 高度合格、无碰撞 | 高度合格、无碰撞 | **无碰撞不等于满足0.8m安全距离** |

全部 W 轨迹证据合法、无reset中断，β最低达到0；u512两组均发生至少11步向墙速度≥0.1m/s的有效接近，不能简单称“从未向墙运动”。

### 1.2 训练覆盖的已确认缺口

此次 u257–u512 采用 **mixed-center**：32 lanes，16零命令/16非零命令，中心重置（禁用了近墙分布）；危险门 `q=1` 的**训练样本数为零**。因此这段训练证明开阔区域控制有效，不能证明 RL 具有近墙提前避险的学习经验。**再重复同分布的256更新不构成有效近墙假设检验。**

### 1.3 ICS 存在应量化的制动时机风险（推论，不是确认根因）

当前 `isaaclab-training/instinctlab/control/ics.py` 的 `ICSConfig` 默认 `d_safe_m=0.8`，`a_max_m_s2=2.0`，`latency_s=0`，`clearance_margin_m=0`，`range_rate_filter=False`；`attenuate_command` 根据 governor **期望速度**及当前可靠 LiDAR 射线计算 scalar β 与 `v_final=β v_gov`。它不是直接依据实际速度产生一个可以瞬间停止机体的反向推力约束。

公式 `v_safe=sqrt(2*a_max*max(d_eff-d_safe,0))`。条件性算例：沿某可靠射线的接近速度恰为0.2m/s，且有效射线距离与评估墙面距离近似相等，则对该射线 `β<1` 的条件约为 `d_eff<0.81m`。实际射线斜率、坐标和期望/实际速度不同，因此**不能当作真实触发距离**，但表明 `d_safe=0.8` 与**实测几何下限≥0.8**共用相同数字，并不保证存在动态停止裕度。执行器惯性、离散响应和观测延迟需逐步测量。记录 `q`、实际 β 的首次降低至<1/归零时刻，必要时记录 actor→governor→ICS→Lee/rotor 时序。

**目标假设树：** H1 近墙状态未训练导致策略不能提前减速；H2 安全层有效裕度/实际响应不足；H3 测距与正式 W 几何口径或时序差异；H4 以上共同作用。任务内基于证据选择训练/控制改动，不能预设 H1 唯一正确。

## 2. 本 Goal 的唯一外部审批——一次总体主授权

建议用户在启动本 Goal 时**一次性签署**本范围：授权 Coordinator/Commander 为完成 W 安全闭环，执行离线分析、代码变更与验证、自动选择/冻结必要的候选方法、分配阶段资源、提交有界真实训练和评估、独立审核、提交/tag/推送。小的内部审批和方法调整由受托控制面自主办理，不逐阶段回问用户。

总**新资源包络建议（上限，非任务保证或预计消耗）**：

| 资源 | 本 Goal 建议新增硬上限 |
| --- | ---: |
| 新训练 transitions | **1,048,576** |
| 新非训练 transitions | **659,200**（20个完整五任务checkpoint评估等量） |
| GPU/仿真进程启动与初始化 | 各不超过 **24** |
| 新逻辑 PPO 更新 | **2,048**（512 transitions/update） |
| 训练 checkpoint writes | **不超过40** |
| 新累计GPU分配秒／进程wall秒含清理 | **各不超过86,400s** |
| 新实验输出及保守预留 | **≤128GiB**，且磁盘操作始终满足现有≥30GiB空余地板和各入口磁盘上限 |
| 自动盲重试 | **0** |

若项目现有 master/store 硬上限小于上表，必须先在本 Goal 所授权的**总上限内**建立新的精确 delegated master、stage 和逐作业授权对象。**不直接重写旧 master/审批/receipt、不得绕过 Store、STOP、coexistence 和真实所有权检查。** 所有历史 INCOMPLETE/null 与 full reservation 原样保留，未FINAL的逻辑预留不可擅自退款或消失。超出总上限需要停止并记录 `STOP_RESOURCE_EXHAUSTED`，不能无限扩额。

每个新实验/方法都须提交新 source/config/input/工具 SHA、对应独立预审、新 experiment ID/readonly snapshot 和正常授权记录；这都是**自动内部步骤**，不要求每次向用户重复索取宽泛许可。仅在无法自动合法完成总授权绑定、异常 STOP、进程所有权不明、证据损坏/硬风险时暂停。

## 3. 工作流：一个 Goal 内的自动分段，不拆分为多个 Goal

### 阶段 A — 小范围现有 W 证据诊断（CPU，完成后自动进入 B）

直接读取：

- `docs/experiments/r5-formal-first-segment-20261015/FINAL_ANALYSIS.md` 与 `EXPERIMENT_RESULTS.json`；
- R0/R1 原始 `evaluation_raw` W 时序（必要时按现有 inventory 定位）；
- 原评估器 `training_workloads/unified_h1/evaluation.py` 的 `sensor_origin_surface_distance`、W trigger（几何≤1.1m）、stop confirmation；
- `isaaclab-training/instinctlab/control/ics.py`、`pipeline.py`、`training_workloads/unified_h1/schedule.py` 及训练 `q=0` 统计；
- 冻结 u512 checkpoint、方法及授权身份。

用原始实际曲线，对 x/y 两墙每 lane 列出：T0几何距离、首次 `d≤1.1`、首次 `q=1`、首次 `β<1`、首次 `β=0`、首次 `d<0.8`、最低距离时刻、停止确认/超过3s时刻、接近/反弹速度、actor residual/ICS输出（缺失记UNKNOWN）、可重建的世界系法向速度；跨坐标/时刻不直接相减。明确 ray clearance 与**原评估 sensor-origin→finite-surface 几何**不同。检查是否 `β=0` 已经越界、`β=0`后实际速度尚未归零、q 与 β 前后关系，以及 x/y/早晚节点一致性。

输出 `W_CAUSAL_DIAGNOSIS.md/.json`；在**同一 Goal 内**作单一主决策：

- **CONTROL_MARGIN_LIMITATION**：β限速太晚／停止裕度不足有证据；优先设计动力学有依据的控制器裕度修正与有限安全验证，绝不能只改评分阈值；
- **COVERAGE_POLICY_LIMITATION**：物理控制可以及时停止，但策略未受危险样本训练；优先近墙课程与安全目标训练；
- **MIXED**：必须既恢复可靠的安全基线又补训练危险样本；
- **INSUFFICIENT_EVIDENCE**：只补决定分支的最小信号/对照，不能重新进行所有 R4 全链审计。

这是很小的阶段，不需要把“写完诊断报告”作为总 Goal 的终点。

### 阶段 B — 有根据的实现与测试

推荐首先重建/改造近墙课程（不是“沿用中心课程盲增updates”），但是否与 ICS 修正联用必须由阶段 A 的证据决定并分别标识工程控制变量／学习方法变量。

- 以 R1/u512 作为**优先的能力保留起点**（Z/T/S/X全通过，R1晚期跟踪较好），但明确 R1 的 W 距离比 R0 更差；R0/u512保留为历史对照或必要基准。
- 真实起点位置与checkpoint SHA，从被接受的原件复哈希取得。已绑定的 R1 u512 SHA 是 `a3b9f9115268270ec000750035085b03980de2aed31987b7b08ae97bf26d8776`；源 checkpoint 参考原 R1 evaluation AUTH 的五节点列表；不能使用不一致来源。
- 新 profile 允许中心+near_x_wall+near_y_wall，包含靠墙、平行墙、背离墙与零命令保持；保持有足够的中心场景回放，防止 Z/T/S/X 灾难性遗忘。W 的 x/y 距离、命令强度和时钟需覆盖真实测试，不许偷偷修改 W 的测试初态或固定验证种子。
- 实际训练必须记录墙/中心各条件，危险 `q=1` 的有效样本数与比例、ICS干预、奖励和按方向安全数据。`q=0` 不可依然宣称“训练了危险避障”。
- 初次实验尽量只改**训练状态分布/课程**，PPO、actor、mean-prior10、residual reward .05、anchor4、ICS、Lee 和 rotor 不动。若阶段 A 证实ICS的安全裕度不足，则优先用独立、冻结的最小工程控制修正和固定策略控制回归验证，再在修正后的**共同基线**进行策略训练／比较，不能把控制修复结果算作RL学习贡献。
- actor 不加仿真真值/世界位姿/地图/特权速度；新增安全逻辑须在已批准测量/状态边界内。若安全层修改，以真实时序验证传感器、世界/机体系、执行器延迟以及命令保存；为部署明确其速度反馈依赖。
- 不允许通过使无人机从起点完全不动来通过W：原W运动前提和接近触发/停车确认必须同时达标。

单元、负例、回归测试必须覆盖 zero/nearwall/center 课程身份、reset、传感器、danger q、0.8几何与 β/制动、旧Z/T/S/X 不退化、训练checkpoint/snapshot/restart、首次512 gate和预算。逐环节测试后**自动推进实验提交**，不另请用户启动“下一目标”。

### 阶段 C — 有界训练与真实评估（无模型忙等）

执行器参考既有本地事件驱动机制：`coordination/batches/R5_FORMAL_FIRST_SEGMENT_20261015/README.md`。使用 tmux/systemd 中的普通 Python 进程轮询**唯一准确作业句柄及所属进程**；等待期间不用 Codex 定时轮询，不消耗模型推理 token。每个真实作业结束并完成原 receipt/owned ACK/settle/postrun 后才唤醒一次独立 Codex 完成分析与下一步冻结，再交给本地控制器执行。

默认分配：首候选从 R1/u512 进行**新的合法 warm-start 方法实验**。建议有限两个预登记后段节点，例如相对起点 +128、+256 updates，方法与命令/墙分布冻结。按原 W+Z/T/S/X 全部32 lanes进行评价；禁止只挑最好lane、隐藏失败、改变安全距离/3s阈值或把无碰撞当达标。首512真实样本 gate 在两个 optimizer 前通过；保存所有checkpoint、原始样本和分类计账。

- 若 W 与 Z/T/S/X **同节点全部通过原门槛**，冻结该方案并进行至少一次新的独立种子或独立初态验证；结束前单独报告跨种子能力与未知部分。
- 若 W 逐步改善但未达标，可在总包络内安排事前固定的等量追加段，仍保留中心样本并跟踪 Z/T/S/X，不必询问用户。
- 若 W 不改善，或安全触发/制动明显错位，按 A 证据转下一项预先明确的**单因素**控制/奖励/课程试验；最多两个科学候选分支，禁止随机超参网格搜索或见一个失败就叠加多个改动。
- 若 Z/T/S/X 回退，停止该候选继续长训，切回原已冻结 u512 并自主设计必要的小范围修正，不将回退 checkpoint 宣称合格。
- **任何阶段不得不经新的已登记实验身份就改参数续跑**；内部授权自动进行，旧日志不改写。

### 阶段 D — 完成与验收

只有真正满足原 `metric-h1-v1` 所有要求（含 Z/T/S/X/W、墙两侧各16/16、完整运动先决条件、原几何 min_distance≥.8、3s 停车、有效证据）且有独立审核，才可写：

`R5_ALL_ORIGINAL_GATES_PASS`。

交付：完整仓库可复现训练配置、原始 checkpoints/身份SHA、方法与控制diff、真实验收 JSON、按任务/方向/墙的训练和评估表、W时序图、独立审核、资源审计、如适用的跨种子结果，以及最终可用的模型身份/限制说明。`R5_ALL_ORIGINAL_GATES_PASS` 仍不等于硬件飞行安全保证或广泛泛化证明。

若总任务包络内无任何方案通过，状态是 `SCIENTIFIC_NEGATIVE_WITH_COMPLETE_EVIDENCE`（科学实验完整、性能未达标），附差距和下一项有依据假设；不能为了让Goal“成功”造数据、只改评分或无限加预算。

异常STOP/所有权不明/权限不成立/不可恢复损坏或硬资源封顶时，明确 `STOPPED_EXCEPTION` 或 `STOP_RESOURCE_EXHAUSTED`，保留证据与原状态，**不自动清除STOP、不盲重试**。

## 4. 避免再“一会一停”的具体约束

- **一个 Goal** 中 A→B→C→D 自动衔接；用户无需参与普通候选选择、阶段预算重配、独立预审或批后分析。
- 训练间隙不做模型 token 轮询，按 job 终态事件唤醒独立分析一次；保证单写者/Store串行和每一步准确审计。
- 可以在批准的大总包络内更换研究方向和重新冻结新候选，不能绕过原科学门槛或控制面守卫。
- 不创建又一个纯“研究建议报告”就停机，不因普通 `0/32`、negative result 或 Reviewer 非阻塞建议而中断整个 Goal；实质 MUST_FIX 在范围内修好后继续。
- 本目标**不承诺有限训练一定成功**；如达标自主闭环完成，否则科学负结果和停止原因也要真实交付。

## 5. Codex 最小交付索引

建议使用新目录 `docs/experiments/r5-w-closure-auto/`：

1. `GOAL_SCOPE_AND_MASTER_AUTHORITY.md` — 一次性委托、累计及阶段资源、STOP边界；
2. `W_CAUSAL_DIAGNOSIS.md` / `.json` — 现有证据 + 每 lane 时序，不猜根因；
3. `METHOD_DECISION_LOG.md` — 决策分支、新旧方法身份、预注册变量、seed及预算；
4. `TRAINING_AND_EVALUATION_MANIFEST.json` — 准确 readonly source、原 u512 SHA、训练课程与资源、评价节点；
5. `RUN_STATE.json` / `controller.log` — 本地零模型轮询及精确作业身份；
6. `FINAL_R5_W_RESULTS.md` / `.json` — 全任务原指标、科学结论；
7. `reviews/final-independent.json` / `EVIDENCE_INDEX.md` — 独立科研审核、保留所有失败及资源结算。

所有“原件”和大模型 checkpoint 可以继续留在本地不可变结果目录，通过inventory/hash引用；不得为提交 GitHub 覆盖或复制巨大文件。平常代码在 `main`、一次写者、提交之后 readonly 实验快照，不覆盖历史。正常 commit/tag/push，但要注意实验前输入版本锁定。

**完成定义是：完整五任务真实通过并保存可交付模型，或者在用户一次授权的硬包络内得出严格有效的最终负结论；不是再写一份分析建议然后结束。**
