# R6 本地直接训练闭环任务需求书（不使用 rlflow）

版本：2026-10-22 / v1。状态：**DESIGN_ONLY_NOT_STARTED**。

本文与 [启动 prompt](InstinctRL_R6_Standalone_Training_Prompt.md) 是本次文档交付；本次不启动训练、不改变旧 Goal 状态。用户以后提交启动 prompt 时，才授权按本契约执行。执行范围是接续 R6 科学目标，替换其运行与审核编排，不是重做 R6 研究或获得一份新的全额预算。

## 1. 任务目标及优先级

完成真实 R6 训练、模型评价、必要改进、最终确认和交付。把时间用于取得科学数据，不再先建全阶段自动化平台。

必须按以下顺序交付实际进展：

1. P1：最小直接入口可用，完成一次真实基线评估、一个有界 PPO 训练段及训练后评价。
2. P2：根据真实失败分布，完成必要修复、课程改进或预登记追加训练；保留全部候选与失败。
3. P3：冻结最终配方，执行剩余预算内预留的三随机流续训确认、最终保留集、独立科学核查及视频/模型交付。

P1 前不要求写完 P2/P3 的所有自动恢复分支、发布流程、录像编排或平台审计。P1 成功是里程碑，不等于整个 Goal 完成。代码行数、CPU 测试数量、控制器部署都不能代替真实训练结果。

本次明确要求实际 PPO 训练；即使 u640 基线在开发域已通过，也执行有依据的课程训练和能力保持对照，不以 `BASELINE_SUFFICIENT` 跳过训练。不为追求正结果无限训练，允许最终结论是无收益或退化。

## 2. 必读证据和基线

先读取最新状态与原件，避免重复已解决的问题：

- [项目状态](../PROJECT_STATUS.md)、[工作流](../../coordination/ACTIVE_WORKFLOW.json)、[单分支协议](../../coordination/protocols/SINGLE_BRANCH_WORKFLOW.md)。
- [原 R6 需求](InstinctRL_R6_Robustness_Autonomous_Goal_Requirements.md)、`docs/experiments/r6-robustness-auto/batch/state.json`、已生效的用户授权、预算补充、测试协议、split、训练配方、容量报告和最新审核。
- [R5 最终结果](../experiments/r5-w-closure-auto/FINAL_R5_W_RESULTS.md)、[训练评估 manifest](../experiments/r5-w-closure-auto/TRAINING_AND_EVALUATION_MANIFEST.json)、[独审](../experiments/r5-w-closure-auto/reviews/final-independent.json)。

冻结模型为 R5 R1/u640，SHA256 必须为：

`aaef7d40de50032cf2c61e0bc7a6e6f277a2f65e252284f3af58c0f25bae5429`

原 checkpoint：`coordination/research/attempts/R5-W-L1-NEARWALL-RECOVERY1-R1-RESEARCH-FORMAL-TRAINING-0c6363bbc0d2/results/research-formal-training/checkpoints/u640.pt`。

原训练 commit：`0c6363bbc0d2947a272f533c02e73d2995ac225a`；config SHA：`df96f98d0d55f997d20ace98468bfa4f4a6a614b1284d6f5b320ee34b9b2adbb`。基线 ICS clearance margin=0.15m，资产与方法沿原 manifest 准确核验。

文档编写时：旧记录报告真实 R6 基线/训练均0；累计196次自动唤醒、43次worker会话扣账/预留，模型总上限512，记录剩余273。旧输出及未释放预留28,529,459,200 bytes。**这些仅为当时读数，执行时重读账本，不硬编码为当前剩余。**

旧阻塞包括资源审核hash过期、会话身份绑定和独审模型连接失败。前两类属于旧编排的特定约束；网络故障也不直接证明本地 Isaac Sim/PPO 无法运行。真实科学/物理/数据缺陷仍须处理，不能将所有旧 MUST_FIX 一律忽略。

## 3. “不使用 rlflow”的明确含义

新入口、批处理、预算核查、过程日志、训练/评价运行和结果分析的**实际依赖链不得导入或调用 rlflow**：

- 不使用 Store/research.db 的新增写入、approval/release/submit/reconcile/settle、Commander 命令或旧 r6_controller 运行链。
- 不把 `tools/rlflow` 加入新执行 PYTHONPATH；不通过 subprocess 偷调旧入口；不换名称复制 Store/多层授权框架。
- 不用旧 review hash、旧聊天thread、Goal paused/idle状态、app-server RPC、MCP连接或模型路由作为训练发车条件。
- 不删除/卸载 rlflow；旧历史文件和账本保留只读，新批处理不得与旧控制器同时发车。

可以只读解析旧 JSON 中的科学配置、模型身份、有效结果、预算和审核意见。复用现有 Isaac Sim 环境、PPO、课程、传感器、checkpoint和评价代码；对纯工具函数的少量依赖做明确的本地实现或提取，不复制一整套调度库。

现有 `robustness_training_execution.run_loaded`、`robustness_training_method` 等接受旧 backend.authorization，本轮允许为新入口引入**本地运行契约**并提取共享的科学校验。不能构造假的 Store approval、伪造历史 AUTHORIZED/receipt，也不能 monkeypatch 授权函数永远返回true。旧入口行为保持不变；新入口明确记录 `execution_mode=local-direct`，不宣称通过旧 rlflow 执行认证。

验证需覆盖入口及其实际训练/评估/分析调用链：静态依赖检查配合CPU导入检查/真实run的加载模块记录，确认没有加载 `rlflow` 或启动其命令。仅新顶层脚本不写 `import rlflow` 不足以通过。

## 4. 新任务授权、审核和必须保留的保护

用户提交配套 prompt 后，本文优先于旧 R6 的“每作业必须经过 Store/独立模型预审/会话交接”的运行要求；仅替换运行编排，不放宽科学和安全边界。无需为移除这些旧流程再向用户逐项申请，也不需要旧控制器为退出自身依赖链签发许可。

新入口前置要求为可执行的本地检查：输入/资产/hash、实际预算、STOP、进程唯一性、设备/显存/磁盘、已知实质缺陷、first-rollout一致性和有限初始化/清理。模型在线不是这些检查的替代品。

独立审核保留在两处：有实质变化的科学方法，以及最终证据/结论。可按需调用独立只读agent，不为普通参数装载、目录、每次恢复和每个checkpoint再开模型会话。

初始复用既有科学方法的直接入口，可在定向CPU检查、实际非训练验证和明确的本地自审完成后启动首段训练；不等待新的模型预审。所有旧与新实质缺陷须列明已修复、已验证或不适用原因，不能带着数值/物理/数据错误发车。

模型服务暂不可用时，已冻结的训练和评估继续运行并保存结果；需要新方法判断或最终独审时记为待办，不冒充审核通过。最终独审缺失时可以报告“训练/评价完成，审核待补”，不能宣布完整科学验收完成。

保留以下硬边界：

- 开发仅在 `/home/mint/rl_dev/instinctRL-dev` 的 main；同目录单写者。运行输入先commit/tag/正常push，再从精确只读快照或detached HEAD运行。快照必要性不变，不新建任务分支、不force push。
- R5历史raw、报告、checkpoint及原生失败不改写；冻结runtime、正式旧源码保持只读，不安装/升级依赖或驱动。
- actor不得新增位姿、里程计、实际/显式估计速度、地图、障碍真值、env ID/课程时钟或simulator私有状态。原critic/reward/evaluation/logging特权边界不扩到actor。
- 原Z/T/S/X/W指标、分母、全方向/两墙、初始瞬态、运动前提、0.8m几何安全底线及原停止判据不降低。
- first-rollout概率gate在任一optimizer前通过；准确记录模型、Adam/RNG/counter/课程及fresh reset。不得靠先更新几个batch修补gate失败。
- 总预算、磁盘/显存余量、STOP、timeout和准确所属进程清理不取消。不得杀无关进程。

## 5. 最小本地结构

建议两个入口：一个负责单次 `baseline/train/evaluate`，一个负责串行批处理；名称和小模块拆分由实现者决定，不为凑文件数把代码堆成单文件。不要开发通用工作流语言、数据库或插件系统。

新输出根：`docs/experiments/r6-direct-training/`。每次运行一个不可覆盖目录，至少有：

- `RUN_INPUTS.json`：运行类型、checkpoint/source/config/资产/协议hash、seed、节点、上限、真实argv。
- `state.json`、`stdout.log`、`stderr.log`：PID与启动身份、当前阶段、进度、错误和最终退出码。
- `results/`：原始证据、计数、评价和checkpoint索引。

顶层只需 `RUN_PLAN.json`、`RESOURCE_LEDGER.json`、批处理状态和日志。机器状态采用临时文件+原子替换，详细日志追加，防止半写文件；提交intent与进程身份用于防重复，不发展成新的多层审批协议。

执行路径：

```text
输入与预算本地预检
  → 真实基线评估（同时验证新入口、保存与清理）
  → 首段PPO训练
  → checkpoint完整性检查与真实评价
  → CPU统计
  → 已登记的下一段，或确需模型判断时写待办
  → 必要迭代 → 最终确认/独审/交付
```

首次实验优先使用已设计并核验的R6条件和课程，默认从u640做相对+128/+256 updates的有界候选，实际节点须在运行前结合现有配方/预算冻结。绝对update与本次新增update分开记账，不把恢复的累计步数算成本次训练量。

原suite的新入口结果应与冻结R5做合理一致性检查；若解耦改变了物理/采样轨迹，定位原因并记录新运行身份，不能宣称逐位复现。非训练冒烟如果必要，必须有限且计预算；不以大量冒烟代替第一次基线。

## 6. 训练期间的模型与人工参与

批处理由普通Python在tmux或已有本地服务中运行。训练等待、CPU统计、预算检查和已登记分支均不调用模型。启动会话完成有界检查后交接并退出；不保持聊天Goal自动推理来等训练，不伪造平台Goal完成/暂停。

不再要求代码证明整个聊天平台绝对无token消耗，更不为此开发RPC/trace采集器。记录**本任务自己启动的**模型worker及调用原因/usage；平台外部活动若无可靠数据写UNKNOWN。实际训练等待不启动本任务模型worker。

可选事件触发模型worker只在批次结果需要新科学决定/具体修复时运行，退出后由本地程序继续。优先使用已有可用的本机CLI，不依赖不相关MCP/连接器；不得换账号、付费渠道或修改认证绕过服务限制。一次事件一次有效处理，失败先核对是否已产生副作用，不盲重试。

模型离线时继续已经登记的训练/评价队列；队列耗尽且需要新判断时保存 `WAITING_FOR_ANALYSIS`，保全数据，不修改方法、不无限唤醒。明确报告本机网络可用性是后续分析条件，不是已有本地作业的运行条件。需要外部恢复且确实无可继续工作时才说明具体阻塞。

常规任务内修复、提交、推送、训练与评估衔接自主完成，不逐阶段问用户。只有范围/预算改变、STOP、专属外部权限或不可确认的进程所有权需要用户决定。

## 7. 预算：接续而非清零

启动时读取旧有效用户授权、账本和预留，生成 `LEGACY_BUDGET_IMPORT.json`，记录原文件hash及逐项映射；旧账本只读。核对旧作业/控制器已终止且不再写账，避免双重发行资源。

新ledger记录 `历史已用及未释放预留 + 新已用及在途预留 ≤ 原总上限`。不能把旧 `used` 字段单独当总消耗；模型会话和输出等还有 `reservations`。保留旧最终确认预留，不借迁移退款；新作业预留结算在本地ledger一次完成，未知用量保守保留。

文档编写时有效总上限如下，执行时若有更晚真实授权须核对：

| 资源 | 沿用总上限，并扣除历史消耗/未释放预留 |
| --- | ---: |
| 训练transitions | 3,000,000 |
| 非训练transitions | 2,000,000 |
| GPU分配秒、live进程累计wall秒（含清理/失败） | 各43,200 |
| live启动、sim初始化 | 各64 |
| 训练checkpoint写入 | 96 |
| 主要科学候选 | 3 |
| 输出/快照及未释放预留 | 128GiB；磁盘空余至少30GiB |
| 资源排队 | 21,600秒 |
| 自动模型会话 | 512；不是本次再获得512次 |

迁移、检查、测试和运行都在原包络内；首段后仍须保有最终确认和交付额度。初次容量核对复用现有报告，重点更新剩余量，不重做一整套预算平台。

模型调用目标是少量、有实际作用的分析/审核，不把剩余273次视为应该花完。已有科学规则可由程序判断时直接执行。模型会话单独记录，不把次数当token数；接口没提供usage则写UNKNOWN。

资源准入复用最近接受的窗口参数与本地数值校验，作为新入口明确的配置，不要求旧queue_review批准新入口。保留完整采样、设备/driver、显存和其他进程约束；不恢复所有瞬时采样≤5%的永久阻断。发车前外部负载与运行中自身高GPU占用分开处理。持续繁忙有界退避，不能无限降低标准或无限占用等待。

无需为了沿用旧输出认证重新申请root配额卷；可使用现有获准目录/卷，只要本地写入/输出预算和磁盘地板可验证。不能关闭文件上限或删历史原件。

## 8. 研究范围、评价与改进

科学范围保持原R6：低速水平运动、停止保持、静态墙面；新起点/朝向/近墙距离/速度，以及分级测量噪声、延迟/有限缺帧和有限外力扰动。复用已冻结测试协议与split，不重新挑一套更容易通过的条件。

不包含穿门/走廊导航、动态障碍、实机、H2返回旧位置或新网络/算法族。允许在总授权内按证据修改课程/段长及相关奖励/优化/测量驱动控制参数，但预登记假设、主要变量、对照及停止规则。

- 先获得u640开发域的真实分条件数据，再定位实现、控制、覆盖或学习问题。
- 课程保留原任务混合及近墙暴露，记录实际center/x/y、命令方向、q-positive、ICS介入、噪声/扰动、终止和reset，不能只报告配置比例。
- 开发集用于调参；最终测试一次揭晓。既有holdout若已被读取或用于调参，如实标污染，不能改名称恢复独立身份。
- 改控制参数先用固定策略做对照；原u512加margin已通过的历史结论保持，不把控制收益归RL。
- 配方冻结后，从共同u640起点进行三条实际不同随机流的确认；明确只是续训稳定性，不是三次独立从零训练。保留原五任务回归、每seed及每条件结果，不用平均掩盖失败。
- 能力退化时停止无条件长训，先修复或换下一个有依据候选。已完成预定训练和验证但无收益，可以作为科学负结论，不冒充性能达标。

## 9. 有限修复与验收

只写直接支撑已实现路径的测试：预算不越界、原件不可覆盖、STOP/timeout清理、锁/恢复不重复提交、checkpoint与概率gate、统计/指标、无rlflow依赖。

不再以“所有未来异常自动修复都实现”作为首训条件。实际出现错误时保全证据、判断原因、修好局部路径、针对性回归、新attempt继续。失败槽和占用仍计账，不设固定失败次数后必须问用户，也不允许原参数盲重试。

原生退出码与科学证据完整性分别记录。已知app.close故障不自动等于科学数据无效，也不自动豁免：先验完整计数/raw/checkpoint和所属进程已清理，未知损坏不能放行。

同一run有可疑提交意图、PID复用或无receipt时先核对真实状态；无法确定不得重复启动。保留单写锁和稳定进程身份，绝不杀无关进程或自动清除STOP。

## 10. 必交结果与结束条件

新根 `docs/experiments/r6-direct-training/` 至少交付：

1. `EXECUTION_SCOPE.md`：生效用户委托、与旧编排的替代关系、工作流说明。
2. `LEGACY_BUDGET_IMPORT.json`、`RESOURCE_LEDGER.json`：历史继承、所有新消耗/预留和剩余量。
3. 本地直接入口、批处理、运行说明：准确start/status/stop/resume命令；不要在需求阶段伪填未来PID或job结果。
4. 基线和首训真实原件、全部run输入/进度/退出记录、checkpoint及hash、训练曲线、候选决策日志。
5. `TRAINING_AND_EVALUATION_MANIFEST.json`、`FINAL_R6_RESULTS.md/.json`：源码/资产/配置、所有seed/条件、原任务保持和真实收益/退化。
6. `NO_RLFLOW_AUDIT.json`、`MODEL_USAGE_AUDIT.json`、独立科学审核与需求审计：依赖链验证、等待期间本任务模型调用、证据完整性。
7. 简单的真实Isaac Sim可视化与 `videos/VIDEO_MANIFEST.json`：可用评估raw渲染，但明确回放不等于新动力学测试，视频不取代数值评价。

训练计数、优化器更新和新checkpoint必须非零且相互对应，不能把读取u640称为已训练。首段默认完成事前登记的128/256update节点；实际科学目标完成还需后续确认和评价。

最终区分：

- `TRAINING_COMPLETE_TARGET_PASS`：实际训练和完整必要验证完成，事前目标通过，独审接受。
- `TRAINING_COMPLETE_SCIENTIFIC_NEGATIVE`：已完成必要训练/评价，证据支持未达标/无收益，独审接受，不称能力成功。
- `TRAINING_EVAL_COMPLETE_REVIEW_PENDING`：训练和评价已完成，独审仍缺；可交付模型和数据，但整个Goal未完成。
- `INCOMPLETE_RESOURCE_LIMIT` / `INCOMPLETE_EXTERNAL_BLOCKER` / `STOPPED_BY_USER`：必要工作未完，明确缺口；不能以预算耗尽冒充科学阴性。

启动会话交接必须报告实际状态和本地查看路径，不把“后台启动”写成“训练完成”。最终确认所有任务进程已清理、历史R5不变、交付commit正常push；仅达到实际完成条件才标记Goal完成。
