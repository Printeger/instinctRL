# R4 后三轮研发在执行

任务 `instinctrl-post-r4-three`。五位原生伙伴首轮已返回，`report --require-full-team` 核验 6/6。独评协议和种子已提前冻结于 `4e77ddae` 与 tag `instinct-post-r4-three-protocol`。`REAR-01` 四臂关闭，正常来物换方向把横移中位从约 .020 提到 .203 m，接触仍 32/32。较慢来物配换向降至 24/32。`MECHANISM-R1` 三臂关闭，后向悬停可下发时将幅度增加到 .45 m/s，正常来物接触从 32/32 降至 24/32，前向、普通通行和静止与 .30 m/s 方向臂一致。

`SHORT-R1` 已完成 32 更新、16,384 条转换和同设备概率复算。`u656`、`u672` 都已核验。前两次协调入口因源码清单 JSON 表示不一致在发车前退出，均为零 GPU，已保留 `SHORT-R1-PREPARE-01/02.json`。有效源码 `244365f9`、快照 `~/instinctrl-runs/INSTINCT-POST-R4-THREE-SOURCE-05`，有效设计 `SHORT-R1/DESIGN-reissue-244365f9eeae.json`。`CHECK-SHORT-R1` 已关闭。u640/u656/u672 后向均 24/32，前飞接触 0/32→1/32→3/32。按 Rebel 的独立复算，先完成同种子、更低 actor 学习率的 `SHORT-R1-B` 短训与配对（源码 `fe24edaa`、快照 `INSTINCT-POST-R4-THREE-SOURCE-06`），再冻结正式 R1。正式长训名额尚未使用。 [项目计划](../docs/experiments/instinct-post-r4-three/PLAN.md) · [阶段输入与账本](../docs/experiments/instinct-post-r4-three/)

`SHORT-R1-B` 及三臂配对已关闭。32 次节点前飞接触 1/32、联合恢复 31/32、后向接触 24/32。R1 采用 `1e-5`，固定 `u640→u896`，2×128 更新、131,072 条转换，从同一 `fe24edaa` 快照发车。`FORMAL-R1` 正在执行，先等 `U768`、`U896` 的相同设备概率与严格续训审计，再等 `CHECK-FORMAL-R1`，然后再定 R2。 [正式设计](../docs/experiments/instinct-post-r4-three/FORMAL-R1.md)

后向回波的离线无静遮挡几何重放在 30° 逐条复得 32/32 的首次命中步，标称 25° 预测提早 6–9 步。它只决定后续低成本感测对照的候选角度，尚无闭环结果。 [射线复算](../docs/experiments/instinct-post-r4-three/MOUNT_GEOMETRY_REPLAY.json)

`FORMAL-R1` 和开发评价已关闭。实际 256 次更新、131,072 条转换、1,692.32 秒，终点 `u896` SHA `5e625e79931bae6c129101be4dc68a954a3a85961d30855617dd0bc7aa30f886`，第二段完整恢复逐字 RNG。开发前向接触 3→5/64，后向 24/32 不变，管道联合恢复 8→7/8，普通通行四布局都增加。剩余两个正式名额。`MECHANISM-R2` 以原 `u640` 和同 v2 动作测试 30/28/25° 标定，正在执行，冻结源码 `5e603968`、快照 `INSTINCT-POST-R4-THREE-SOURCE-07`。按三臂真实回波、接触、进度与静漂选第二轮短训。 [关闭训练复核](../docs/experiments/instinct-post-r4-three/FORMAL-R1-TRAINING-AUDIT.json) · [R2 研究问题](../docs/experiments/instinct-post-r4-three/ROUND2_DEVELOPMENT.md)

`MECHANISM-R2` 已关闭，30/28/25° 后向首回波中位第 41/39/35 步，接触 24/16/24（各 32）。选择 28° 作 `SHORT-R2`（同 u640、1e-5、32 更新、两节点；源码 `5e603968`），运行状态由已提交输入和 `~/instinctrl-runs/R6-DIRECT-INSTINCT-GPU-R5-RECOVERY-THREE-SHORT-R2-U672/` 回执负责。最终独评仍未运行。 [标定配对复算](../docs/experiments/instinct-post-r4-three/MECHANISM-R2-RECOMPUTED.json)

# R4 后管道恢复正式训练和独评完成，等待审查

任务 `instinctrl-r4-recovery-long`。五位原生伙伴首轮及综合核验 6/6。固定 `u640` 的三臂物理对照中，回切使八条管道前飞均无接触且恢复，恢复窗平均 0.743 m。后向配对前向 0/32、后向 32/32 接触。短 PPO 16、32、64 次完成，依据 64 次的接触和静态退化，正式固定每段 16 次目标、112 次回放。

正式源 `b40a2bb5`、冻结提交 `42efdb4c`、数据关闭提交 `525c6e5d`。八段合计 1,024 次更新、524,288 条训练及 16 份权重。终点 `u1664` SHA-256 `81a8294652d4374f9ee6e942f147876f303bf8802237db217a541e3c34d38b67`。三臂完整 R6 和四布局独评共 649,920 条转换。前向接触 64/64、3/64、2/64，后向均 32/32。终点街道恢复改善，管道联合恢复降至 7/8，静漂和 R6 跟踪门槛未达。全作业及协调进程已退出，保留旧合格产品基线与 R4 对照，停在唯一审查点。 [完整报告](../docs/experiments/instinct-r4-recovery/REPORT.md) · [交付复核](../docs/experiments/instinct-r4-recovery/COMPLETION_AUDIT.json)

# R3 后的侧让续训与独评已结束，等待审查

任务 `instinctrl-post-r3-long`。五位原生伙伴独立首轮和 Maverick 综合已核验 6/6。同权重 25 步方向保留使前向碰撞从 16/16 降到 0/16，后向仍 16/16。共享执行层短测把前飞逼近从 8/8 降到 3/8，普通通行和静止悬停无变化。四类场景预检完成 7,680 条 PhysX 转换并复算。完整候选从 u640 分四段新增 384 更新和 196,608 条转换，固定 u1024，随后执行全套 R6 和四类独立物理留出。 [冻结方案](../docs/experiments/instinct-post-r3/LONG_PLAN.md) · [贡献记录](contributions.jsonl)

实际完成四段 384 更新、196,608 条训练转换，终点权重为 u1024，另完成 R6 和四类留出共 262,720 条评价转换。终点和同选项起点的前向碰撞同为 5/64，后向均 32/32；终点普通通行与漂移退化，R6 四项扰动未满分。Scientist 核实两臂幸存的管道前飞在恢复窗全部 ICS 零下发，Producer 核实 14 份权重和 10 个关闭回执，所有进程已退出。冻结源码 `14dc9c0f`、输入 `81a82af0`、结果 `eca6d933`。保留旧产品基线，停止在唯一审查点。 [审查报告](../docs/experiments/instinct-post-r3/REPORT.md)

# 三轮全量 GPU 训练完成，等待审查

任务 `instinctrl-gpu-three`。三轮各完成 128 次 PPO 更新、65,536 条物理训练转换和 193,600 条 R6 评估转换。第二、三轮动态和视场配对均复算。第三轮恢复 21 个 R6 条件的 32/32，通过率恢复，最差配对跟踪仍超过旧合格模型的容差。第三轮在有逼近历史时形成可执行侧向动作，八类动态配对仍为 24/64 碰撞，视场配对仍为 32/32。保留旧合格模型作为基线，第三轮作为研究候选。五位原生伙伴与 Maverick 首轮核验 6/6，所有正式作业关闭，已停止在唯一审查点。 [审查报告](../docs/experiments/instinct-gpu-three/REPORT.md) · [贡献记录](contributions.jsonl)

# 五轮动态闭环训练完成，等待审查

任务 `instinctrl-five-development-cycles`，五位原生伙伴及 Maverick 首轮均记录。第 4 至 8 轮分别完成方向模仿、候选风险、联合动作与风险、重标策略轨迹和零指令威胁方向参考。共 113,541 条训练样本、9,488 次更新。第 8 轮开发 3/192 碰撞，独立确认 8/288，安全通行 96/96 且平均进度 3.230 m。保留横穿及管道内悬停失败，下一步解析失效后再做新的六自由度闭环。已核对输入、源版本、数据、模型和退出状态。 [研究报告](../docs/experiments/instinct-dynamic-five/REPORT.md) · [记录](contributions.jsonl)

# 第二、三轮正式评价完成，等待审查

任务 `instinctrl-next-two-cycles`。Rebel、Inventor、Hacker、Scientist、Producer 的原生首轮和 Maverick 综合均已记录，首轮核验 6/6。两轮均从 u640 完整训练到 u768，各评价 193,600 条转换，21 个条件各 32/32。第二轮以真实重置首帧修复延迟训练启动，配对护栏通过。第三轮加测距时序辅助，停令锚误差下降 3.67 mm，跟踪退化超限。固定扫描对照显示策略普遍降速，逼近选择性仍缺。六个作业按准确进程身份核对关闭。下步先做跨场景动态闭环，保留第二轮作为时序实验对照。 [正式审查点](../docs/experiments/instinct-next-two-cycles/REPORT.md) · [贡献账本](contributions.jsonl)

# 最近完成的任务

`third-training-anchor`，2026-11-22 至 2026-11-23。Maverick 主持。R6 开发评价的通过率均为 32/32，重算延迟加丢帧停令锚误差后得到 50.13 和 57.51 mm。以连续误差做独立保留集比较，正式测试三对同种子、同量续训和冻结 u640。候选只改锚奖励 4→8。正式锚误差相对两个对照分别增加 0.81 和 0.37 mm，均未达标。七模型全部硬护栏通过，非零跟踪、清晰停令和有效射线覆盖也通过。见 [结项报告](../docs/experiments/third-training/formal/REPORT.md)。

两段真实恢复预检 u640→u642→u646 与全部正式运行均已结算。系统盘普通目录的 30 GiB 守卫贯穿本地直接链，native 临时卷以普通用户权限复用。调度接续中的首次 CUDA 诊断错误保留，纠正 native 绑定后用同设备概率证据完成六条训练验收。证据和原件保留，主工作树原有证据格式编辑未参与运行。见 [正式冻结](../docs/experiments/third-training/formal/FREEZE.json) 和 [结项复核](../docs/experiments/third-training/formal/COMPLETION_AUDIT.json)。

# 历史初始化工作流

任务 `maverick-v11-init`，2026-11-22。上游 `Printeger/maverick-codex@9e1a07f8f9c654b2b45c2782291d190aa3bc8dac`，首次执行 init。Maverick 主持日常开发，按问题进展切换视角；交付时按实际贡献汇报。下一项开发任务可直接开始。项目研究状态以 `docs/PROJECT_STATUS.md` 为准。

## 静态验证

`python3 .agents/skills/maverick/scripts/maverick.py doctor --project .` 返回 PASS，项目知识 READY；跨学科地图目录与上游逐文件相同。项目级配置仅设置 agents 的启用与并发。

## 原生子 Agent 验证

PASS。在项目新开的 Codex CLI 0.171.0 会话 `01a20399-bd60-7102-858d-d5a09ba85612` 指定两个原生角色。已核对父子线程关系、运行时 agent_role、返回标记与仓库原文。

| 角色 | 实际子线程 ID | 返回标记 | 只读结果 |
| --- | --- | --- | --- |
| mav_hacker | 01a2039a-01d2-7671-afa0-9c0a03976de8 | 49c90f24be14 | `tools/r6_direct/analyze.py:44` 定义 main，第 74 行调用。 |
| mav_inventor | 01a2039a-14b6-7230-9afe-7eece3b19894 | f5e336b95f05 | `control-information.md:21` 的同观测机制解释 `FINAL_R6_RESULTS.md:5` 的天花板：成功计数无法区分策略余量。 |

验收转折：当前对话缺少原生角色选择字段，改在新项目会话验证角色绑定与返回值。贡献记录见 `.maverick/contributions.jsonl`。

## Maverick V1.2.0 升级验收

任务 `maverick-v120-upgrade` 从上游 `d3b3683919e8aac934237cbf47a62ef5f43db1ba` 执行 upgrade。旧版任务和原生验收事实保留。新任务先取得五份独立返回，恢复同一任务只补缺项。

项目新开的只读 CLI 父线程 `01a2072b-a4be-7292-a7aa-ff6420b7c13e` 实测 gpt-6.2-sol high。核对五个 native_subagent 的父线程、角色、模型、推理强度与带标记的最终返回。五者均为 gpt-6.2-sol。

- mav_rebel · high · 01a2072b-f36d-79f3-998d-b25d74d13a94 · R12-A7
- mav_inventor · high · 01a2072c-0e30-7ec1-aad1-b4bfaf60a1bb · I12-B8
- mav_hacker · medium · 01a2072c-a5d8-74e0-a1f4-7cae2d44ddbc · H12-C9
- mav_scientist · high · 01a2072c-bf59-7db1-b967-0b6cca793fdb · S12-D4
- mav_producer · medium · 01a2072d-4c83-7e61-b04d-af06fa049947 · P12-E5

五份原始返回及 Maverick 综合决定见本任务的 contributions.jsonl。doctor 为 PASS，知识 READY；report --require-full-team 已核对 6/6。原偏好为空，未发现保留人格冲突。
