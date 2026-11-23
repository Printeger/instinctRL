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
